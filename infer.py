import json
import logging
import os
import torch
import esm
from dataclasses import dataclass, field
from typing import Optional

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    HfArgumentParser,
)


from configuration import RippleLMConfig
from model import RippleLMForConditionalGeneration
from data import RippleLMDataset
from collate import RippleLMCollator
from utils import set_seed


set_seed(42)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default="output/merged_model",
        metadata={"help": "Path to the merged model or the base model with adapter."}
    )
    do_sample: bool = field(
        default=False, metadata={"help": "Enable sampling; otherwise greedy decoding."}
    )
    temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    top_k: int = field(default=50, metadata={"help": "Top-k filtering parameter."})
    top_p: float = field(
        default=0.95, metadata={"help": "Top-p (nucleus) filtering parameter."}
    )


@dataclass
class DataArguments:

    csv_path: str = field(metadata={"help": "Path to the inference data file (CSV)."})

    esm_model_name: str = field(
        default="./esm2_t12_35M_UR50D.pt",
        metadata={"help": "Local path to fair-esm .pt checkpoint for online embedding computation."},
    )

    output_path: str = field(
        default="output.jsonl", metadata={"help": "Path to save the output JSONL file."}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum sequence length."}
    )
    ephod_label_col: str = field(
        default="ephod_label",
        metadata={"help": "Optional label column for ephod in the CSV."},
    )
    pptstab_label_col: str = field(
        default="pptstab_label",
        metadata={"help": "Optional label column for pptstab in the CSV."},
    )


def load_model_for_inference(model_args: ModelArguments, data_args: DataArguments):

    logger.info(f"Loading tokenizer and model from {model_args.model_name_or_path}...")


    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    protein_token_id = int(tokenizer.convert_tokens_to_ids('<|protein_token|>'))
    thermo_token_id = int(tokenizer.convert_tokens_to_ids('<|thermo_token|>'))
    ph_token_id = int(tokenizer.convert_tokens_to_ids('<|ph_token|>'))


    base_llm = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    # Load frozen ESM model via fair-esm (local .pt checkpoint)
    logger.info(f"Loading ESM model from: {data_args.esm_model_name}")
    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_local(data_args.esm_model_name)
    esm_model = esm_model.to(torch.bfloat16)


    lm_config = base_llm.config
    config = RippleLMConfig(
        text_config=lm_config,
        protein_token='<|protein_token|>',
        protein_token_index=protein_token_id,
        thermo_token='<|thermo_token|>',
        thermo_token_index=thermo_token_id,
        ph_token='<|ph_token|>',
        ph_token_index=ph_token_id,
        enable_property_task=False,
        esm_model_name=data_args.esm_model_name,
    )

    model = RippleLMForConditionalGeneration(config, base_llm, esm_model=esm_model)

    mpf_path = os.path.join(model_args.model_name_or_path, "mpf.pth")
    if os.path.exists(mpf_path):
        logger.info(f"Loading projector weights from {mpf_path}")
        mpf_state_dict = torch.load(mpf_path, map_location="cpu")
        model.mpf.load_state_dict(mpf_state_dict)
        model.mpf.to(device="cuda",dtype=torch.bfloat16)
    else:
        logger.warning(
            f"Could not find mpf.pth at {mpf_path}. "
            "If using a trained model, ensure the projector weights are present."
        )

    thermo_head_path = os.path.join(model_args.model_name_or_path, "thermo_head.pth")
    if os.path.exists(thermo_head_path):
        logger.info(f"Loading thermo_head weights from {thermo_head_path}")
        thermo_head_state_dict = torch.load(thermo_head_path, map_location="cpu")
        model.thermo_head.load_state_dict(thermo_head_state_dict)
        model.thermo_head.to(device="cuda", dtype=torch.bfloat16)
    else:
        logger.warning(f"Could not find thermo_head.pth at {thermo_head_path}.")

    ph_head_path = os.path.join(model_args.model_name_or_path, "ph_head.pth")
    if os.path.exists(ph_head_path):
        logger.info(f"Loading ph_head weights from {ph_head_path}")
        ph_head_state_dict = torch.load(ph_head_path, map_location="cpu")
        model.ph_head.load_state_dict(ph_head_state_dict)
        model.ph_head.to(device="cuda", dtype=torch.bfloat16)
    else:
        logger.warning(f"Could not find ph_head.pth at {ph_head_path}.")

    model.esm_model.to("cuda")
    model.eval()

    return model, tokenizer, esm_alphabet, config

def _parse_generated_output(text: str, model_name: str) -> str:
    """Parses the raw generated text to extract the clean response."""
    if "llama2" in model_name.lower() or "llama-2" in model_name.lower() or "biomedgpt" in model_name.lower():
        return text.split("[/INST]")[-1].strip().split("</s>")[0]
    elif "llama3" in model_name.lower() or "llama-3" in model_name.lower():  # Assumes Llama-3 or similar chat format
        return (
            text.split("assistant")[-1].split("\n\n")[-1].strip().split("<|eot_id|>")[0]
        )
    elif 'dsllama' in model_name.lower():
        return text.split("<｜Assistant｜>")[-1].strip().split("<｜end▁of▁sentence｜>")[0]
    elif 'qwen' in model_name.lower():
        return text.split("<|im_start|>assistant")[-1].strip().split("<|im_end|>")[0]
    else:
        return text.strip()

def prepare_inference_dataset(data_args: DataArguments, protein_token, thermo_token, ph_token):

    dataset = RippleLMDataset(
        csv_path=data_args.csv_path,
        protein_token=protein_token,
        thermo_token=thermo_token,
        ph_token=ph_token,
        ephod_label_col=data_args.ephod_label_col,
        pptstab_label_col=data_args.pptstab_label_col,
    )
    return dataset

def run():

    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()


    model, tokenizer, esm_alphabet, config = load_model_for_inference(model_args, data_args)


    test_dataset = prepare_inference_dataset(
        data_args,
        config.protein_token,
        config.thermo_token,
        config.ph_token,
    )


    data_collator = RippleLMCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        esm_alphabet=esm_alphabet,
    )


    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=model_args.do_sample,
        top_k=model_args.top_k,
        top_p=model_args.top_p,
        temperature=model_args.temperature,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True
    )


    if os.path.exists(data_args.output_path):
        logger.warning(
            f"Output file {data_args.output_path} already exists. Appending to it."
        )


    for i, item in enumerate(tqdm(test_dataset, desc="Generating responses")):


        data = data_collator([item])

        prompt_mask = data["labels"] == -100
        data["input_ids"] = data["input_ids"][prompt_mask].unsqueeze(0)
        data["attention_mask"] = data["attention_mask"][prompt_mask].unsqueeze(0)


        data.pop("labels")
        data.pop("ephod_label", None)
        data.pop("pptstab_label", None)
        for k, v in data.items():
            if v is not None and isinstance(v, torch.Tensor):
                data[k] = v.to("cuda")


        with torch.no_grad():
            output_tokens = model.generate(**data, generation_config=generation_config)

        raw_gen_text = tokenizer.decode(output_tokens[0], skip_special_tokens=False)
        clean_gen_text = _parse_generated_output(raw_gen_text, model_args.model_name_or_path)
        ground_truth = item['target_text'].strip()


        with open(data_args.output_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "gen": clean_gen_text,
                    "gt": ground_truth,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
    logger.info(f"Inference complete. Results saved to {data_args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
