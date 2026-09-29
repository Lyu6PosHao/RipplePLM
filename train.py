import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import esm
import transformers
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import (
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig,
)
from configuration import RippleLMConfig

from data import RippleLMDataset
from model import RippleLMForConditionalGeneration
from utils import set_seed,print_trainable_parameters,smart_tokenizer_and_embedding_resize
from collate import RippleLMCollator
from text_metrics import compute_metrics_from_lists


set_seed(42)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default="test_model/model001",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    lora_r: Optional[int] = field(
        default=-1, metadata={"help": "LoRA attention dimension (rank)."}
    )
    lora_alpha: Optional[int] = field(
        default=None, metadata={"help": "The alpha parameter for LoRA scaling."}
    )
    lora_targets: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of module names to apply LoRA to."},
    )
    modules_to_save: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated list of module names to keep trainable without applying LoRA."
        },
    )
    merge_when_finished: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to merge the LoRA adapter into the base model upon completion."
        },
    )


@dataclass
class DataArguments:

    csv_path: str = field(metadata={"help": "Path to the training data file."})

    esm_model_name: str = field(
        default="./esm2_t12_35M_UR50D.pt",
        metadata={"help": "Local path to fair-esm .pt checkpoint for online embedding computation."},
    )

    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    ephod_label_col: str = field(
        default="ephod_label",
        metadata={"help": "Column name for ephod supervision label."},
    )
    pptstab_label_col: str = field(
        default="pptstab_label",
        metadata={"help": "Column name for pptstab supervision label."},
    )
    enable_property_task: bool = field(
        default=True,
        metadata={"help": "Whether to train thermo/ph token based property classification."},
    )
    val_csv_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to validation CSV for periodic generative evaluation during training."},
    )
    eval_max_samples: int = field(
        default=100,
        metadata={"help": "Maximum number of validation samples for generative evaluation."},
    )


def load_model_tokenizer_for_training(
    model_args: ModelArguments, data_args: DataArguments
) :

    mpf_exists=False
    if os.path.exists(os.path.join(model_args.model_name_or_path,'mpf.pth')):
        logger.warning(
            f"Found trained MPF checkpoint at {model_args.model_name_or_path}. Will use that instead of loading a pure LLM."
        )
        mpf_exists=True

    lm_model=AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": int(os.environ.get("LOCAL_RANK") or 0)},)

    tokenizer=AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    ori_tokenizer_len = len(tokenizer)
    tokenizer.add_special_tokens({'pad_token': '<|pad_token|>'})
    tokenizer.add_tokens(['<|protein_token|>', '<|thermo_token|>', '<|ph_token|>'])
    cur_tokenizer_len = len(tokenizer)
    smart_tokenizer_and_embedding_resize(
        tokenizer, lm_model, cur_tokenizer_len - ori_tokenizer_len
    )

    lm_model.config.vocab_size=len(tokenizer)

    # Load frozen ESM model via fair-esm (local .pt checkpoint)
    logger.info(f"Loading ESM model from: {data_args.esm_model_name}")
    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_local(data_args.esm_model_name)
    esm_model = esm_model.to(torch.bfloat16)

    config=RippleLMConfig(
        text_config=lm_model.config,
        protein_token='<|protein_token|>',
        protein_token_index=int(tokenizer.convert_tokens_to_ids('<|protein_token|>')),
        thermo_token='<|thermo_token|>',
        thermo_token_index=int(tokenizer.convert_tokens_to_ids('<|thermo_token|>')),
        ph_token='<|ph_token|>',
        ph_token_index=int(tokenizer.convert_tokens_to_ids('<|ph_token|>')),
        enable_property_task=data_args.enable_property_task,
        esm_model_name=data_args.esm_model_name,
    )

    model = RippleLMForConditionalGeneration(config, lm_model, esm_model=esm_model)
    if mpf_exists:
        model.mpf.load_state_dict(
            torch.load(os.path.join(model_args.model_name_or_path, "mpf.pth"), map_location="cpu")
        )
    thermo_head_path = os.path.join(model_args.model_name_or_path, "thermo_head.pth")
    if os.path.exists(thermo_head_path):
        model.thermo_head.load_state_dict(
            torch.load(thermo_head_path, map_location="cpu")
        )
    ph_head_path = os.path.join(model_args.model_name_or_path, "ph_head.pth")
    if os.path.exists(ph_head_path):
        model.ph_head.load_state_dict(
            torch.load(ph_head_path, map_location="cpu")
        )

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_targets.split(","),
        exclude_modules="esm_model\\..*",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=(
            model_args.modules_to_save.split(",")
            if model_args.modules_to_save is not None
            else None
        ),
    )
    model = get_peft_model(model, lora_config)

    if model_args.lora_r == 1:
        logging.info("lora_r is 1, freezing all LoRA parameters.")
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False

    print_trainable_parameters(model)
    return model, tokenizer, esm_alphabet, config


def prepare_dataset(data_args: DataArguments, protein_token, thermo_token, ph_token) :

    dataset = RippleLMDataset(
        csv_path=data_args.csv_path,
        protein_token=protein_token,
        thermo_token=thermo_token,
        ph_token=ph_token,
        ephod_label_col=data_args.ephod_label_col,
        pptstab_label_col=data_args.pptstab_label_col,
    )

    return dataset

class RippleSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        eval_generation_config: Optional[GenerationConfig] = None,
        eval_max_samples: int = 100,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._thermo_loss_buffer = []
        self._ph_loss_buffer = []
        self._mpf_reg_loss_buffer = []
        self._eval_generation_config = eval_generation_config
        self._eval_max_samples = eval_max_samples

    def _get_cls_weight(self):
        """Linear decay of classification weight: 0.4 → 0.2 over training.

        Reflects latent-thought crystallisation schedule:
        early (high weight) forces latent tokens to encode property info;
        late (lower weight) lets the LM focus on generation quality while
        retaining a non-zero property supervision signal throughout.
        """
        if self.state.max_steps <= 0:
            return 0.4
        progress = min(self.state.global_step / self.state.max_steps, 1.0)
        return 0.2 + 0.2 * (1.0 - progress)

    def create_optimizer(self):
        """Use 10x learning rate for property classification heads."""
        if self.optimizer is not None:
            return self.optimizer

        head_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "thermo_head" in name or "ph_head" in name:
                head_params.append(param)
            else:
                other_params.append(param)

        head_lr = self.args.learning_rate * 10
        self.optimizer = AdamW(
            [
                {"params": other_params, "lr": self.args.learning_rate},
                {"params": head_params, "lr": head_lr},
            ],
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )
        logger.info(
            f"Custom optimizer: {len(other_params)} params @ lr={self.args.learning_rate}, "
            f"{len(head_params)} head params @ lr={head_lr}"
        )
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        thermo_loss = getattr(outputs, "thermo_loss", None)
        ph_loss = getattr(outputs, "ph_loss", None)
        mpf_reg_loss = getattr(outputs, "mpf_reg_loss", None)

        if model.training:
            if thermo_loss is not None:
                self._thermo_loss_buffer.append(
                    float(thermo_loss.detach().float().item())
                )
            if ph_loss is not None:
                self._ph_loss_buffer.append(
                    float(ph_loss.detach().float().item())
                )
            if mpf_reg_loss is not None:
                self._mpf_reg_loss_buffer.append(
                    float(mpf_reg_loss.detach().float().item())
                )

        # Latent-thought crystallisation schedule
        cls_losses = [x for x in [thermo_loss, ph_loss] if x is not None]
        if cls_losses:
            cls_loss = sum(cls_losses) / len(cls_losses)
            w_cls = self._get_cls_weight()
            if loss is not None:
                loss = (1.0 - w_cls) * loss + w_cls * cls_loss
            else:
                loss = cls_loss

        base_config = getattr(model, "config", None)
        if base_config is None and hasattr(model, "get_base_model"):
            base_config = getattr(model.get_base_model(), "config", None)
        reg_weight = getattr(base_config, "mpf_disentangle_weight", 0.05)
        if mpf_reg_loss is not None:
            if loss is not None:
                loss = loss + reg_weight * mpf_reg_loss
            else:
                loss = reg_weight * mpf_reg_loss

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        logs["w_cls"] = self._get_cls_weight()
        if self._thermo_loss_buffer:
            logs["thermo_loss"] = sum(self._thermo_loss_buffer) / len(
                self._thermo_loss_buffer
            )
            self._thermo_loss_buffer.clear()
        if self._ph_loss_buffer:
            logs["ph_loss"] = sum(self._ph_loss_buffer) / len(self._ph_loss_buffer)
            self._ph_loss_buffer.clear()
        if self._mpf_reg_loss_buffer:
            logs["mpf_reg_loss"] = sum(self._mpf_reg_loss_buffer) / len(
                self._mpf_reg_loss_buffer
            )
            self._mpf_reg_loss_buffer.clear()
        try:
            return super().log(logs, start_time)   # 适配 4.51.0+
        except TypeError:
            return super().log(logs)               # 适配 4.43.1

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None or self._eval_generation_config is None:
            return {}

        was_training = self.model.training
        self.model.eval()

        device = self.args.device
        max_samples = min(len(eval_dataset), self._eval_max_samples)
        all_gen = []
        all_gt = []

        for i in range(max_samples):
            item = eval_dataset[i]
            batch = self.data_collator([item])

            # Extract prompt-only tokens (where labels == -100)
            prompt_mask = batch["labels"][0] == -100
            prompt_ids = batch["input_ids"][0][prompt_mask].unsqueeze(0)
            prompt_attn = batch["attention_mask"][0][prompt_mask].unsqueeze(0)
            prompt_length = prompt_ids.shape[1]

            gen_batch = {
                "input_ids": prompt_ids,
                "attention_mask": prompt_attn,
            }
            for key in [
                "wt_esm_input_ids", "wt_esm_attention_mask",
                "mt_esm_input_ids", "mt_esm_attention_mask",
                "position", "length", "mutation_type_id",
            ]:
                if key in batch and batch[key] is not None:
                    gen_batch[key] = batch[key]

            for k, v in gen_batch.items():
                if isinstance(v, torch.Tensor):
                    gen_batch[k] = v.to(device)

            with torch.no_grad():
                output_tokens = self.model.generate(
                    **gen_batch, generation_config=self._eval_generation_config
                )

            generated_ids = output_tokens[0][prompt_length:]
            gen_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            gt_text = item["target_text"].strip()

            all_gen.append(gen_text)
            all_gt.append(gt_text)

        metrics = compute_metrics_from_lists(all_gen, all_gt)
        metrics = {f"{metric_key_prefix}_{k}": v for k, v in metrics.items()}

        if self.args.process_index == 0:
            logger.info(f"Generative eval ({max_samples} samples): {metrics}")
        self.log(metrics)

        if was_training:
            self.model.train()
        return metrics

def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    model, tokenizer, esm_alphabet, config = load_model_tokenizer_for_training(model_args, data_args)


    train_dataset = prepare_dataset(
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

    # Load validation dataset for periodic generative evaluation
    eval_dataset = None
    eval_generation_config = None
    if data_args.val_csv_path:
        eval_dataset = RippleLMDataset(
            csv_path=data_args.val_csv_path,
            protein_token=config.protein_token,
            thermo_token=config.thermo_token,
            ph_token=config.ph_token,
            ephod_label_col=data_args.ephod_label_col,
            pptstab_label_col=data_args.pptstab_label_col,
        )
        eval_generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        logger.info(
            f"Loaded {len(eval_dataset)} validation samples for generative evaluation "
            f"(using up to {data_args.eval_max_samples} per eval)."
        )

    training_args.remove_unused_columns = False
    trainer = RippleSeq2SeqTrainer(
        eval_generation_config=eval_generation_config,
        eval_max_samples=data_args.eval_max_samples,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)


    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


    if model_args.merge_when_finished and (os.environ.get("LOCAL_RANK") in [None, "0"]):
        output_dir = training_args.output_dir.rstrip("/")
        merged_model_path = f"{output_dir}_merged"
        os.makedirs(merged_model_path, exist_ok=True)

        logging.info(f"Merging LoRA adapter and saving to {merged_model_path}...")
        merged_model = model.merge_and_unload()

        torch.save(
            merged_model.mpf.state_dict(),
            os.path.join(merged_model_path, "mpf.pth"),
        )
        torch.save(
            merged_model.thermo_head.state_dict(),
            os.path.join(merged_model_path, "thermo_head.pth"),
        )
        torch.save(
            merged_model.ph_head.state_dict(),
            os.path.join(merged_model_path, "ph_head.pth"),
        )
        merged_model.language_model.save_pretrained(merged_model_path)
        tokenizer.save_pretrained(merged_model_path)

        logging.info(f"Saved the merged model to {merged_model_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
