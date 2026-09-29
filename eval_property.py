"""
Property prediction evaluation for RipplePLM.

Evaluates the model's thermo_head (thermal stability) and ph_head (optimal pH)
against expert pseudo-labels (PPT-Stab / EpHod) on held-out test sets.

Usage:
    python eval_property.py \
        --model_name_or_path <merged_model_dir> \
        --csv_path <test_csv> \
        --esm_model_name <esm_checkpoint> \
        --output_path <report.json>
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import esm
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
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
        metadata={"help": "Path to the merged model directory."},
    )


@dataclass
class DataArguments:
    csv_path: str = field(metadata={"help": "Path to the test CSV file."})
    esm_model_name: str = field(
        default="./esm2_t12_35M_UR50D.pt",
        metadata={"help": "Local path to fair-esm .pt checkpoint."},
    )
    output_path: str = field(
        default="property_eval_report.json",
        metadata={"help": "Path to save the evaluation report (JSON)."},
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "Maximum sequence length."}
    )
    ephod_label_col: str = field(
        default="ephod_label",
        metadata={"help": "Column name for pH optimum label in the CSV."},
    )
    pptstab_label_col: str = field(
        default="pptstab_label",
        metadata={"help": "Column name for thermal stability label in the CSV."},
    )


def load_model_for_eval(model_args: ModelArguments, data_args: DataArguments):
    """Load model with enable_property_task=True for property head evaluation."""
    logger.info(f"Loading tokenizer and model from {model_args.model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    protein_token_id = int(tokenizer.convert_tokens_to_ids("<|protein_token|>"))
    thermo_token_id = int(tokenizer.convert_tokens_to_ids("<|thermo_token|>"))
    ph_token_id = int(tokenizer.convert_tokens_to_ids("<|ph_token|>"))

    base_llm = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )

    logger.info(f"Loading ESM model from: {data_args.esm_model_name}")
    esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet_local(
        data_args.esm_model_name
    )
    esm_model = esm_model.to(torch.bfloat16)

    lm_config = base_llm.config
    config = RippleLMConfig(
        text_config=lm_config,
        protein_token="<|protein_token|>",
        protein_token_index=protein_token_id,
        thermo_token="<|thermo_token|>",
        thermo_token_index=thermo_token_id,
        ph_token="<|ph_token|>",
        ph_token_index=ph_token_id,
        enable_property_task=True,
        esm_model_name=data_args.esm_model_name,
    )

    model = RippleLMForConditionalGeneration(config, base_llm, esm_model=esm_model)

    # Load MPF weights
    mpf_path = os.path.join(model_args.model_name_or_path, "mpf.pth")
    if os.path.exists(mpf_path):
        logger.info(f"Loading MPF weights from {mpf_path}")
        mpf_state_dict = torch.load(mpf_path, map_location="cpu")
        model.mpf.load_state_dict(mpf_state_dict)
        model.mpf.to(device="cuda", dtype=torch.bfloat16)
    else:
        logger.warning(f"Could not find mpf.pth at {mpf_path}.")

    # Load thermo_head weights
    thermo_head_path = os.path.join(model_args.model_name_or_path, "thermo_head.pth")
    if os.path.exists(thermo_head_path):
        logger.info(f"Loading thermo_head weights from {thermo_head_path}")
        thermo_head_state_dict = torch.load(thermo_head_path, map_location="cpu")
        model.thermo_head.load_state_dict(thermo_head_state_dict)
        model.thermo_head.to(device="cuda", dtype=torch.bfloat16)
    else:
        logger.warning(f"Could not find thermo_head.pth at {thermo_head_path}.")

    # Load ph_head weights
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


def compute_classification_metrics(y_true, y_pred, task_name):
    """Compute classification metrics for a binary property task."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    metrics = {
        "task": task_name,
        "num_samples": len(y_true),
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_precision * 100, 2),
        "macro_recall": round(macro_recall * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "per_class": {
            "class_0": {
                "precision": round(precision[0] * 100, 2),
                "recall": round(recall[0] * 100, 2),
                "f1": round(f1[0] * 100, 2),
                "support": int(support[0]) if support is not None else 0,
            },
            "class_1": {
                "precision": round(precision[1] * 100, 2),
                "recall": round(recall[1] * 100, 2),
                "f1": round(f1[1] * 100, 2),
                "support": int(support[1]) if support is not None else 0,
            },
        },
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def run():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    model, tokenizer, esm_alphabet, config = load_model_for_eval(model_args, data_args)

    test_dataset = RippleLMDataset(
        csv_path=data_args.csv_path,
        protein_token=config.protein_token,
        thermo_token=config.thermo_token,
        ph_token=config.ph_token,
        ephod_label_col=data_args.ephod_label_col,
        pptstab_label_col=data_args.pptstab_label_col,
    )

    data_collator = RippleLMCollator(
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        esm_alphabet=esm_alphabet,
    )

    thermo_preds, thermo_labels = [], []
    ph_preds, ph_labels = [], []

    for i in tqdm(range(len(test_dataset)), desc="Evaluating properties"):
        item = test_dataset[i]
        batch = data_collator([item])

        # Move tensors to GPU
        for k, v in batch.items():
            if v is not None and isinstance(v, torch.Tensor):
                batch[k] = v.to("cuda")

        with torch.no_grad():
            outputs = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )

        # Collect thermostability predictions
        if outputs.thermo_logits is not None:
            pred = outputs.thermo_logits.argmax(dim=-1).cpu().tolist()
            thermo_preds.extend(pred)
            thermo_labels.append(item["pptstab_label"])

        # Collect pH optimum predictions
        if outputs.ph_logits is not None:
            pred = outputs.ph_logits.argmax(dim=-1).cpu().tolist()
            ph_preds.extend(pred)
            ph_labels.append(item["ephod_label"])

    # Compute metrics
    report = {
        "model": model_args.model_name_or_path,
        "csv_path": data_args.csv_path,
        "total_samples": len(test_dataset),
    }

    if thermo_preds and thermo_labels:
        thermo_metrics = compute_classification_metrics(
            thermo_labels, thermo_preds, "thermostability"
        )
        report["thermostability"] = thermo_metrics
        print("\n" + "=" * 60)
        print("Thermostability (PPT-Stab pseudo-label agreement)")
        print("=" * 60)
        print(f"  Samples:   {thermo_metrics['num_samples']}")
        print(f"  Accuracy:  {thermo_metrics['accuracy']:.2f}%")
        print(f"  Macro F1:  {thermo_metrics['macro_f1']:.2f}%")
        print(f"  Precision: {thermo_metrics['macro_precision']:.2f}%")
        print(f"  Recall:    {thermo_metrics['macro_recall']:.2f}%")
        print(f"  Confusion Matrix: {thermo_metrics['confusion_matrix']}")
        print(classification_report(
            thermo_labels, thermo_preds, labels=[0, 1],
            target_names=["Class 0", "Class 1"], zero_division=0,
        ))

    if ph_preds and ph_labels:
        ph_metrics = compute_classification_metrics(
            ph_labels, ph_preds, "ph_optimum"
        )
        report["ph_optimum"] = ph_metrics
        print("\n" + "=" * 60)
        print("pH Optimum (EpHod pseudo-label agreement)")
        print("=" * 60)
        print(f"  Samples:   {ph_metrics['num_samples']}")
        print(f"  Accuracy:  {ph_metrics['accuracy']:.2f}%")
        print(f"  Macro F1:  {ph_metrics['macro_f1']:.2f}%")
        print(f"  Precision: {ph_metrics['macro_precision']:.2f}%")
        print(f"  Recall:    {ph_metrics['macro_recall']:.2f}%")
        print(f"  Confusion Matrix: {ph_metrics['confusion_matrix']}")
        print(classification_report(
            ph_labels, ph_preds, labels=[0, 1],
            target_names=["Class 0", "Class 1"], zero_division=0,
        ))

    # Save report
    os.makedirs(os.path.dirname(data_args.output_path) or ".", exist_ok=True)
    with open(data_args.output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation report saved to {data_args.output_path}")
    print(f"\nReport saved to: {data_args.output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run()
