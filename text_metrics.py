
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from typing import Tuple, List

import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm







def get_out_gt_from_line(file_path: str, line: str) -> Tuple[str, str]:
    
    if file_path.endswith(".txt"):
        parts = line.strip().split("<iamsplit>")
        if len(parts) != 2:
            raise ValueError(f"Invalid format in .txt file line: {line}")
        out, gt = parts[0].strip(), parts[1].strip()
    else:
        data = json.loads(line)
        out, gt = data["gen"].strip(), data["gt"].strip()
    return out, gt


def eval_mol2text(
    input_file: str,
    text_model: str = "allenai/scibert_scivocab_uncased",
    text_trunc_length: int = 512,
) -> Tuple[float, float, float, float, float, float]:
    
    with open(input_file, "r") as f:
        lines = f.readlines()[:]

    # text_tokenizer = BertTokenizerFast.from_pretrained(
    #     text_model, clean_up_tokenization_spaces=True
    # )
    rouge_evaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    special_tokens_to_filter = {"[PAD]", "[CLS]", "[SEP]"}
    references_for_bleu = []
    hypotheses_for_bleu = []
    meteor_scores = []
    rouge_scores = []

    for line in tqdm(lines, desc="Evaluating"):
        out_text, gt_text = get_out_gt_from_line(input_file, line)
        
        gt_tokens=gt_text.split(" ")
        gt_tokens_filtered = [
            token for token in gt_tokens if token not in special_tokens_to_filter
        ]

        
        
        
        out_tokens=out_text.split(" ")
        out_tokens_filtered = [
            token for token in out_tokens if token not in special_tokens_to_filter
        ]

        references_for_bleu.append([gt_tokens_filtered])
        hypotheses_for_bleu.append(out_tokens_filtered)
        meteor_scores.append(meteor_score([gt_tokens_filtered], out_tokens_filtered))

        
        rs = rouge_evaluator.score(out_text, gt_text)
        rouge_scores.append(rs)

    
    bleu2 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(
        references_for_bleu, hypotheses_for_bleu, weights=(0.25, 0.25, 0.25, 0.25)
    )
    avg_meteor_score = np.mean(meteor_scores)

    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores])
    print('b-2, b-4, meteor, r-1, r-2, r-L')

    return bleu2, bleu4, avg_meteor_score,rouge_1 ,rouge_2, rouge_l


def compute_metrics_from_lists(
    generations: List[str],
    references: List[str],
) -> dict:
    """Compute BLEU/ROUGE/METEOR from lists of generated and reference strings."""
    rouge_evaluator = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])

    references_for_bleu = []
    hypotheses_for_bleu = []
    meteor_scores_list = []
    rouge_scores_list = []

    for gen_text, gt_text in zip(generations, references):
        gt_tokens = gt_text.split(" ")
        gen_tokens = gen_text.split(" ")

        references_for_bleu.append([gt_tokens])
        hypotheses_for_bleu.append(gen_tokens)
        meteor_scores_list.append(meteor_score([gt_tokens], gen_tokens))

        rs = rouge_evaluator.score(gen_text, gt_text)
        rouge_scores_list.append(rs)

    bleu2 = corpus_bleu(references_for_bleu, hypotheses_for_bleu, weights=(0.5, 0.5))
    bleu4 = corpus_bleu(
        references_for_bleu, hypotheses_for_bleu, weights=(0.25, 0.25, 0.25, 0.25)
    )
    avg_meteor = np.mean(meteor_scores_list)
    rouge_1 = np.mean([rs["rouge1"].fmeasure for rs in rouge_scores_list])
    rouge_2 = np.mean([rs["rouge2"].fmeasure for rs in rouge_scores_list])
    rouge_l = np.mean([rs["rougeL"].fmeasure for rs in rouge_scores_list])

    return {
        "bleu2": bleu2,
        "bleu4": bleu4,
        "rouge1": rouge_1,
        "rouge2": rouge_2,
        "rougeL": rouge_l,
        "meteor": avg_meteor,
    }


if __name__ == "__main__":
    

    results = eval_mol2text(input_file="/env_1/llzh/RipplePLM/DSLlama_finetuned_on_MutaDescribe_temporal_online_v3_s2_merged/result.jsonl")
    
    formatted_results = [f"{item * 100:.2f}" for item in results]
    print("&".join(formatted_results))

