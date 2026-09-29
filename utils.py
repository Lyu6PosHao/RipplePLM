import logging
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)



def set_seed(seed: int = 42) -> None:
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_nb_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            num_bytes = (
                param.quant_storage.itemsize if hasattr(param, "quant_storage") else 1
            )
            num_params *= 2 * num_bytes
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    return trainable_params, all_param


def print_trainable_parameters(model: nn.Module) -> None:
    
    trainable_params, all_param = get_nb_trainable_parameters(model)
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

def smart_tokenizer_and_embedding_resize(tokenizer, model, num_new_tokens: int) -> None:
    
    if num_new_tokens <= 0:
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be non-negative")
        logging.info("No new tokens added.")
        return

    model.resize_token_embeddings(len(tokenizer))
    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data

    input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
        dim=0, keepdim=True
    )

    input_embeddings[-num_new_tokens:] = input_embeddings_avg
    output_embeddings[-num_new_tokens:] = output_embeddings_avg
    logging.info(
        f"{num_new_tokens} new tokens added to the tokenizer; embed_tokens and lm_head resized."
    )
