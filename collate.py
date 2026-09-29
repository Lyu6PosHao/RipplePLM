import torch
from typing import List, Dict
from transformers import PreTrainedTokenizer

class RippleLMCollator:


    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, esm_alphabet=None):

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.esm_alphabet = esm_alphabet
        self.esm_batch_converter = esm_alphabet.get_batch_converter() if esm_alphabet else None


        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer has no pad_token. Setting it to eos_token: '{self.tokenizer.eos_token}'")

    def __call__(self, batch: List[Dict]) -> Dict:

        # ESM tokenization for protein sequences
        wt_seqs = [s['wt_seq'] for s in batch]
        mt_seqs = [s['mt_seq'] for s in batch]

        wt_data = [(f"wt_{i}", seq) for i, seq in enumerate(wt_seqs)]
        mt_data = [(f"mt_{i}", seq) for i, seq in enumerate(mt_seqs)]
        _, _, wt_tokens = self.esm_batch_converter(wt_data)
        _, _, mt_tokens = self.esm_batch_converter(mt_data)
        wt_attention_mask = (wt_tokens != self.esm_alphabet.padding_idx).long()
        mt_attention_mask = (mt_tokens != self.esm_alphabet.padding_idx).long()

        protein_batch = {
            'wt_esm_input_ids': wt_tokens,
            'wt_esm_attention_mask': wt_attention_mask,
            'mt_esm_input_ids': mt_tokens,
            'mt_esm_attention_mask': mt_attention_mask,
            'position': [s['position'] for s in batch],
            'length': [s['length'] for s in batch],
            'mutation_type_id': [s['mutation_type_id'] for s in batch],
            'ephod_label': torch.tensor(
                [
                    int(s['ephod_label'])
                    for s in batch
                ],
                dtype=torch.long,
            ),
            'pptstab_label': torch.tensor(
                [
                    int(s['pptstab_label'])
                    for s in batch
                ],
                dtype=torch.long,
            ),
        }




        conversations=[]
        for s in batch:
            system_prompt = 'You are an expert biochemist. Your task is to analyze a protein mutation.'

            user_prompt = (
                f"{s['context_text'][:self.max_length//2]}\n\n\n"
                f"{s['mut_prompt']}"
            )
            assistant_response = s['target_text']
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_response}
            ]
            conversations.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            )



        tokenized_inputs = self.tokenizer(
            conversations,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.max_length
        )




        labels = tokenized_inputs.input_ids.clone()

        prompts_only = []
        for s in batch:
            system_prompt = 'You are an expert biochemist. Your task is to analyze a protein mutation.'

            user_prompt = (
                f"{s['context_text'][:self.max_length//2]}\n\n\n"
                f"{s['mut_prompt']}"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            prompts_only.append(
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            )


        prompt_lengths = [len(self.tokenizer(p).input_ids) for p in prompts_only]


        for i in range(len(batch)):
            labels[i, :prompt_lengths[i]] = -100


        labels[labels == self.tokenizer.pad_token_id] = -100

        text_batch = {
            'input_ids': tokenized_inputs.input_ids,
            'attention_mask': tokenized_inputs.attention_mask,
            'labels': labels
        }


        final_batch = {**protein_batch, **text_batch}


        return final_batch
