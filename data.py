import os
import re,json,random
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'

NON_STANDARD_AAS = 'BZUOX'

ALL_AAS = STANDARD_AAS + NON_STANDARD_AAS

def create_mutation_map() -> Dict[str, int]:

    mutation_map = {}


    mutation_map["<UNK>"] = 0

    idx = 1
    for original_aa in ALL_AAS:
        for mutated_aa in ALL_AAS:

            if original_aa != mutated_aa:
                key = f"{original_aa}->{mutated_aa}"
                mutation_map[key] = idx
                idx += 1

    return mutation_map


class RippleLMDataset(Dataset):

    def __init__(
        self,
        csv_path: str,
        protein_token: str,
        thermo_token: str,
        ph_token: str,
        ephod_label_col: str,
        pptstab_label_col: str,
    ):
        super().__init__()

        self.protein_token = protein_token
        self.thermo_token = thermo_token
        self.ph_token = ph_token
        self.ephod_label_col = ephod_label_col
        self.pptstab_label_col = pptstab_label_col

        print("Loading dataset...")
        try:
            self.metadata_df = pd.read_csv(csv_path)
            print(F"Dataset loaded successfully. Found {len(self.metadata_df)} samples.")

            self.metadata_df = self.metadata_df.dropna(subset=['all_description','function']).reset_index(drop=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The CSV file was not found at {csv_path}")

        self.mutation_map = create_mutation_map()
        self.has_protein2 = 'protein2' in self.metadata_df.columns

        print(f"Dataset filtered successfully. Found {len(self.metadata_df)} samples for training.")

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Dict:

        row = self.metadata_df.iloc[idx]
        entry = row['entry']


        try:
            protein_id, mutation_info = entry.split('-', 1)
        except ValueError:
            raise ValueError(f"Entry '{entry}' at index {idx} has an invalid format. Expected 'PROTEINID-MUTATION'.")


        match = re.match(r'([A-Z])(\d+)([A-Z])', mutation_info)
        if not match:
            raise ValueError(f"Mutation info '{mutation_info}' from entry '{entry}' has an invalid format. Expected 'A123B'.")

        wt_aa, pos_str, mt_aa = match.groups()

        mutation_position = int(pos_str) - 1

        wt_seq = row['protein1']
        if self.has_protein2 and pd.notna(row.get('protein2', None)):
            mt_seq = row['protein2']
        else:
            mt_seq = wt_seq[:mutation_position] + mt_aa + wt_seq[mutation_position+1:]

        context_text = f"Wild-type protein function: {row['function']}"
        template = "Next is a feature of the mutation {} to {} at position {}. Please generate a {} text to describe it. The feature is {}."
        uni_despt = row["uniprot_description"] if not pd.isna(row["uniprot_description"]) else ''
        GPT_despt = row["GPT_description"] if not pd.isna(row["GPT_description"]) else ''
        mut_prompt = template.format(wt_aa,mt_aa,str(mutation_position), "long detailed" if len(GPT_despt) >= 1 else "brief summary", self.protein_token*4)
        mut_prompt+=f" {self.thermo_token} {self.ph_token}"
        target_text=(uni_despt + ' ' + GPT_despt).strip()
        ephod_label = int(row[self.ephod_label_col])
        pptstab_label = int(row[self.pptstab_label_col])
        target_text = (
            f"{target_text}"
        ).strip()



        mutation_key = f"{wt_aa}->{mt_aa}"
        mutation_type_id = self.mutation_map.get(mutation_key, 0)
        if mutation_type_id == 0:
            print(f"Warning: Could not find mutation type for '{mutation_key}' from entry '{entry}'.")

        sample = {
            'entry': entry,
            'length': len(wt_seq),
            'wt_seq': wt_seq,
            'mt_seq': mt_seq,
            'wt_aa': wt_aa,
            'mt_aa': mt_aa,
            'position': int(mutation_position),
            'mutation_type_id': mutation_type_id,
            'context_text': context_text,
            'target_text': target_text,
            'mut_prompt':mut_prompt,
            'ephod_label': ephod_label,
            'pptstab_label': pptstab_label,
        }

        return sample



class LiteratureDataset(Dataset):
    def __init__(self, path, **kwargs) -> None:
        super().__init__()
        self.uniprot2pubmed = json.load(open(os.path.join(path, "uniprot_pubmed.json"), "r"))
        self.uniprot2seq = {}
        self.uniprot2func = {}
        uniprot_data = json.load(open(os.path.join(path, "uniprot_accession.json"), "r"))
        keys = set()
        for id in uniprot_data:
            for key in uniprot_data[id]:
                if key not in keys:
                    keys.add(key)
            if "Sequence" in uniprot_data[id] and id in self.uniprot2pubmed:
                self.uniprot2seq[id] = uniprot_data[id]["Sequence"]
                if "Description" in uniprot_data[id]:
                    pattern1 = r'\(PubMed:\d+(, PubMed:\d+)*\)'
                    pattern2 = r'\(By similarity\)'
                    self.uniprot2func[id] = "; ".join([re.sub(pattern2, '', re.sub(pattern1, '', text)) for text in uniprot_data[id]["Description"]])
        self.uniprot_ids = list(self.uniprot2seq.keys())
        self.pubmed_corpus = {}
        with open(os.path.join(path, "corpus.jsonl"), "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                if data["title"] is not None and data["abstract"] is not None:
                    self.pubmed_corpus[data["pubmed"]] = data["title"] + " " + data["abstract"]
                elif data["title"] is not None:
                    self.pubmed_corpus[data["pubmed"]] = data["title"]
                elif data["abstract"] is not None:
                    self.pubmed_corpus[data["pubmed"]] = data["abstract"]
        for id in self.uniprot2pubmed:
            self.uniprot2pubmed[id] = [pid for pid in self.uniprot2pubmed[id] if pid in self.pubmed_corpus]

    def get_by_uniport(self, id):
        print(id)
        if id in self.uniprot2func:
            print("Function:", self.uniprot2func[id])
            print("---------------------------------------------")
        for j in self.uniprot2pubmed[id]:
            print(self.pubmed_corpus[j])

    def __len__(self):
        return len(self.uniprot_ids)

    def __getitem__(self, index):
        id = self.uniprot_ids[index]
        seq = self.uniprot2seq[id]
        text_id = random.sample(self.uniprot2pubmed[id], k=1)[0]
        return id, self.pubmed_corpus[text_id]

    def get_example(self):
        for i in range(len(self)):
            seq, text = self[i]
            yield "Accession: " + self.uniprot_ids[i] + "\tSequence:" + seq[:30] + "...\tText:" + text[:100]
        raise RuntimeError("Number of examples exceed dataset length!")

