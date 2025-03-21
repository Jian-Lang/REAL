from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


config = [
    ('FakeSV', 'thenlper/gte-large-zh'),
    ('FakeTT', 'thenlper/gte-large'),
    ('FVC', 'thenlper/gte-large'),
]

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class MyTextDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data_df = pd.read_json(os.path.join(dataset_dir, 'lm_ocr.jsonl'), lines=True, dtype={'vid': 'str'})

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        vid = self.data_df.loc[index, 'vid']
        
        text = self.data_df.loc[index, 'lm_ocr']
        return vid, text

class CollateClass:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        vids, texts = zip(*batch)
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        return vids, inputs

for dataset_name, model_name in config:
    dataset_dir = f'data/{dataset_name}'
    output_file = os.path.join(dataset_dir, 'retrieve', 'fea_retrieval_text_gte-large.pt')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, device_map='cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MyTextDataset(dataset_dir)
    collate_fn = CollateClass(tokenizer)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=8)

    features = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding texts for {dataset_name}"):
            vids, inputs = batch
            
            inputs = inputs.to(device)
            outputs = model(**inputs)
            
            # Get embeddings from the last hidden state
            # embeddings = outputs.last_hidden_state[:, 0, :]  # Take [CLS] token embedding
            embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = embeddings.cpu()

            for i, vid in enumerate(vids):
                features[vid] = embeddings[i]
    # print length of features
    print(f"Length of features: {len(features)}")
    print(f"Saving features to {output_file}")
    torch.save(features, output_file)
    print(f"Finished processing dataset: {dataset_name}\n")