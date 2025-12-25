from transformers import CLIPModel, ChineseCLIPModel
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
import torch
import torch.nn as nn

config = [
    ['FakeSV', 'google-bert/bert-base-chinese'],
    ['FakeTT', 'google-bert/bert-base-uncased'],
    ['FVC', 'bert-base-multilingual-uncased']
]

dataset_dir_base = 'data/'

class MyTextDataset(Dataset):
    def __init__(self, dataset_dir):
        self.data_df = pd.read_json(os.path.join(dataset_dir, 'data.jsonl'), lines=True, dtype={'vid': 'str'})

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        vid = self.data_df.loc[index, 'vid']
        
        title = self.data_df.loc[index, 'title']
        ocr = self.data_df.loc[index, 'ocr']
        # trans = self.data_df.loc[index, 'transcript']
        
        text = title + ' ' + ocr

        return vid, text

def collate_fn(batch):
    vids, texts = zip(*batch)
    
    return vids, texts

for cfg in config:
    dataset_name, model_id = cfg
    max_length = 512
    print(f"Processing dataset: {dataset_name}")
    

    dataset_dir = os.path.join(dataset_dir_base, dataset_name)
    output_file = os.path.join(dataset_dir, 'fea', 'SVFEND/fea_text.pt')

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, device_map='cuda' if torch.cuda.is_available() else 'cpu')

    # ori_embedding_weight = model.text_model.embeddings.position_embedding.weight
    # if max_length > 77:
    #     interpolated_weight = torch.nn.functional.interpolate(
    #         ori_embedding_weight.view(1, 1, 77, 768),
    #         size=(max_length, 768), mode='bilinear').squeeze(0).squeeze(0)

    #     model.text_model.embeddings.position_ids = torch.arange(max_length).unsqueeze(0)
    #     model.text_model.embeddings.position_embedding = nn.Embedding.from_pretrained(interpolated_weight, freeze=False)

    dataset = MyTextDataset(dataset_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=collate_fn, num_workers=8)
    
    features = {}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Encoding texts for {dataset_name}"):
            vids, texts = batch

            inputs = tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length=max_length).to(device)

            cls_text = model(**inputs)['last_hidden_state'][:, 0, :]
            cls_text = cls_text.cpu()
            
            for i, vid in enumerate(vids):
                features[vid] = cls_text[i]

    print(f"Saving features to {output_file}")
    torch.save(features, output_file)
    print(f"Finished processing dataset: {dataset_name}\n")