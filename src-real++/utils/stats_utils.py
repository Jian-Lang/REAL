import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

    
def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Convert to M (millions)
    def to_millions(num):
        return num / 1_000_000

    total_params_m = to_millions(total_params)
    trainable_params_m = to_millions(trainable_params)

    # Return total and trainable in M
    return total_params_m, trainable_params_m

def save_tsne_tensor(dataset, model_name, vids, labels, outputs):
    tensors = outputs['tsne_tensor']
    ori = outputs['ori']
    new_data = {}
    
    if ori:
        ori = '_ori'
    else:
        ori = ''

    # Iterate through vids, labels, and tensors
    for vid, label, fea in zip(vids, labels, tensors):
        # Store the data for each video
        new_data[vid] = {
            "label": label,
            "tensor": fea.detach().cpu(),
        }

    # Define the output file path
    output_path = Path(f"statis/data/tsne_tensors_{dataset}_{model_name}{ori}.pt")

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if output_path.exists():
        # Load existing data
        existing_data = torch.load(output_path, weights_only=True)
        
        # Merge existing data with new data
        existing_data.update(new_data)
        
        # Save the merged data
        torch.save(existing_data, output_path)
    else:
        # Save the new data as a torch tensor
        torch.save(new_data, output_path)

    print(f"Data saved/updated in {output_path}")
    

def save_manipulate_fea(dataset, model_name, vids, labels, outputs):
    ma_fea_text = outputs['ma_fea_text']
    ma_fea_vision = outputs['ma_fea_vision']
    ma_fea_audio = outputs['ma_fea_audio']
    
    ori_fea_text = outputs['ori_fea_text']
    ori_fea_vision = outputs['ori_fea_vision']
    ori_fea_audio = outputs['ori_fea_audio']
    
    new_data = {}

    # Iterate through vids, labels, and tensors
    for vid, label, ma_fea_text, ma_fea_vision, ma_fea_audio, ori_fea_text, ori_fea_vision, ori_fea_audio in zip(vids, labels, ma_fea_text, ma_fea_vision, ma_fea_audio, ori_fea_text, ori_fea_vision, ori_fea_audio):
        # Store the data for each video
        new_data[vid] = {
            "label": label,
            "ma_fea_text": ma_fea_text.detach().cpu(),
            "ma_fea_vision": ma_fea_vision.detach().cpu(),
            "ma_fea_audio": ma_fea_audio.detach().cpu(),
            "ori_fea_text": ori_fea_text.detach().cpu(),
            "ori_fea_vision": ori_fea_vision.detach().cpu(),
            "ori_fea_audio": ori_fea_audio.detach().cpu(),
        }

    # Define the output file path
    output_path = Path(f"statis/data/manipulate_fea_{dataset}_{model_name}.pt")

    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if the file already exists
    if output_path.exists():
        # Load existing data
        existing_data = torch.load(output_path, weights_only=True)
        
        # Merge existing data with new data
        existing_data.update(new_data)
        
        # Save the merged data
        torch.save(existing_data, output_path)
    else:
        # Save the new data as a torch tensor
        torch.save(new_data, output_path)

    print(f"Data saved/updated in {output_path}")
