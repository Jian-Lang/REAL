import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
import soundfile as sf
import numpy as np


from vggish_input import waveform_to_examples_target

config = [
    ['FakeTT'], ['FVC'], ['FakeSV']
]

def wavfile_to_examples(wav_file, return_tensor=True):
    wav_data, sr = sf.read(wav_file, dtype='int16')
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    return waveform_to_examples_target(samples, sr, return_tensor)

def generate_dummy_feature():

    return torch.randn(36, 2, 96, 64)

class MyDataset(Dataset):
    def __init__(self, src_file):
        self.data = pd.read_json(src_file, lines=True, dtype={'vid': str})
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid = self.data.iloc[index]['vid']
        audio_path = os.path.join(self.audio_dir, f'{vid}.wav')
        
        if os.path.exists(audio_path):
            features = wavfile_to_examples(audio_path)
        else:
            print(f"Warning: Audio file not found for vid {vid}. Using dummy feature.")
            features = generate_dummy_feature()
        
        return vid, features

def customed_collate_fn(batch):
    vids, features = zip(*batch)
    features = torch.stack(features)
    return vids, features

for dataset in config:
    dataset_name = dataset[0]
    src_file = f'data/{dataset_name}/data.jsonl'
    output_dir = f'data/{dataset_name}/fea/SVFEND'
    audio_dir = f'data/{dataset_name}/audios'  
    
    output_file = os.path.join(output_dir, 'vggish_pre_features.pt')
    if os.path.exists(output_file):
        print(f"Skipping {dataset_name} as output file already exists")
        continue
        
    print(f"Processing {dataset_name}")
    

    os.makedirs(output_dir, exist_ok=True)
    
    save_dict = {}
    dataset = MyDataset(src_file)
    dataset.audio_dir = audio_dir  # Add audio_dir to dataset instance
    
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=customed_collate_fn, num_workers=8)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            vids, features = batch
            for i, vid in enumerate(vids):
                save_dict[vid] = features[i].view(36, 12288).detach().cpu()


    torch.save(save_dict, output_file)