import pandas as pd
import torch
from transformers import AutoTokenizer, AutoProcessor
from pathlib import Path
from PIL import Image
import os
import numpy as np

from ..Base.base_data import Base_Dataset, FakeSV_Dataset, FakeTT_Dataset, FVC_Dataset


class SVFEND_Dataset(Base_Dataset):
    def __init__(self, fold: int, split: str, task: str, num_pos: int=5, num_neg: int=5, ablation: str='No', **kargs):
        super().__init__()
        self.fea_path = self.data_path / 'fea' / 'SVFEND'    

        self.vggishfeapath = self.fea_path / 'vggish_pre_features.pt'  # (batch, 36, dim)
        self.framefeapath= self.fea_path / 'vgg19_features.pt' # (batch, 32, dim)
        self.c3dfeapath= self.fea_path / 'c3d_features.pt'
        self.textfeapath = self.fea_path / 'fea_text.pt'
        self.data = self._get_data(fold, split, task)
        # self.data['description'] = self.data['title']

        self.frame_fea = torch.load(self.framefeapath, weights_only=True)
        self.c3d_fea = torch.load(self.c3dfeapath, weights_only=True)
        self.vggish_fea = torch.load(self.vggishfeapath, weights_only=True)
        self.text_fea = torch.load(self.textfeapath, weights_only=True)
        self.sim_df = pd.read_json(self.data_path / 'retrieve/sim.jsonl', lines=True, dtype={'vid': str})
        self.num_pos = num_pos
        self.num_neg = num_neg
        
    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['vid']
        label = item['label']

        audioframes = self.vggish_fea[vid]
        frames = self.frame_fea[vid]
        c3d = self.c3d_fea[vid]
        text_fea = self.text_fea[vid]
        
        sim_pos_vids = [v for v in self.sim_df[self.sim_df['vid'] == vid].iloc[0]['similarities'][0]['vid'][:self.num_pos]]
        sim_neg_vids = [v for v in self.sim_df[self.sim_df['vid'] == vid].iloc[0]['similarities'][1]['vid'][:self.num_neg]]
        
        text_fea_pos = torch.stack([self.text_fea[v] for v in sim_pos_vids])
        text_fea_neg = torch.stack([self.text_fea[v] for v in sim_neg_vids])
        vision_fea_pos = torch.stack([self.frame_fea[v].mean(-2) for v in sim_pos_vids])
        vision_fea_neg = torch.stack([self.frame_fea[v].mean(-2) for v in sim_neg_vids])
        audio_fea_pos = torch.stack([self.vggish_fea[v].mean(-2) for v in sim_pos_vids])
        audio_fea_neg = torch.stack([self.vggish_fea[v].mean(-2) for v in sim_neg_vids])
        
        return {
            'vid': vid,
            'label': torch.tensor(label),
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'text_fea': text_fea,
            'text_fea_pos': text_fea_pos,
            'text_fea_neg': text_fea_neg,
            'vision_fea_pos': vision_fea_pos,
            'vision_fea_neg': vision_fea_neg,
            'audio_fea_pos': audio_fea_pos,
            'audio_fea_neg': audio_fea_neg,
        }

class SVFEND_Collator:
    def __init__(self, **kargs):
        pass
    def __call__(self, batch):
        num_frames = 83
        num_audioframes = 50 
        
        vids = [item['vid'] for item in batch]
        
        text_fea = [item['text_fea'] for item in batch]
        text_fea = torch.stack(text_fea)

        frames = [item['frames'] for item in batch]
        # frames, frames_masks = pad_frame_sequence(num_frames, frames)
        frames = torch.stack(frames)

        audioframes  = [item['audioframes'] for item in batch]
        audioframes = torch.stack(audioframes)
        # audioframes, audioframes_masks = pad_frame_sequence(num_audioframes, audioframes)

        c3d = [item['c3d'] for item in batch]
        c3d = torch.stack(c3d)
        # _, c3d_masks = pad_frame_sequence(num_frames, c3d)

        labels = [item['label'] for item in batch]
        labels = torch.stack(labels)
        
        text_fea_pos = [item['text_fea_pos'] for item in batch]
        text_fea_pos = torch.stack(text_fea_pos)
        text_fea_neg = [item['text_fea_neg'] for item in batch]
        text_fea_neg = torch.stack(text_fea_neg)
        
        vision_fea_pos = [item['vision_fea_pos'] for item in batch]
        vision_fea_pos = torch.stack(vision_fea_pos)
        vision_fea_neg = [item['vision_fea_neg'] for item in batch]
        vision_fea_neg = torch.stack(vision_fea_neg)
        
        audio_fea_pos = [item['audio_fea_pos'] for item in batch]
        audio_fea_pos = torch.stack(audio_fea_pos)
        audio_fea_neg = [item['audio_fea_neg'] for item in batch]
        audio_fea_neg = torch.stack(audio_fea_neg)
        
        return {
            'vids': vids,
            'labels': labels,
            'text_fea': text_fea,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'text_fea_pos': text_fea_pos,
            'text_fea_neg': text_fea_neg,
            'vision_fea_pos': vision_fea_pos,
            'vision_fea_neg': vision_fea_neg,
            'audio_fea_pos': audio_fea_pos,
            'audio_fea_neg': audio_fea_neg,
        }
        
class FakeSV_SVFEND_Dataset(SVFEND_Dataset, FakeSV_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)
        
class FakeSV_SVFEND_Collator(SVFEND_Collator):
    pass

        
class FakeTT_SVFEND_Dataset(SVFEND_Dataset, FakeTT_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)

class FakeTT_SVFEND_Collator(SVFEND_Collator):
    pass


class FVC_SVFEND_Dataset(SVFEND_Dataset, FVC_Dataset):
    def __init__(self, fold: int, split: str, task: str, **kargs):
        super().__init__(fold=fold, split=split, task=task, **kargs)

class FVC_SVFEND_Collator(SVFEND_Collator):
    pass


def pad_frame_sequence(seq_len,lst):
    attention_masks = []
    result=[]
    for video in lst:
        video=torch.FloatTensor(video)
        ori_len=video.shape[0]
        if ori_len>=seq_len:
            gap=ori_len//seq_len
            video=video[::gap][:seq_len]
            mask = np.ones((seq_len))
        else:
            video=torch.cat((video,torch.zeros([seq_len-ori_len,video.shape[1]],dtype=torch.float)),dim=0)
            mask = np.append(np.ones(ori_len), np.zeros(seq_len-ori_len))
        result.append(video)
        mask = torch.IntTensor(mask)
        attention_masks.append(mask)
    return torch.stack(result), torch.stack(attention_masks)