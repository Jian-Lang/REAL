import copy
import json
import os
import time
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from einops import rearrange
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, BertModel
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
from torch_geometric.utils import dense_to_sparse

from .coattention import CoAttention
from ..Base.utils import orthogonal_loss, l2_loss_fn


class ModalityProtoGenerator(nn.Module):
    def __init__(self, fea_dim=256, dropout=0.2):
        super(ModalityProtoGenerator, self).__init__()
        self.linear = nn.LazyLinear(fea_dim)
        self.gat = GATConv(fea_dim, fea_dim, heads=8, dropout=dropout, concat=False)
        self.edge_index = None
    
    def forward(self, x):
        # x: (batch_size, num_nodes, fea_dim)
        x = self.linear(x)
        batch_size, num_nodes, fea_dim = x.size()
        x = x.view(-1, fea_dim)  # Reshape to (batch_size * num_nodes, fea_dim)

        if self.edge_index is None:
            # Create edge_index for a fully connected graph without self-loops
            adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
            edge_index, _ = dense_to_sparse(adj)
            self.edge_index = edge_index.to(x.device)

        # Adjust edge_index for each graph in the batch
        edge_indices = []
        for i in range(batch_size):
            offset = i * num_nodes
            edge_indices.append(self.edge_index + offset)
        edge_index = torch.cat(edge_indices, dim=1)  # Shape: (2, num_edges * batch_size)

        # Create a batch tensor indicating the graph membership of each node
        batch = torch.arange(batch_size).unsqueeze(1).repeat(1, num_nodes).view(-1).to(x.device)

        # Apply GATConv
        proto = self.gat(x, edge_index)

        # Reshape back to (batch_size, num_nodes, fea_dim)
        proto = proto.view(batch_size, num_nodes, -1)
        return proto

class AddLinear(nn.Module):
    def __init__(self, fea_dim=256):
        super(AddLinear, self).__init__()
        self.linear = nn.LazyLinear(fea_dim)
        
    def forward(self, x):
        return self.linear(x)

class SVFEND(nn.Module):
    def __init__(self, encoder_name='bert-base-uncased', fea_dim=128, dropout=0.1, ori=False, ablation='No', **kargs):
        super(SVFEND, self).__init__()

        # self.bert = BertModel.from_pretrained(encoder_name).requires_grad_(False)

        self.text_dim = 768
        self.comment_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 32
        self.num_audioframes = 36
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4
        self.audio_dim = 12288

        self.dropout = dropout   
        
        self.vggish_layer = torch.hub.load('torchvggish', 'vggish', source='github')
        net_structure = list(self.vggish_layer.children())      
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        # freeze vggish
        for param in self.vggish_modified.parameters():
            param.requires_grad = False
        
        self.co_attention_ta = CoAttention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.num_audioframes, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.co_attention_tv = CoAttention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout, d_model=fea_dim,
                                        visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim, pos=False)
        self.trm = nn.TransformerEncoderLayer(d_model=self.dim, nhead=2, dropout=dropout, batch_first=True)


        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim),torch.nn.ReLU(),nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(fea_dim, fea_dim), torch.nn.ReLU(),nn.Dropout(p=self.dropout))

        self.classifier = nn.Linear(fea_dim,2)
        
        self.text_pos_self_attn = ModalityProtoGenerator(fea_dim, dropout)
        self.text_neg_self_attn = ModalityProtoGenerator(fea_dim, dropout)
        self.vision_pos_self_attn = ModalityProtoGenerator(fea_dim, dropout)
        self.vision_neg_self_attn = ModalityProtoGenerator(fea_dim, dropout)
        self.audio_pos_self_attn = ModalityProtoGenerator(fea_dim, dropout)
        self.audio_neg_self_attn = ModalityProtoGenerator(fea_dim, dropout)
        
        self.text_pos_mean = nn.LazyLinear(fea_dim)
        self.text_neg_mean = nn.LazyLinear(fea_dim)
        self.vision_pos_mean = nn.LazyLinear(fea_dim)
        self.vision_neg_mean = nn.LazyLinear(fea_dim)
        self.audio_pos_mean = nn.LazyLinear(fea_dim)
        self.audio_neg_mean = nn.LazyLinear(fea_dim)
        
        self.add_linear_text = AddLinear(fea_dim)
        self.add_linear_vision = AddLinear(fea_dim)
        self.add_linear_audio = AddLinear(fea_dim)
        self.ori = ori
        self.alpha = 0.01
        self.beta = 0.01
        self.ablation = ablation


    def forward(self, **kwargs):
        fea_text_pos = kwargs['text_fea_pos']
        fea_text_neg = kwargs['text_fea_neg']
        fea_vision_pos = kwargs['vision_fea_pos']
        fea_vision_neg = kwargs['vision_fea_neg']
        fea_audio_pos = kwargs['audio_fea_pos']
        fea_audio_neg = kwargs['audio_fea_neg']
        
        if self.ablation == 'w/o-retriever':
            fea_text_pos = fea_text_pos[torch.randperm(fea_text_pos.size(0)), :, :]
            fea_text_neg = fea_text_neg[torch.randperm(fea_text_neg.size(0)), :, :]
            fea_vision_pos = fea_vision_pos[torch.randperm(fea_vision_pos.size(0)), :, :]
            fea_vision_neg = fea_vision_neg[torch.randperm(fea_vision_neg.size(0)), :, :]
            fea_audio_pos = fea_audio_pos[torch.randperm(fea_audio_pos.size(0)), :, :]
            fea_audio_neg = fea_audio_neg[torch.randperm(fea_audio_neg.size(0)), :, :]
        
        text_fea = kwargs['text_fea']
        fea_text = self.linear_text(text_fea) 
        
        audioframes=kwargs['audioframes']
        fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio) 
        
        frames=kwargs['frames']
        fea_img = self.linear_img(frames)
        
        fea_text_pos = torch.cat([text_fea.unsqueeze(1), fea_text_pos], dim=1)
        fea_text_neg = torch.cat([text_fea.unsqueeze(1), fea_text_neg], dim=1)
        fea_vision_pos = torch.cat([frames.mean(-2).unsqueeze(1), fea_vision_pos], dim=1)
        fea_vision_neg = torch.cat([frames.mean(-2).unsqueeze(1), fea_vision_neg], dim=1)
        fea_audio_pos = torch.cat([audioframes.mean(-2).unsqueeze(1), fea_audio_pos], dim=1)
        fea_audio_neg = torch.cat([audioframes.mean(-2).unsqueeze(1), fea_audio_neg], dim=1)
        
        if self.ablation == 'w/o-graph':
            text_pos_proto = self.text_pos_mean(fea_text_pos).mean(-2)
            text_neg_proto = self.text_neg_mean(fea_text_neg).mean(-2)
            vision_pos_proto = self.vision_pos_mean(fea_vision_pos).mean(-2)
            vision_neg_proto = self.vision_neg_mean(fea_vision_neg).mean(-2)
            audio_pos_proto = self.audio_pos_mean(fea_audio_pos).mean(-2)
            audio_neg_proto = self.audio_neg_mean(fea_audio_neg).mean(-2)
        else:
            text_pos_proto = self.text_pos_self_attn(fea_text_pos)[:, 0, :]
            text_neg_proto = self.text_neg_self_attn(fea_text_neg)[:, 0, :]
            vision_pos_proto = self.vision_pos_self_attn(fea_vision_pos)[:, 0, :]
            vision_neg_proto = self.vision_neg_self_attn(fea_vision_neg)[:, 0, :]
            audio_pos_proto = self.audio_pos_self_attn(fea_audio_pos)[:, 0, :]
            audio_neg_proto = self.audio_neg_self_attn(fea_audio_neg)[:, 0, :]
        
        add_fea_text = self.add_linear_text(fea_text)
        add_fea_vision = self.add_linear_vision(fea_img)
        add_fea_audio = self.add_linear_audio(fea_audio)
        
        add_fea_text_opt = add_fea_text
        add_fea_vision_opt = add_fea_vision.mean(-2)
        add_fea_audio_opt = add_fea_audio.mean(-2)
        
        ori_fea_text = fea_text.clone()
        ori_fea_vision = fea_img.mean(-2).clone()
        ori_fea_audio = fea_audio.mean(-2).clone()
        
        if not self.ori:
            fea_text = fea_text + add_fea_text
            fea_img = fea_img + add_fea_vision
            fea_audio = fea_audio + add_fea_audio
        
        ma_fea_text = fea_text.clone()
        ma_fea_vision = fea_img.mean(-2).clone()
        ma_fea_audio = fea_audio.mean(-2).clone()
        
        # iterate kwargs and print shape
        # for key, value in kwargs.items():
        #     print(f'{key}: {value.shape}')
        ### User Intro ###
        ### Audio Frames ###
         #(batch,36,12288)
        
        fea_audio, fea_text = self.co_attention_ta(v=fea_audio, s=fea_text, v_len=fea_audio.shape[1], s_len=fea_text.shape[1])
        fea_audio = torch.mean(fea_audio, -2)

        ### Image Frames ###
        #(batch,30,4096)
        fea_img, fea_text = self.co_attention_tv(v=fea_img, s=fea_text, v_len=fea_img.shape[1], s_len=fea_text.shape[1])
        fea_img = torch.mean(fea_img, -2)

        fea_text = torch.mean(fea_text, -2)

        ### C3D ###
        c3d = kwargs['c3d'] # (batch, 36, 4096)
        fea_video = self.linear_video(c3d) #(batch, frames, 128)
        if len(fea_video.shape) == 3:
            fea_video = torch.mean(fea_video, -2)

        fea_text = fea_text.unsqueeze(1)
        fea_img = fea_img.unsqueeze(1)
        fea_audio = fea_audio.unsqueeze(1)
        fea_video = fea_video.unsqueeze(1)
        # tsne_tensor = fea_text.clone()

        fea=torch.cat((fea_text, fea_audio, fea_video, fea_img), 1) # (bs, 6, 128)
        # tsne_tensor = fea.mean(1).clone()
        
        fea = self.trm(fea)
        tsne_tensor = fea.mean(1).clone()
        fea = torch.mean(fea, -2)
        
        output = self.classifier(fea)
        
        return {
            'pred': output,
            'tsne_tensor': tsne_tensor,
            'text_pos_proto': text_pos_proto,
            'text_neg_proto': text_neg_proto,
            'vision_pos_proto': vision_pos_proto,
            'vision_neg_proto': vision_neg_proto,
            'audio_pos_proto': audio_pos_proto,
            'audio_neg_proto': audio_neg_proto,
            'add_fea_text': add_fea_text_opt,
            'add_fea_vision': add_fea_vision_opt,
            'add_fea_audio': add_fea_audio_opt,
            'ori': self.ori,
            'ori_fea_text': ori_fea_text,
            'ori_fea_vision': ori_fea_vision,
            'ori_fea_audio': ori_fea_audio,
            'ma_fea_text': ma_fea_text,
            'ma_fea_vision': ma_fea_vision,
            'ma_fea_audio': ma_fea_audio,
        }
    
    def cal_loss(self, **kwargs):
        if self.ori:
            pred = kwargs['pred']
            label = kwargs['label']
            loss = F.cross_entropy(pred, label)
            return loss, loss
        
        else:
            labels = kwargs['label']
            pred = kwargs['pred']
            fea_text = kwargs['add_fea_text']
            fea_vision = kwargs['add_fea_vision']
            fea_audio = kwargs['add_fea_audio']
            text_pos_proto = kwargs['text_pos_proto']
            text_neg_proto = kwargs['text_neg_proto']
            vision_pos_proto = kwargs['vision_pos_proto']
            vision_neg_proto = kwargs['vision_neg_proto']
            audio_pos_proto = kwargs['audio_pos_proto']
            audio_neg_proto = kwargs['audio_neg_proto']

            l2_loss = 0
            orth_loss = 0

            label_pos = (labels == 1).float()
            label_neg = (labels == 0).float()

            text_l2_pos = l2_loss_fn(fea_text, text_pos_proto, label_pos)
            text_l2_neg = l2_loss_fn(fea_text, text_neg_proto, label_neg)
            text_orth_pos = orthogonal_loss(fea_text, text_pos_proto, label_neg)
            text_orth_neg = orthogonal_loss(fea_text, text_neg_proto, label_pos)
            
            vision_l2_pos = l2_loss_fn(fea_vision, vision_pos_proto, label_pos)
            vision_l2_neg = l2_loss_fn(fea_vision, vision_neg_proto, label_neg)
            vision_orth_pos = orthogonal_loss(fea_vision, vision_pos_proto, label_neg)
            vision_orth_neg = orthogonal_loss(fea_vision, vision_neg_proto, label_pos)

            audio_l2_pos = l2_loss_fn(fea_audio, audio_pos_proto, label_pos)
            audio_l2_neg = l2_loss_fn(fea_audio, audio_neg_proto, label_neg)
            audio_orth_pos = orthogonal_loss(fea_audio, audio_pos_proto, label_neg)
            audio_orth_neg = orthogonal_loss(fea_audio, audio_neg_proto, label_pos)

            if self.ablation != 'w/o-real':
                l2_loss += text_l2_pos + vision_l2_pos + audio_l2_pos
                orth_loss += text_orth_pos + vision_orth_pos + audio_orth_pos
            if self.ablation != 'w/o-fake':
                l2_loss += text_l2_neg + vision_l2_neg + audio_l2_neg
                orth_loss += text_orth_neg + vision_orth_neg + audio_orth_neg
            
            cls_loss = F.cross_entropy(pred, labels)
            loss = cls_loss + self.alpha * l2_loss + self.beta * orth_loss
            return loss, cls_loss