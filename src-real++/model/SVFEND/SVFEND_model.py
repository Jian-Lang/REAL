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
from ..Base.utils import l2_loss_fn


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
        return proto.mean(-2)

class AddLinear(nn.Module):
    def __init__(self, fea_dim=256):
        super(AddLinear, self).__init__()
        self.linear = nn.LazyLinear(fea_dim)
        
    def forward(self, x):
        return self.linear(x)

class SVFEND(nn.Module):
    def __init__(self, weight_lambda=0.5, encoder_name='bert-base-uncased', fea_dim=128, dropout=0.1, ori=False, loss_alpha=1.0, loss_beta=1.0, **kargs):
        super(SVFEND, self).__init__()

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
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
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
        
        self.add_linear_text = AddLinear(fea_dim)
        self.add_linear_vision = AddLinear(fea_dim)
        self.add_linear_audio = AddLinear(fea_dim)
        self.ori = ori
        self.weight_lambda = weight_lambda

    def forward(self, **kwargs):
        fea_text_pos = kwargs['text_fea_pos']
        fea_text_neg = kwargs['text_fea_neg']
        fea_vision_pos = kwargs['vision_fea_pos']
        fea_vision_neg = kwargs['vision_fea_neg']
        fea_audio_pos = kwargs['audio_fea_pos']
        fea_audio_neg = kwargs['audio_fea_neg']
        
        text_fea = kwargs['text_fea']
        fea_text = self.linear_text(text_fea) 
        
        audioframes=kwargs['audioframes']
        fea_audio = self.vggish_modified(audioframes) #(batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio) 
        
        frames=kwargs['frames']
        fea_img = self.linear_img(frames)
        
        text_pos_proto = self.text_pos_self_attn(fea_text_pos)
        text_neg_proto = self.text_neg_self_attn(fea_text_neg)
        vision_pos_proto = self.vision_pos_self_attn(fea_vision_pos)
        vision_neg_proto = self.vision_neg_self_attn(fea_vision_neg)
        audio_pos_proto = self.audio_pos_self_attn(fea_audio_pos)
        audio_neg_proto = self.audio_neg_self_attn(fea_audio_neg)
        
        add_fea_text = self.add_linear_text(fea_text)
        add_fea_vision = self.add_linear_vision(fea_img)
        add_fea_audio = self.add_linear_audio(fea_audio)
        
        add_fea_text_opt = add_fea_text
        add_fea_vision_opt = add_fea_vision.mean(-2)
        add_fea_audio_opt = add_fea_audio.mean(-2)
        vote_audio_pos_proto = F.cosine_similarity(add_fea_audio_opt, audio_pos_proto, dim=-1)
        vote_audio_neg_proto = F.cosine_similarity(add_fea_audio_opt, audio_neg_proto, dim=-1)
        vote_vision_pos_proto = F.cosine_similarity(add_fea_vision_opt, vision_pos_proto, dim=-1)
        vote_vision_neg_proto = F.cosine_similarity(add_fea_vision_opt, vision_neg_proto, dim=-1)
        vote_text_pos_proto = F.cosine_similarity(add_fea_text_opt, text_pos_proto, dim=-1)
        vote_text_neg_proto = F.cosine_similarity(add_fea_text_opt, text_neg_proto, dim=-1)

        vote_real = (vote_audio_pos_proto + vote_vision_pos_proto + vote_text_pos_proto) / 3 * 20 # scale factor for cos_sim fed in softmax
        vote_fake = (vote_audio_neg_proto + vote_vision_neg_proto + vote_text_neg_proto) / 3 * 20
        votes = torch.stack([vote_real, vote_fake], dim=-1)
        ori_fea_text = fea_text.clone()
        ori_fea_vision = fea_img.mean(-2).clone()
        ori_fea_audio = fea_audio.mean(-2).clone()
        
        if not self.ori:
            fea_text = fea_text + add_fea_text
            fea_img = fea_img + add_fea_vision
            fea_audio = fea_audio + add_fea_audio
        
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

        fea=torch.cat((fea_text, fea_audio, fea_video, fea_img), 1)
        
        fea = self.trm(fea)
        tsne_tensor = fea.mean(1).clone()
        fea = torch.mean(fea, -2)

        cls_output = self.classifier(fea)
        if self.ori:
            output = cls_output
        else:
            output = self.weight_lambda * torch.softmax(cls_output, dim=-1) + (1 - self.weight_lambda) * torch.softmax(votes, dim=-1)
        return {
            'pred': output,
            'cls_output': cls_output,
            'votes': votes,
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
            'ma_fea_text': add_fea_text_opt,
            'ma_fea_vision': add_fea_vision_opt,
            'ma_fea_audio': add_fea_audio_opt,
        }
    
    def cal_loss(self, **kwargs):
        label = kwargs['label']
        cls_output = kwargs['cls_output']
        votes = kwargs['votes']
        loss = torch.tensor(0.0, device=label.device)
        votes_loss = torch.tensor(0.0, device=label.device)
        pull_loss = torch.tensor(0.0, device=label.device)
        push_loss = torch.tensor(0.0, device=label.device)

        cls_loss = F.cross_entropy(cls_output, label)
        loss += cls_loss
        if self.ori:  
            return loss, cls_loss

        fea_text = kwargs['add_fea_text']
        fea_vision = kwargs['add_fea_vision']
        fea_audio = kwargs['add_fea_audio']
        
        text_pos_proto = kwargs['text_pos_proto']
        text_neg_proto = kwargs['text_neg_proto']
        vision_pos_proto = kwargs['vision_pos_proto']
        vision_neg_proto = kwargs['vision_neg_proto']
        audio_pos_proto = kwargs['audio_pos_proto']
        audio_neg_proto = kwargs['audio_neg_proto']

        label_pos = (label == 0).float()
        label_neg = (label == 1).float()

        pull_text_pos = l2_loss_fn(fea_text, text_pos_proto, label_pos)
        pull_text_neg = l2_loss_fn(fea_text, text_neg_proto, label_neg)
        push_text_pos = -l2_loss_fn(fea_text, text_neg_proto, label_pos)
        push_text_neg = -l2_loss_fn(fea_text, text_pos_proto, label_neg)

        pull_vision_pos = l2_loss_fn(fea_vision, vision_pos_proto, label_pos)
        pull_vision_neg = l2_loss_fn(fea_vision, vision_neg_proto, label_neg)
        push_vision_pos = -l2_loss_fn(fea_vision, vision_neg_proto, label_pos)
        push_vision_neg = -l2_loss_fn(fea_vision, vision_pos_proto, label_neg)

        pull_audio_pos = l2_loss_fn(fea_audio, audio_pos_proto, label_pos)
        pull_audio_neg = l2_loss_fn(fea_audio, audio_neg_proto, label_neg)
        push_audio_pos = -l2_loss_fn(fea_audio, audio_neg_proto, label_pos)
        push_audio_neg = -l2_loss_fn(fea_audio, audio_pos_proto, label_neg)

        pull_loss += pull_text_pos + pull_vision_pos + pull_audio_pos
        pull_loss += pull_text_neg + pull_vision_neg + pull_audio_neg
        pull_loss /= 6
        push_loss += push_text_pos + push_vision_pos + push_audio_pos
        push_loss += push_text_neg + push_vision_neg + push_audio_neg
        push_loss /= 6

        loss += self.loss_alpha * pull_loss
        loss += self.loss_alpha * push_loss

        counts = torch.bincount(label, minlength=2)
        class_weights = None
        if counts[0] > 0 and counts[1] > 0:
            n_samples = len(label)
            class_weights = torch.tensor([n_samples / (2. * counts[0]), n_samples / (2. * counts[1])], device=label.device)

        votes_loss = F.cross_entropy(votes, label, weight=class_weights)
        loss += self.loss_beta * votes_loss

        return loss, cls_loss