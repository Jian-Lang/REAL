import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import av
from PIL import Image
from tqdm import tqdm
import os
import numpy as np
from C3D_model import C3D


config = [
    ['FakeTT'], ['FVC'], ['FakeSV']
]

NUM_FRAMES = 16


c3d = C3D(nb_classes=487)  
c3d.load_state_dict(torch.load('pretrained/C3D/c3d.pickle'))  
c3d = c3d.cuda()
c3d.eval()

def preprocess_for_c3d(pil_image):
    pil_image = pil_image.resize((112, 112))
    tensor = torch.FloatTensor(np.array(pil_image)).permute(2, 0, 1)
    tensor = tensor / 255.0
    return tensor

def robust_frame_extraction(video_path, num_frames):
    pil_images = []
    
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            total_frames = stream.frames
            duration = stream.duration * stream.time_base
            
            if total_frames == 0 or duration <= 0:
                raise ValueError(f"The video has no valid frames or duration: {video_path}")
            
            target_timestamps = [t * duration / num_frames for t in range(num_frames)]
            
            for timestamp in target_timestamps:
                container.seek(int(timestamp * stream.time_base.denominator), stream=stream)
                for frame in container.decode(video=0):
                    pil_image = frame.to_image()
                    pil_images.append(pil_image)
                    break
    
    except Exception as e:
        print(f"Error processing video {video_path}: {str(e)}")
    
    if len(pil_images) < num_frames:
        last_frame = pil_images[-1] if pil_images else Image.new('RGB', (112, 112), color='black')
        pil_images.extend([last_frame] * (num_frames - len(pil_images)))
    
    return pil_images[:num_frames]

class MyDataset(Dataset):
    def __init__(self, src_file, video_dir):
        self.data = pd.read_json(src_file, lines=True, dtype={'vid': str})
        self.video_dir = video_dir
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid = self.data.iloc[index]['vid']
        video_path = os.path.join(self.video_dir, f'{vid}.mp4')
        pil_images = robust_frame_extraction(video_path, NUM_FRAMES)
        return vid, pil_images

def customed_collate_fn(batch):
    vids, pil_images = zip(*batch)
    images = []
    for images_list in pil_images:
        images.extend([preprocess_for_c3d(img) for img in images_list])
    inputs = torch.stack(images)
    inputs = inputs.view(-1, 3, NUM_FRAMES, 112, 112)
    return vids, inputs

def extract_c3d_features(model, inputs):
    _, features = model(inputs)

    return features

for dataset_config in config:
    dataset = dataset_config[0]
    output_file = os.path.join(f'data/{dataset}/fea', 'SVFEND/c3d_features.pt')
    
    if os.path.exists(output_file):
        print(f'Skipping {dataset} as features already exist')
        continue
        
    print(f'Processing {dataset}...')
    
    src_file = f'data/{dataset}/data.jsonl'
    video_dir = f'data/{dataset}/videos'
    os.makedirs(f'data/{dataset}/fea', exist_ok=True)

    save_dict = {}
    dataloader = DataLoader(MyDataset(src_file, video_dir), batch_size=16, collate_fn=customed_collate_fn, num_workers=8)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            vids, inputs = batch
            inputs = inputs.cuda()
            batch_size = inputs.size(0)
            
            features = extract_c3d_features(c3d, inputs)
            features = features.view(batch_size, -1)
            assert features.shape == (batch_size, 4096)

            features = features.detach().cpu()
            
            for i, vid in enumerate(vids):
                save_dict[vid] = features[i]


    torch.save(save_dict, output_file)