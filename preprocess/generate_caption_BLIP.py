import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, Blip2ForConditionalGeneration

# Configuration for datasets and the BLIP2 model
config = [
    ['FakeSV', 'Salesforce/blip2-opt-2.7b'],
    ['FakeTT', 'Salesforce/blip2-opt-2.7b'],
    ['FVC', 'Salesforce/blip2-opt-2.7b']
]

dataset_dir_base = 'data'  # Base directory for datasets
frames_path = 'frames_16'  # Directory where frames are stored

class MyDataset(Dataset):
    """
    Custom Dataset to load video IDs and their corresponding frames.
    """
    def __init__(self, dataset_dir):
        vid_file = os.path.join(dataset_dir, 'vids.csv')
        with open(vid_file, 'r') as f:
            self.vids = [line.strip() for line in f]
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]
        frames = []
        for i in range(16):  # Assuming 16 frames per video
            frame_path = os.path.join(
                self.dataset_dir,
                frames_path,
                f'{vid}',
                f'frame_{i:03d}.jpg'
            )
            if os.path.exists(frame_path):
                frame = Image.open(frame_path).convert('RGB')
                frames.append(frame)
            else:
                # Substitute missing frames with a black image
                frames.append(Image.new('RGB', (224, 224), color='black'))
        return vid, frames

def collate_fn(batch):
    """
    Custom collate function to prepare batches for the DataLoader.
    """
    vids, all_frames = zip(*batch)  # Unzip the batch
    # Flatten the list of frames
    all_frames = [frame for frames in all_frames for frame in frames]
    return vids, all_frames

def generate_captions():
    """
    Main function to generate captions for each frame in the datasets.
    """
    for cfg in config:
        dataset_name, model_id = cfg
        print(f"Processing dataset: {dataset_name}")
        
        dataset_dir = os.path.join(dataset_dir_base, dataset_name)
        output_file = os.path.join(dataset_dir, 'retrieve', 'caption.jsonl')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"Loading model: {model_id}")
        # Load the BLIP2 processor and model
        processor = AutoProcessor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id)
        
        # Move model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize the dataset and dataloader
        dataset = MyDataset(dataset_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Adjust based on your GPU memory
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=8  # Adjust based on your CPU cores
        )
        
        print(f"Starting caption generation for {dataset_name}...")
        
        with torch.no_grad():
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for vids, all_frames in tqdm(dataloader, desc=f"Generating captions for {dataset_name}"):
                    # Move frames to device
                    inputs = processor(images=all_frames, return_tensors="pt").to(device)
                    
                    # Generate captions
                    outputs = model.generate(**inputs, max_new_tokens=50)  # Adjust max_new_tokens as needed
                    
                    # Decode captions
                    captions = processor.batch_decode(outputs, skip_special_tokens=True)
                    
                    # Group captions per video
                    batch_size = len(vids)
                    for i in range(batch_size):
                        vid = vids[i]
                        video_captions = captions[i*16:(i+1)*16]  # Assuming 16 frames per video
                        json_line = json.dumps({
                            'vid': vid,
                            'captions': video_captions
                        }, ensure_ascii=False)
                        f_out.write(json_line + '\n')
        
        print(f"Saved captions to {output_file}")
        print(f"Finished processing dataset: {dataset_name}\n")

if __name__ == "__main__":
    generate_captions()