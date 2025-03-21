import os
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import requests
from dotenv import load_dotenv


# Ensure you have the OpenAI library installed:
# pip install openai

# Configuration for datasets
config = [
    'FakeTT',
    'FVC',
    'FakeSV',
]
dataset_dir_base = 'data'  # Base directory for datasets

class MyDataset(Dataset):
    """
    Custom Dataset to load video IDs and their corresponding title, transcript, and captions.
    """
    def __init__(self, dataset_dir):
        # vid_file = os.path.join(dataset_dir, 'vids.csv')
        # with open(vid_file, 'r', encoding='utf-8') as f:
        #     self.vids = [line.strip() for line in f if line.strip()]
        self.dataset_dir = dataset_dir
        self.data_df = pd.read_json(os.path.join(dataset_dir, 'data.jsonl'), lines=True, dtype={'vid': str})
        self.caption_df = pd.read_json(os.path.join(dataset_dir, 'retrieve', 'caption.jsonl'), lines=True, dtype={'vid': str})
        self.vids = self.data_df['vid'].tolist()
    def __len__(self):
        return len(self.vids)

    def __getitem__(self, idx):
        vid = self.vids[idx]

        # read data
        try:
            title = self.data_df[self.data_df['vid'] == vid]['title'].iloc[0]
        except Exception as e:
            title = ""
            print(f"Error reading title for vid {vid}: {e}")
        transcript = self.data_df[self.data_df['vid'] == vid]['transcript'].iloc[0]
        
        # read caption
        captions = self.caption_df[self.caption_df['vid'] == vid]['captions'].iloc[0]
        
        captions = '\n'.join(captions)

        return vid, title, transcript, captions

def collate_fn(batch):
    """
    Custom collate function to prepare batches for the DataLoader.
    """
    vids, titles, transcripts, all_captions = zip(*batch)
    return vids, titles, transcripts, all_captions

def generate_prompt(title, transcript, captions):
    """
    Constructs a prompt for the OpenAI model using title, transcript, and captions.
    """
    prompt = f"""
Video Title: {title}
Audio Transcript: {transcript}
Visual Captions: {captions}
Suppose you are a multimodal information organizing expert.
Organize the information from the visual, textual, and audio content of the given news video.
Provide a concise and accurate description that effectively represents the news video's content for the purpose of video-to-video retrieval.
The response should begin with "The video describes".
"""
    return prompt

def call_openai_api(prompt, max_retries=5, backoff_factor=2):
    """
    Calls the OpenAI API with the given prompt and handles retries using the requests library.
    """
    url = os.getenv('OPENAI_API_URL')
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": os.getenv('OPENAI_MODEL'),  # You can choose a different model if desired
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,  # Adjust based on your needs
        "temperature": 0.7  # Adjust for creativity
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Rate limit error
                wait_time = backoff_factor ** attempt
                print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"HTTP error: {e}. Retrying...")
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}. Retrying...")
        except Exception as e:
            print(f"Unexpected error: {e}. Skipping this item.")
            return "Error generating caption."
    return "Error generating caption after multiple attempts."

def generate_integrated_captions():
    """
    Main function to generate integrated captions for each video in the datasets.
    """
    for dataset_name in config:
        print(f"Processing dataset: {dataset_name}")
        
        dataset_dir = os.path.join(dataset_dir_base, dataset_name)
        output_file = os.path.join(dataset_dir, 'retrieve', 'query.jsonl')

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize the dataset and dataloader
        dataset = MyDataset(dataset_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Adjust based on your requirements
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0  # Adjust based on your CPU cores
        )
        
        print(f"Starting integrated caption generation for {dataset_name}...")
        
        # Load existing data if the file exists
        if os.path.exists(output_file):
            df_existing = pd.read_json(output_file, lines=True, dtype={'vid': str})
        else:
            df_existing = pd.DataFrame(columns=['vid', 'query'])
        pbar = tqdm(dataloader, desc=f"For {dataset_name}")
        for batch in pbar:
            vids, titles, transcripts, all_captions = batch
            batch_size = len(vids)
            pbar.set_description(f"For {dataset_name} - Processing {vids[0]}")
            
            for i in range(batch_size):
                vid = vids[i]
                
                # Skip if vid already exists
                if vid in df_existing['vid'].values:
                    continue
                
                title = titles[i]
                transcript = transcripts[i]
                captions = all_captions[i]
                
                # Construct the prompt
                prompt = generate_prompt(title, transcript, captions)
                
                # Call OpenAI API to get integrated caption
                query = call_openai_api(prompt)
                
                # Append new data to the DataFrame
                new_data = pd.DataFrame([{'vid': vid, 'query': query}])
                df_existing = pd.concat([df_existing, new_data], ignore_index=True)
                
                # Save the DataFrame to the file
                df_existing.to_json(output_file, orient='records', lines=True, force_ascii=False)
        
        print(f"Saved integrated captions to {output_file}")
        print(f"Finished processing dataset: {dataset_name}\n")

if __name__ == "__main__":
    load_dotenv()
    generate_integrated_captions()