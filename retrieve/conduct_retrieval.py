import torch
import pandas as pd
import numpy as np
import json
from scipy.spatial.distance import cdist
from pathlib import Path
from tqdm import tqdm


input_list = [
    ('FakeSV'),
    ('FakeTT'),
    ('FVC'),
]


def compute_similarities_self(query_ids, base_ids, features, labels_dict, topk=100, batch_size=100, ignore_self=True):
    query_vectors = np.array([features[id] for id in query_ids if id in features])
    base_vectors = np.array([features[id] for id in base_ids if id in features])
    
    results = []
    
    for i in tqdm(range(0, len(query_ids), batch_size)):
        batch_ids = query_ids[i:i+batch_size]
        batch_vectors = query_vectors[i:i+batch_size]
        base_id_to_index = {id: idx for idx, id in enumerate(base_ids)}
        
        sims = 1 - cdist(batch_vectors, base_vectors, metric='cosine')
        
        for j, sim in enumerate(sims):
            query_id = batch_ids[j]
            
            if ignore_self:
                self_index = base_id_to_index.get(query_id, -1)
                if self_index != -1 and self_index < len(sim):
                    sim[self_index] = -np.inf
            
            # 分别存储标签为 0 和 1 的结果
            results_0 = []
            results_1 = []
            
            for idx, similarity in sorted(enumerate(sim), key=lambda x: x[1], reverse=True):
                base_id = base_ids[idx]
                label = labels_dict.get(base_id, -1)  # 如果没有标签,默认为 -1
                
                if label == 0 and len(results_0) < topk:
                    results_0.append({"vid": base_id, "similarity": float(similarity)})
                elif label == 1 and len(results_1) < topk:
                    results_1.append({"vid": base_id, "similarity": float(similarity)})
                
                if len(results_0) >= topk and len(results_1) >= topk:
                    break
            
            results.append({
                'vid': query_id, 
                'similarities': [
                    {'vid': [r['vid'] for r in results_0], 'sim': [r['similarity'] for r in results_0]},
                    {'vid': [r['vid'] for r in results_1], 'sim': [r['similarity'] for r in results_1]}
                ]
            })
    
    return results

for dataset in input_list:
    # Set dataset path
    dataset_path = Path('data') / dataset
    
    # Load feature
    feature = torch.load(dataset_path / 'retrieve' / 'fea_retrieval_text_gte-large.pt', weights_only=True)
    
    # Set output path
    output_path = dataset_path / 'retrieve' / 'sim.jsonl'

    print(f'Processing {dataset}')
    print(f'Feature shape: {next(iter(feature.values())).shape}')

    # Load train, validation, and test video IDs
    train_vids = pd.read_csv(dataset_path / 'vids' / 'vid_time3_train.txt', header=None, dtype=str)[0].tolist()
    valid_vids = pd.read_csv(dataset_path / 'vids' / 'vid_time3_valid.txt', header=None, dtype=str)[0].tolist()
    test_vids = pd.read_csv(dataset_path / 'vids' / 'vid_time3_test.txt', header=None, dtype=str)[0].tolist()

    train_valid_vids = list(set(train_vids + valid_vids))
    all_vids = list(set(train_valid_vids + test_vids))

    # Load labels
    labels = pd.read_json(dataset_path / 'label.jsonl', lines=True, dtype={'vid': str})
    labels_dict = labels.set_index('vid')['label'].to_dict()
    
    # only select all_vids and train_valid_vids by they are in labels_dict
    all_vids = [vid for vid in all_vids if vid in labels_dict]
    train_valid_vids = [vid for vid in train_valid_vids if vid in labels_dict]

    # Compute similarities
    result = compute_similarities_self(all_vids, train_valid_vids, feature, labels_dict, topk=100)

    # Save results
    with open(output_path, 'w') as f:
        for item in result:
            f.write(json.dumps(item) + '\n')

    print(f'Results saved to {output_path}')
    print('----------------------------')


