import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import logging

# Ensure src directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
from torch.utils.data import DataLoader

# Project utilities
from utils.core_utils import (
    load_model,
    get_dataset,
    get_collator,
    BinaryClassificationMetric,
)

# Dataset aliases
DATASET_ALIAS = {
    "FakeSV": "FSV",
    "FakeTT": "FTT",
    "FVC": "FVC",
}
DATASETS = list(DATASET_ALIAS.keys())

# Model name aliases
MODEL_ALIAS = {
    "SV-FEND": "SVFEND",
    "FakingRec": "FakeRec",
}

# Suppress noisy warnings/logging
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("jieba").setLevel(logging.ERROR)


def canonical_model(name: str) -> str:
    return MODEL_ALIAS.get(name, name)


def build_dataloader(model_name: str, dataset_name: str, cfg: Dict, batch_size: int) -> DataLoader:
    model_name = canonical_model(model_name)
    
    # Build a clean dataloader config for the target dataset
    source_data_args = cfg.get("data", {})
    
    target_data_args = {}
    
    # Carry over necessary structure-related args only
    if "num_pos" in source_data_args:
        target_data_args["num_pos"] = source_data_args["num_pos"]
    if "num_neg" in source_data_args:
        target_data_args["num_neg"] = source_data_args["num_neg"]
    if "ablation" in source_data_args:
        target_data_args["ablation"] = source_data_args["ablation"]
    
    # Task type
    task_arg = cfg.get("task", "binary")

    # Build the target dataset dataloader (test split)
    dataset = get_dataset(model_name, dataset_name, fold="default", split="test", task=task_arg, **target_data_args)

    collator = get_collator(model_name, dataset_name, **target_data_args)
    cpu_count = os.cpu_count() or 4
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, shuffle=False, num_workers=min(cpu_count, batch_size // 2))


def evaluate(model, dataloader: DataLoader, device: str):
    evaluator = BinaryClassificationMetric(device)
    model.eval()

    # Print debug info only for the first batch
    printed_for_first_batch = False

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            vids = batch.pop("vids", None)
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            labels = inputs.pop("labels")

            # --- DEBUG START ---
            if not printed_for_first_batch:
                print("\n--- CROSS-PLATFORM DEBUG ---")
                print(f"Sample VIDs: {vids[:10] if vids else 'N/A'}")
                print(f"Labels content: {labels[:30]}")

            output = model(**inputs)
            pred = output["pred"] if isinstance(output, dict) else output
            _, preds = torch.max(pred, 1)

            if not printed_for_first_batch:
                print(f"Predictions:    {preds[:30]}")
                printed_for_first_batch = True
            # --- DEBUG END ---

            evaluator.update(preds, labels)
            
    metrics = evaluator.compute()
    return metrics["acc"], metrics["macro_f1"]


def run_cross_platform(model_name: str, ckpt_path: str, config_path: str, src_dataset: str, batch_size: int = 128):
    """Run cross-platform evaluation with direct model loading."""
    import yaml
    
    # Load configuration
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    try:
        model = load_model(canonical_model(model_name), **cfg.get("para", {})).to(device)
    except Exception as e:
        print(f"[ERROR] failed to load model {model_name} => {e}")
        return
    
    # Load checkpoint
    try:
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[ERROR] failed to load checkpoint {ckpt_path} => {e}")
        return

    acc_f1_values: List[str] = []

    for tgt_dataset in DATASETS:
        if tgt_dataset == src_dataset:
            continue
        dataloader = build_dataloader(model_name, tgt_dataset, cfg, batch_size)
        acc, f1 = evaluate(model, dataloader, device)
        acc_percent = round(acc * 100, 2)
        f1_percent = round(f1 * 100, 2)
        acc_f1_values.extend([f"{acc_percent:.2f}", f"{f1_percent:.2f}"])

    print(" & ".join(acc_f1_values))
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cross-platform evaluation: load model by checkpoint and config to evaluate transfer across datasets.")
    parser.add_argument("--model", required=True, type=str, help="Model name")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", required=True, type=str, help="Path to configuration file (.yaml)")
    parser.add_argument("--src_dataset", required=True, type=str, choices=DATASETS, help="Source dataset name")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    run_cross_platform(
        args.model,
        args.ckpt,
        args.config,
        args.src_dataset,
        batch_size=args.batch_size,
    ) 