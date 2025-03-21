import os
import shutil
import torch.nn as nn
import random
import numpy as np
import torch
from loguru import logger
import importlib
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from transformers import BatchEncoding, BatchFeature
from torch import is_tensor


class WarmUpStepLR(_LRScheduler):
    def __init__(self, optimizer, warmup_total_steps, warmup_rate, last_step=-1, **kargs):
        if warmup_rate < 0 or warmup_rate > 1:
            raise ValueError("warmup_rate should be between 0 and 1")
        self.total_steps = warmup_total_steps
        self.warmup_steps = int(warmup_total_steps * warmup_rate)
        self.last_step = last_step
        self.last_epoch = last_step
        super(WarmUpStepLR, self).__init__(optimizer, last_step)

    def get_lr(self):
        if self.last_step < self.warmup_steps:
            return [base_lr * ((self.last_step + 1) / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            factor = max(0, 1 - (self.last_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
            return [base_lr * factor for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        self.last_epoch = step 
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_worker_seed(worker_id, worker_seed):
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


def load_model(model_name: str, **kargs):
    try:
        # Attempt to import the module
        module = importlib.import_module(f"model.{model_name}.{model_name}_model")
    except ImportError:
        raise ImportError("Failed to import the 'model' module. Please ensure it exists and is in the correct path.")

    try:
        # Attempt to get the model class
        model_class = getattr(module, model_name)
    except AttributeError:
        raise ValueError(f"Model '{model_name}' not found in the 'model' module.")

    try:
        # Attempt to instantiate the model
        model = model_class(**kargs)
    except TypeError as e:
        raise TypeError(f"Error instantiating model '{model_name}': {str(e)}. Please check the provided arguments.")

    # Set the model name
    model.name = model_name

    return model


def get_dataset(model_name: str, dataset_name: str, **kargs):
    try:
        # Attempt to import the module
        module = importlib.import_module(f"model.{model_name}.{model_name}_data")
    except ImportError:
        raise ImportError("Failed to import the 'data' module. Please ensure it exists and is in the correct path.")

    try:
        # Attempt to get the dataset class
        dataset_class = getattr(module, f'{dataset_name}_{model_name}_Dataset')
    except AttributeError:
        raise ValueError(f"Dataset '{dataset_name}' not found in the 'data' module.")

    try:
        # Attempt to instantiate the dataset
        dataset = dataset_class(**kargs)
    except TypeError as e:
        raise TypeError(f"Error instantiating dataset '{dataset_name}': {str(e)}. Please check the provided arguments.")

    # Set the dataset name
    dataset.name = dataset_name

    return dataset

def get_collator(model_name: str, dataset_name: str, **kargs):
    try:
        # Attempt to import the module
        module = importlib.import_module(f"model.{model_name}.{model_name}_data")
    except ImportError:
        raise ImportError("Failed to import the 'data' module. Please ensure it exists and is in the correct path.")

    try:
        # Attempt to get the model class
        collator_class = getattr(module, f'{dataset_name}_{model_name}_Collator')
    except AttributeError:
        raise ValueError(f"Collator '{dataset_name}' not found in the 'data' module.")

    try:
        # Attempt to instantiate the model
        collator = collator_class(**kargs)
    except TypeError as e:
        raise TypeError(f"Error instantiating collator '{dataset_name}': {str(e)}. Please check the provided arguments.")

    # Set the model name
    collator.name = dataset_name

    return collator

def get_optimizer(model: nn.Module, **kargs):
    optimizer_name = kargs.pop('name')
    optimizer = None
    match optimizer_name:
        case "AdamW":
            optimizer = torch.optim.AdamW
        case "Adam":
            optimizer = torch.optim.Adam
        case _:
            raise NotImplementedError(f"Optimizer {optimizer_name} not implemented")
    return optimizer(model.parameters(), **kargs)

def get_scheduler(optimizer, **kargs):
    scheduler_name = kargs.pop('name')
    scheduler = None
    match scheduler_name:
        case "WarmUpStepLR":
            scheduler = WarmUpStepLR(optimizer, **kargs)
        case "SVFENDLR":
            def lr_lambda(step):
                p = float(step) / (100 * kargs['steps_per_epoch'])
                return 1. / (1. + 10 * p) ** 0.75
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        case "DummyLR":
            scheduler = LambdaLR(optimizer, lambda x: 1)
        case _:
            raise NotImplementedError(f"Scheduler {scheduler_name} not implemented")
    return scheduler

class BinaryClassificationMetric():
    def __init__(self, device):
        self.accuracy = Accuracy(task="multiclass", num_classes=2).to(device)
        self.f1_score = F1Score(task="multiclass", average='macro', num_classes=2).to(device)
        self.precision = Precision(task="multiclass", average='macro', num_classes=2).to(device)
        self.recall = Recall(task="multiclass", average='macro', num_classes=2).to(device)
        self.single_f1_socre = F1Score(task="multiclass", average=None, num_classes=2).to(device)
        self.single_recall = Recall(task="multiclass", average=None, num_classes=2).to(device)
        self.single_precision = Precision(task="multiclass", average=None, num_classes=2).to(device)
        
    def _reset(self):
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset() 
        self.recall.reset()
        self.single_f1_socre.reset()
        self.single_recall.reset()
        self.single_precision.reset()
    
    def update(self, preds, labels):
        self.accuracy.update(preds, labels)
        self.f1_score.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.single_f1_socre.update(preds, labels)
        self.single_recall.update(preds, labels)
        self.single_precision.update(preds, labels)
    
    def compute(self):
        acc = self.accuracy.compute()
        macro_f1 = self.f1_score.compute()
        macro_prec = self.precision.compute()
        macro_rec = self.recall.compute()
        single_f1 = self.single_f1_socre.compute()
        a_f1, b_f1 = single_f1[0], single_f1[1]
        single_prec = self.single_precision.compute()
        a_prec, b_prec = single_prec[0], single_prec[1]
        single_rec = self.single_recall.compute()
        a_rec, b_rec = single_rec[0], single_rec[1]
        self._reset()
        return {
            'acc': acc.item(),
            'macro_f1': macro_f1.item(),
            'macro_prec': macro_prec.item(),
            'macro_rec': macro_rec.item(),
            'a_f1': a_f1.item(),
            'b_f1': b_f1.item(),
            'a_prec': a_prec.item(),
            'b_prec': b_prec.item(),
            'a_rec': a_rec.item(),
            'b_rec': b_rec.item()
        }
        
class TernaryClassificationMetric():
    def __init__(self, device):
        self.accuracy = Accuracy(task="multiclass", num_classes=3).to(device)
        self.f1_score = F1Score(task="multiclass", average='macro', num_classes=3).to(device)
        self.precision = Precision(task="multiclass", average='macro', num_classes=3).to(device)
        self.recall = Recall(task="multiclass", average='macro', num_classes=3).to(device)
        self.single_f1_socre = F1Score(task="multiclass", average=None, num_classes=3).to(device)
        self.single_recall = Recall(task="multiclass", average=None, num_classes=3).to(device)
        self.single_precision = Precision(task="multiclass", average=None, num_classes=3).to(device)
        
    def _reset(self):
        self.accuracy.reset()
        self.f1_score.reset()
        self.precision.reset() 
        self.recall.reset()
        self.single_f1_socre.reset()
        self.single_recall.reset()
        self.single_precision.reset()
    
    def update(self, preds, labels):
        self.accuracy.update(preds, labels)
        self.f1_score.update(preds, labels)
        self.precision.update(preds, labels)
        self.recall.update(preds, labels)
        self.single_f1_socre.update(preds, labels)
        self.single_recall.update(preds, labels)
        self.single_precision.update(preds, labels)
    
    def compute(self):
        acc = self.accuracy.compute()
        macro_f1 = self.f1_score.compute()
        macro_prec = self.precision.compute()
        macro_rec = self.recall.compute()
        single_f1 = self.single_f1_socre.compute()
        a_f1, b_f1, c_f1 = single_f1[0], single_f1[1], single_f1[2]
        single_prec = self.single_precision.compute()
        a_prec, b_prec, c_prec = single_prec[0], single_prec[1], single_prec[2]
        single_rec = self.single_recall.compute()
        a_rec, b_rec, c_rec = single_rec[0], single_rec[1], single_rec[2]
        self._reset()
        return {
            'acc': acc.item(),
            'macro_f1': macro_f1.item(),
            'macro_prec': macro_prec.item(),
            'macro_rec': macro_rec.item(),
            'a_f1': a_f1.item(),
            'b_f1': b_f1.item(),
            'c_f1': c_f1.item(),
            'a_prec': a_prec.item(),
            'b_prec': b_prec.item(),
            'c_prec': c_prec.item(),
            'a_rec': a_rec.item(),
            'b_rec': b_rec.item(),
            'c_rec': c_rec.item()
        }

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def is_movable(obj):
    if is_tensor(obj):
        return True
    elif isinstance(obj, BatchEncoding):
        return True
    elif isinstance(obj, BatchFeature):
        return True
    else:
        return False

def copy_config_file():
    src_dir = 'src/model'
    dest_dir = 'src/config'
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.yaml'):
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                shutil.copy2(src_file_path, dest_file_path)