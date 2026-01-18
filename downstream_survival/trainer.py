"""
Universal Trainer Class
Supports different model types and training configurations
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import label_binarize
import pandas as pd

import sys
import os

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Only import necessary modules, avoid dependency on utils.utils
from models.model_factory import ModelFactory
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import AUROC as TM_AUROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_serializable(obj: Any) -> Any:
    """Universal JSON-safe serialization converter

    - Convert numpy scalars to Python scalars
    - Convert numpy arrays to lists
    - Move torch tensors to CPU and convert to lists
    - Convert other non-serializable objects to strings
    """
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if obj is np.nan:
        return None
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # torch tensors
    if torch.is_tensor(obj):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            return str(obj)
    # Fallback to string for other common non-serializable types
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	"""
	Save dataset split information (using patient_id/case_id instead of indices for reproducibility)

	Key fix: Extract case_ids from original dataset in Subset objects, using actual case_id instead of indices
	This way splits can be correctly matched even if dataset order differs
	"""
	try:
		# Get case_ids for each split (extract from original dataset in Subset)
		splits = []
		for i, dataset in enumerate(split_datasets):
			if hasattr(dataset, 'case_ids'):
				# Direct MultimodalDataset object
				splits.append(pd.Series(dataset.case_ids))
			elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
				# Is Subset object, need to extract case_ids from original dataset
				base_dataset = dataset.dataset
				indices = dataset.indices
				
				if hasattr(base_dataset, 'case_ids'):
					# Extract corresponding case_id from original dataset's case_ids
					base_case_ids = base_dataset.case_ids
					if isinstance(base_case_ids, list):
						case_ids = [base_case_ids[idx] for idx in indices]
					else:
						# If other type (like numpy array), convert to list
						case_ids = [base_case_ids[idx] for idx in indices]
					splits.append(pd.Series(case_ids))
				else:
					# fallback: use indices
					splits.append(pd.Series([f"sample_{j}" for j in indices]))
			else:
				# fallback: use indices
				splits.append(pd.Series([f"sample_{j}" for j in range(len(dataset))]))
		
		if not boolean_style:
			# Create DataFrame, each column is a split's case_ids
			# Use longest split as DataFrame length, pad shorter ones with NaN
			max_len = max(len(s) for s in splits) if splits else 0
			
			# Create dictionary, each key corresponds to a split's case_ids
			data_dict = {}
			for i, col_key in enumerate(column_keys):
				if i < len(splits):
					case_ids = splits[i].tolist()
					# Pad with NaN to make lengths consistent
					while len(case_ids) < max_len:
						case_ids.append(None)
					data_dict[col_key] = case_ids
				else:
					data_dict[col_key] = [None] * max_len
			
			df = pd.DataFrame(data_dict)
			# Remove rows that are all NaN
			df = df.dropna(how='all')
		else:
			df = pd.concat(splits, ignore_index = True, axis=0)
			index = df.values.tolist()
			one_hot = np.eye(len(split_datasets)).astype(bool)
			bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
			df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

		df.to_csv(filename, index=False)
		print(f"✅ Saved split information to: {filename} (using case_id)")
	except Exception as e:
		print(f"⚠️ Failed to save split information: {e}")
		import traceback
		traceback.print_exc()
		# Create a simple split record
		split_info = {
			'split_type': column_keys,
			'train_size': len(split_datasets[0]) if len(split_datasets) > 0 else 0,
			'val_size': len(split_datasets[1]) if len(split_datasets) > 1 else 0,
			'test_size': len(split_datasets[2]) if len(split_datasets) > 2 else 0
		}
		pd.DataFrame([split_info]).to_csv(filename, index=False)
		print(f"✅ Saved simplified split information to: {filename}")

def print_network(model: nn.Module):
    """Print network architecture and parameter statistics"""
    print("=" * 50)
    print("Model Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    # Calculate parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 50)

def get_optim(model: nn.Module, opt: str, lr: float, reg: float) -> torch.optim.Optimizer:
    """Get optimizer"""
    if opt == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=lr, weight_decay=reg)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=lr, momentum=0.9, weight_decay=reg)
    else:
        raise NotImplementedError
    
    return optimizer

def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler

    Args:
        optimizer: Optimizer
        scheduler_config: Scheduler configuration dictionary

    Returns:
        Learning rate scheduler or None
    """
    scheduler_type = scheduler_config.get('type', None)
    
    if scheduler_type is None:
        return None
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 50)
        gamma = scheduler_config.get('gamma', 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', 200)
        eta_min = scheduler_config.get('eta_min', 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'cosine_warm_restart':
        T_0 = scheduler_config.get('T_0', 10)  # First restart cycle length
        T_mult = scheduler_config.get('T_mult', 2)  # Cycle length multiplication factor
        eta_min = scheduler_config.get('eta_min', 0.0)  # Minimum learning rate
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
    
    elif scheduler_type == 'plateau':
        mode = scheduler_config.get('mode', 'min')
        patience = scheduler_config.get('patience', 10)
        factor = scheduler_config.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, patience=patience, factor=factor, verbose=True
        )
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    else:
        print(f"⚠️ Unknown scheduler type: {scheduler_type}")
        return None

def get_split_loader(dataset, training=False, weighted=False, batch_size=1, generator=None):
    """Get data loader

    Args:
        dataset: Dataset
        training: Whether in training mode
        weighted: Whether to use weighted sampling
        batch_size: Batch size
        generator: Random number generator (for consistent sampling order)
    """
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(dataset)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), generator=generator)
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def make_weights_for_balanced_classes_split(dataset):
    """Create weights for balanced classes"""
    N = float(len(dataset))
    
    # Get labels, adapt to MultimodalDataset format
    labels = []
    unique_labels = set()
    
    for i in range(len(dataset)):
        # Handle Subset objects
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'get_label'):
            # Get original dataset index through Subset's indices
            original_idx = dataset.indices[i]
            label = dataset.dataset.get_label(original_idx)
        else:
            # Handle MultimodalDataset directly
            label = dataset.get_label(i)
        
        unique_labels.add(label)
        labels.append(label)
    
    # Use dataset's label mapping
    if hasattr(dataset, 'label_to_int'):
        label_to_int = dataset.label_to_int
    else:
        # If no label mapping, create default mapping
        label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # Convert string labels to numbers
    numeric_labels = [label_to_int[label] for label in labels]
    labels = np.array(numeric_labels)
    
    class_counts = np.bincount(labels)
    class_weights = N / class_counts
    weights = [class_weights[labels[i]] for i in range(len(dataset))]
    return torch.DoubleTensor(weights)


class Logger:
    """
    Unified training metrics logger
    Integrates accuracy statistics, training log recording, and best metrics tracking
    """
    
    def __init__(self, n_classes: int, log_dir: str = None, fold: int = 0):
        """
        Initialize metrics logger

        Args:
            n_classes: Number of classes
            log_dir: Log save directory (optional)
            fold: Fold index
        """
        self.n_classes = n_classes
        self.log_dir = log_dir
        self.fold = fold
        
        # Class statistics
        self.batch_log = {
            'class_stats': [{"count": 0, "correct": 0} for _ in range(self.n_classes)],
            'labels': [],
            'probs': [],
            'loss': 0.0
        }
        
        # Training logs
        self.epoch_logs = []
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_val_auc': 0.0,
            'best_epoch': 0
        }
        
        # Initialize file logging (if log_dir provided)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.csv_path = os.path.join(log_dir, f'fold_{fold}_training_log.csv')
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'epoch', 'train_loss', 'train_acc', 'train_auc',
                    'val_loss', 'val_acc', 'val_auc', 'learning_rate',
                    'timestamp'
                ])
    
    def reset_epoch_stats(self):
        """Reset current epoch statistics"""
        self.batch_log = {
            'class_stats': [{"count": 0, "correct": 0} for _ in range(self.n_classes)],
            'labels': [],
            'probs': [],
            'loss': 0.0
        }
    
    def log_batch(self, Y_hat, Y, Y_prob, loss):
        """
        Log batch predictions, loss, labels, probs

        Args:
            Y_hat: Predicted results (int, tensor, or array)
            Y: True labels (int, tensor, or array)
            Y_prob: Prediction probabilities (tensor or array)
            loss: Loss value (tensor)
        """
        # Convert to Tensor uniformly (not to numpy), for subsequent torch.cat
        if not torch.is_tensor(Y_hat):
            Y_hat = torch.as_tensor(Y_hat)
        if not torch.is_tensor(Y):
            Y = torch.as_tensor(Y)
        if not torch.is_tensor(Y_prob):
            Y_prob = torch.as_tensor(Y_prob)

        # Count classification correct numbers
        if Y_hat.numel() == 1 and Y.numel() == 1:
            label_class = int(Y.item())
            self.batch_log['class_stats'][label_class]["count"] += 1
            self.batch_log['class_stats'][label_class]["correct"] += (int(Y_hat.item() == Y.item()))
        else:
            unique_labels = torch.unique(Y)
            for label_class in unique_labels.tolist():
                cls_mask = (Y == label_class)
                self.batch_log['class_stats'][label_class]["count"] += int(cls_mask.sum().item())
                self.batch_log['class_stats'][label_class]["correct"] += int((Y_hat[cls_mask] == Y[cls_mask]).sum().item())

        # Append to logs (keep as Tensor)
        self.batch_log['labels'].append(Y)
        self.batch_log['probs'].append(Y_prob)
        self.batch_log['loss'] += float(loss.item())
    
    def get_class_accuracy(self, class_idx: int) -> Tuple[Optional[float], int, int]:
        """
        Get accuracy for specified class

        Returns:
            (accuracy, correct_count, total_count)
        """
        count = self.batch_log['class_stats'][class_idx]["count"]
        correct = self.batch_log['class_stats'][class_idx]["correct"]
        
        if count == 0:
            return None, correct, count
        else:
            return float(correct) / count, correct, count
    
    def get_overall_accuracy(self) -> float:
        """Get overall accuracy"""
        total_correct = sum(stat["correct"] for stat in self.batch_log['class_stats'])
        total_count = sum(stat["count"] for stat in self.batch_log['class_stats'])
        
        if total_count == 0:
            return 0.0
        return float(total_correct) / total_count
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """
        Log epoch metrics

        Args:
            epoch: Current epoch
            train_metrics: Training metrics dictionary
            val_metrics: Validation metrics dictionary
            lr: Learning rate
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_metrics.get('loss', 0.0),
            'train_acc': train_metrics.get('acc', 0.0),
            'train_auc': train_metrics.get('auc', 0.0),
            'val_loss': val_metrics.get('loss', 0.0),
            'val_acc': val_metrics.get('acc', 0.0),
            'val_auc': val_metrics.get('auc', 0.0),
            'learning_rate': lr,
            'timestamp': timestamp
        }
        
        self.epoch_logs.append(epoch_log)
        
        # Write to CSV file (if file logging is enabled)
        if self.log_dir:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_log['epoch'], epoch_log['train_loss'], epoch_log['train_acc'], 
                    epoch_log['train_auc'], epoch_log['val_loss'], epoch_log['val_acc'], 
                    epoch_log['val_auc'], epoch_log['learning_rate'], epoch_log['timestamp']
                ])
        
        # Update best metrics
        self._update_best_metrics(epoch, val_metrics)
        
        # Print progress
        print(f"📊 Epoch {epoch:3d} | "
              f"Train: Loss={train_metrics.get('loss', 0.0):.4f}, "
              f"Acc={train_metrics.get('acc', 0.0):.4f}, "
              f"AUC={train_metrics.get('auc', 0.0):.4f} | "
              f"Val: Loss={val_metrics.get('loss', 0.0):.4f}, "
              f"Acc={val_metrics.get('acc', 0.0):.4f}, "
              f"AUC={val_metrics.get('auc', 0.0):.4f}")
    
    def _update_best_metrics(self, epoch: int, val_metrics: Dict):
        """Update best metrics"""
        val_loss = val_metrics.get('loss', float('inf'))
        val_acc = val_metrics.get('acc', 0.0)
        val_auc = val_metrics.get('auc', 0.0)
        
        if val_loss < self.best_metrics['best_val_loss']:
            self.best_metrics['best_val_loss'] = val_loss
            self.best_metrics['best_epoch'] = epoch
            
        if val_acc > self.best_metrics['best_val_acc']:
            self.best_metrics['best_val_acc'] = val_acc
            
        if val_auc > self.best_metrics['best_val_auc']:
            self.best_metrics['best_val_auc'] = val_auc
    
    def save_summary(self, test_metrics: Dict = None):
        """Save training summary"""
        summary = {
            'fold': self.fold,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.epoch_logs),
            'final_epoch': self.epoch_logs[-1] if self.epoch_logs else None,
            'test_metrics': test_metrics
        }
        
        if self.log_dir:
            # Save JSON summary
            summary_path = os.path.join(self.log_dir, f'fold_{self.fold}_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=to_serializable)
        
        # Print summary
        print(f"\n🎯 Fold {self.fold} training summary:")
        print(f"   Best validation loss: {self.best_metrics['best_val_loss']:.4f} (Epoch {self.best_metrics['best_epoch']})")
        print(f"   Best validation accuracy: {self.best_metrics['best_val_acc']:.4f}")
        print(f"   Best validation AUC: {self.best_metrics['best_val_auc']:.4f}")
        
        if test_metrics:
            print(f"   Test accuracy: {test_metrics.get('acc', 0.0):.4f}")
            print(f"   Test AUC: {test_metrics.get('auc', 0.0):.4f}")
        
        return summary

class EarlyStopping:
    """
    Early stopping mechanism

    Supports early stopping based on any metric (score), can be loss, AUC, accuracy, etc.
    Specify whether to maximize or minimize the metric via mode parameter
    """
    
    def __init__(self,
                 patience: int = 20,
                 stop_epoch: int = 50,
                 verbose: bool = False,
                 mode: str = 'max',
                 min_delta: float = 0.0):
        """
        Initialize early stopping mechanism

        Args:
            patience: How many epochs to tolerate without improvement
            stop_epoch: Earliest epoch after which early stopping is allowed
            verbose: Whether to print detailed information
            mode: 'max' means maximize metric (like AUC, accuracy), 'min' means minimize metric (like loss)
            min_delta: Minimum threshold for improvement, only considered improvement if exceeding this threshold
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Set initial best score based on mode
        if self.mode == 'max':
            self.best_score = -np.Inf
        elif self.mode == 'min':
            self.best_score = np.Inf
        else:
            raise ValueError(f"mode must be 'max' or 'min', current: {mode}")

    def __call__(self, epoch: int, score: float, model: nn.Module, ckpt_name: str = 'checkpoint.pt') -> bool:
        """
        Check if early stopping should be triggered

        Args:
            epoch: Current epoch number
            score: Current metric value (can be loss, AUC, accuracy, etc.)
            model: Model object
            ckpt_name: Checkpoint save path

        Returns:
            Whether early stopping should be triggered
        """
        # Determine if improved
        if self.mode == 'max':
            # Maximization mode: higher score is better
            is_better = score > (self.best_score + self.min_delta)
        else:
            # Minimization mode: lower score is better
            is_better = score < (self.best_score - self.min_delta)
        
        if is_better:
            # If improved, update best score and save model
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0
        else:
            # If not improved, increase counter
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            # Check if early stopping should be triggered
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        
        return self.early_stop

    def save_checkpoint(self, score: float, model: nn.Module, ckpt_name: str):
        """
        Save model checkpoint
        
        Args:
            score: Current metric value
            model: Model object
            ckpt_name: Checkpoint save path
        """
        if self.verbose:
            mode_str = 'increased' if self.mode == 'max' else 'decreased'
            print(f'Validation score {mode_str} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)

class Trainer:
    """
    Universal Trainer Class
    Supports different model types and training configurations
    """
    
    def __init__(self, 
                 configs: Dict,
                 log_dir: str = None):
        """
        Initialize trainer

        Args:
            configs: Configuration dictionary
            log_dir: Log save directory
        """
        self.model_config = configs['model_config']
        self.experiment_config = configs['experiment_config']
        self.results_dir = self.experiment_config['results_dir']
        self.log_dir = log_dir or './logs'
        
        # Validate configuration completeness
        required_training_params = ['max_epochs', 'lr', 'reg', 'opt', 'early_stopping', 'batch_size']
        missing_training_params = [param for param in required_training_params if param not in self.experiment_config]
        if missing_training_params:
            raise ValueError(f"Training configuration missing required parameters: {missing_training_params}")
        
        # Extract parameters from config
        self.max_epochs = self.experiment_config['max_epochs']
        self.lr = self.experiment_config['lr']
        self.reg = self.experiment_config['reg']
        self.opt = self.experiment_config['opt']
        self.early_stopping = self.experiment_config['early_stopping']
        self.batch_size = self.experiment_config['batch_size']
        
        # Initialize model and loss function
        self.model = None
        self.loss_fn = None
        self.scheduler = None

    def _init_model(self) -> nn.Module:
        """Initialize model"""
        # Get parameters from model_config and build configuration
        config = self.model_config.copy()
        
        # Create model using model factory
        model = ModelFactory.create_model(config)
        
        return model.to(device)
    
    def train_fold(self, 
                   datasets: Tuple[Any, Any, Any],
                   fold_idx: int) -> Tuple[Dict, float, float, float, float]:
        """
        Level 1: Fold training main entry point

        Args:
            datasets: (train_dataset, val_dataset, test_dataset)
            fold_idx: fold index

        Returns:
            (results_dict, test_auc, val_auc, test_acc, val_acc)
        """
        print(f'\nTraining Fold {fold_idx}!')
        
        # Create directories and logger
        metrics_logger = Logger(self.model_config['n_classes'], self.log_dir, fold_idx)

        # Save dataset splits
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'],
                   os.path.join(self.results_dir, 'splits_{}.csv'.format(fold_idx)))

        # Calculate and validate split ratios
        total_samples = len(train_split) + len(val_split) + len(test_split)
        train_ratio = len(train_split) / total_samples
        val_ratio = len(val_split) / total_samples
        test_ratio = len(test_split) / total_samples

        print(f"Training on {len(train_split)} samples ({train_ratio:.1%})")
        print(f"Validating on {len(val_split)} samples ({val_ratio:.1%})")
        print(f"Testing on {len(test_split)} samples ({test_ratio:.1%})")

        # Validate ratios are reasonable (train should be largest, val and test should be equal)
        if val_ratio != test_ratio:
            print(f"⚠️ Warning: Validation set ({val_ratio:.1%}) and test set ({test_ratio:.1%}) ratios are inconsistent")
        if train_ratio < 0.5:
            print(f"⚠️ Warning: Training set ratio ({train_ratio:.1%}) is too low, may affect model training")

        # Initialize model and loss function
        model = self._init_model()
        self.loss_fn = model.loss_fn
        print_network(model)
        optimizer = get_optim(model, self.opt, self.lr, self.reg)
        
        # Initialize learning rate scheduler
        scheduler_config = self.experiment_config.get('scheduler_config', {})
        self.scheduler = get_scheduler(optimizer, scheduler_config)
        if self.scheduler:
            print(f"🎯 Using learning rate scheduler: {scheduler_config.get('type', 'unknown')}")
        
        # Initialize data loaders
        seed = self.experiment_config['seed']
        train_loader = get_split_loader(train_split, training=True, weighted=True, batch_size=1, generator=torch.Generator().manual_seed(seed))
        val_loader = get_split_loader(val_split, training=False, weighted=False, batch_size=1, generator=torch.Generator().manual_seed(seed))
        test_loader = get_split_loader(test_split, training=False, weighted=False, batch_size=1, generator=torch.Generator().manual_seed(seed))

        # Initialize early stopping
        # Get early stopping parameters from config, support custom metrics and modes
        early_stopping_config = self.experiment_config.get('early_stopping_config', {})
        
        if self.early_stopping:
            # If early_stopping is a dict, use config from dict; otherwise use default config
            if isinstance(self.early_stopping, dict):
                config = {**early_stopping_config, **self.early_stopping}
            else:
                config = early_stopping_config
            
            # Get config parameters, use defaults
            patience = config.get('patience', 25)
            stop_epoch = config.get('stop_epoch', 10)
            verbose = config.get('verbose', True)
            mode = config.get('mode', 'max')  # 'max' for auc/acc, 'min' for loss
            min_delta = config.get('min_delta', 0.0)
            metric = config.get('metric', 'auc')  # 'auc', 'acc', 'loss'
            
            early_stopping_obj = EarlyStopping(
                patience=patience, 
                stop_epoch=stop_epoch, 
                verbose=verbose,
                mode=mode,
                min_delta=min_delta
            )
            # Save metric configuration for subsequent metric selection
            early_stopping_obj.metric = metric
        else:
            early_stopping_obj = None
        
        # 2. Training
        for epoch in range(self.max_epochs):
            # Train and validate
            train_metrics = self._train_single_epoch(epoch, train_loader, optimizer, model, metrics_logger)
            val_metrics, stop = self._validate_single_epoch(fold_idx, epoch, val_loader, model, early_stopping_obj)
            
            # Log metrics
            metrics_logger.log_epoch(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau requires validation loss
                    self.scheduler.step(val_metrics['loss'])
                else:
                    # Other schedulers use epoch
                    self.scheduler.step()
            
            if stop: 
                break
        
        # 3. Final evaluation and return results
        # Save model
        checkpoint_path = os.path.join(self.results_dir, "s_{}_checkpoint.pt".format(fold_idx))
        if self.early_stopping:
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            torch.save(model.state_dict(), checkpoint_path)

        # Final evaluation
        _, val_accuracy, val_auc, _ = self._evaluate_model(val_loader, model)
        results_dict, test_accuracy, test_auc, eval_logger = self._evaluate_model(test_loader, model)
        
        print('Val accuracy: {:.4f}, ROC AUC: {:.4f}'.format(val_accuracy, val_auc))
        print('Test accuracy: {:.4f}, ROC AUC: {:.4f}'.format(test_accuracy, test_auc))

        # Print accuracy for each class
        for i in range(self.model_config['n_classes']):
            acc, correct, count = eval_logger.get_class_accuracy(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        # Save training summary
        metrics_logger.save_summary({
            'acc': test_accuracy,
            'auc': test_auc,
            'loss': 1-test_accuracy
        })
            
        return results_dict, test_auc, val_auc, test_accuracy, val_accuracy

    def _train_single_epoch(self, epoch: int, loader: DataLoader, optimizer: torch.optim.Optimizer, model: nn.Module, logger: Logger) -> Dict:
        """
        Level 3: Standard model single epoch training
        """
        model.train()
        
        # 🔧 Reset epoch statistics to ensure independent statistics for each epoch
        logger.reset_epoch_stats()

        print('\n')
        batch_size = self.experiment_config['batch_size']
        total_loss = 0
        for batch_idx, (data, label) in enumerate(loader):
            # Labels are already tensors, move directly to device
            label = label.to(device)
            
            # data is now in dictionary format, each channel contains a tensor
            # Need to move each channel's tensor to device
            for channel in data:
                data[channel] = data[channel].to(device)
            results = model(data, label)
            Y_prob = results['probabilities']
            Y_hat = results['predictions']
            
            # Calculate loss
            results['labels'] = label
            loss = self.loss_fn(results['logits'], results['labels'], results)
            total_loss += loss
            # Log metrics
            logger.log_batch(Y_hat, label, Y_prob, loss)
            
            if (batch_idx + 1) % batch_size == 0:
                # Backpropagation
                if hasattr(model, 'group_loss_fn'):
                    results['group_loss'] = model.group_loss_fn(results)
                    total_loss += results['group_loss']
                total_loss = total_loss/batch_size
                results['total_loss'] = total_loss.item()
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if hasattr(model, 'verbose_items'):
                    items = model.verbose_items(results)
                    if len(items) > 0:
                        print('Batch {}/{}: '.format(batch_idx + 1, len(loader)) + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
                total_loss = 0
        
        if len(loader) % batch_size != 0:
            # Calculate the number of remaining batches
            remaining_batches = len(loader) % batch_size
            # Backpropagation
            if hasattr(model, 'group_loss_fn'):
                results['group_loss'] = model.group_loss_fn(results)
                total_loss += results['group_loss']
            total_loss = total_loss / remaining_batches  # Average using remaining batch count
            results['total_loss'] = total_loss.item()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if hasattr(model, 'verbose_items'):
                items = model.verbose_items(results)
                if len(items) > 0:
                    print('Final batch: ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
            total_loss = 0
        # Calculate average metrics
        train_loss = logger.batch_log['loss'] / len(loader)

        print('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}'.format(epoch, train_loss, logger.get_overall_accuracy()))
        if hasattr(model, 'verbose_items'):
            results['is_epoch'] = True
            items = model.verbose_items(results)
            if len(items) > 0:
                print('- ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
        
        # Calculate and return metrics
        return self._calculate_epoch_metrics(logger)

    def _calculate_epoch_metrics(self, logger: Dict) -> Dict:
        """Calculate epoch metrics"""
        n_classes = self.model_config['n_classes']
        
        # Calculate accuracy
        train_acc = 0.0
        for i in range(n_classes):
            acc, correct, count = logger.get_class_accuracy(i)
            if acc is not None:
                train_acc += acc
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        labels = torch.cat(logger.batch_log['labels'], dim=0)
        probs = torch.cat(logger.batch_log['probs'], dim=0) # [N, C]
        train_acc /= n_classes
        train_loss = logger.batch_log['loss'] / len(labels)

        # Calculate AUC - using torchmetrics (native Tensor/GPU support)
        if n_classes == 2:
            auroc = TM_AUROC(task='binary').to(probs.device)
            train_auc = float(auroc(probs[:, 1], labels.long()).item())
        else:
            auroc = TM_AUROC(task='multiclass', num_classes=n_classes, average='macro').to(probs.device)
            train_auc = float(auroc(probs, labels.long()).item())
        
        metrics = {
            'loss': train_loss,
            'acc': train_acc,
            'auc': train_auc
        }
        return metrics

    def _validate_single_epoch(self, cur: int, epoch: int, loader: DataLoader, model: nn.Module, early_stopping=None) -> Tuple[Dict, bool]:
        """Validation function"""
        model.eval()
        n_classes = self.model_config['n_classes']
        logger = Logger(n_classes=n_classes)
        
        # Reset model's group_logits and group_labels to ensure validation starts from clean state
        if hasattr(model, 'group_logits'):
            model.group_logits = []
        if hasattr(model, 'group_labels'):
            model.group_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                label = label.to(device)
                
                # data is now in dictionary format, each channel contains a tensor
                # Need to move each channel's tensor to device
                for channel in data:
                    data[channel] = data[channel].to(device)

                results = model(data, label)
                Y_prob = results['probabilities']
                Y_hat = results['predictions']

                results['labels'] = label
                loss = self.loss_fn(results['logits'], results['labels'], results)
                logger.log_batch(Y_hat, label, Y_prob, loss)
        
        # Calculate AUC loss at validation end
        if hasattr(model, 'group_loss_fn') and hasattr(model, 'group_logits') and model.group_logits:
            results['group_loss'] = model.group_loss_fn(results)
            logger.batch_log['loss'] += results['group_loss']
            
        val_loss = logger.batch_log['loss']/len(loader)
        val_acc = logger.get_overall_accuracy()
        labels = torch.cat(logger.batch_log['labels'], dim=0)
        prob = torch.cat(logger.batch_log['probs'], dim=0)
        
        if n_classes == 2:
            auroc = TM_AUROC(task='binary').to(prob.device)
            auc = float(auroc(prob[:, 1], labels.long()).item())
        else:
            auroc = TM_AUROC(task='multiclass', num_classes=n_classes, average='macro').to(prob.device)
            auc = float(auroc(prob, labels.long()).item())

        print('\nVal Set, val_loss: {:.4f}, val_accuracy: {:.4f}, auc: {:.4f}'.format(val_loss, val_acc, auc))
        
        if hasattr(model, 'verbose_items'):
            results['is_epoch'] = True
            results['total_loss'] = val_loss
            items = model.verbose_items(results)
            if len(items) > 0:
                print('- ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))

        for i in range(n_classes):
            acc, correct, count = logger.get_class_accuracy(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        val_metrics = {
            'loss': val_loss,
            'acc': val_acc,
            'auc': auc
        }

        if early_stopping:
            assert self.results_dir
            # Select metric to use based on configured metric
            metric_name = getattr(early_stopping, 'metric', 'auc')
            if metric_name == 'loss':
                score = val_loss
            elif metric_name == 'auc':
                score = auc
            elif metric_name == 'acc' or metric_name == 'accuracy':
                score = val_acc
            else:
                # Default to auc
                score = auc
                print(f"⚠️ Warning: Unknown early stopping metric '{metric_name}', using default 'auc'")
            
            early_stopping(epoch, score, model, 
                         ckpt_name=os.path.join(self.results_dir, "s_{}_checkpoint.pt".format(cur)))
            
            if early_stopping.early_stop:
                print("Early stopping")
                return val_metrics, True

        return val_metrics, False

    def _evaluate_model(self, loader: DataLoader, model: nn.Module, drop_prob: Optional[float] = None) -> Tuple[Dict, float, float, Logger]:
        """
        Model evaluation summary
        
        Args:
            loader: Data loader
            model: Model
            drop_prob: Modality dropout probability (0.0-1.0), passed to model during forward
        """
        model.eval()
        logger = Logger(n_classes=self.model_config['n_classes'])

        # Reset model's group_logits and group_labels to ensure testing starts from clean state
        if hasattr(model, 'group_logits'):
            model.group_logits = []
        if hasattr(model, 'group_labels'):
            model.group_labels = []

        dataset_ref = loader.dataset
        case_ids_list: List[str]
        if hasattr(dataset_ref, 'case_ids'): # Direct dataset (has case_ids attribute)
            base = dataset_ref.case_ids
            case_ids_list = list(base) if not isinstance(base, list) else base
        elif hasattr(dataset_ref, 'dataset'):
            case_ids_list = dataset_ref.dataset.case_ids
        else:
            raise ValueError(f"Expected dataset with case_ids attribute, got {type(dataset_ref)}")  
        patient_results = {}
 
        for batch_idx, (data, label) in enumerate(loader):
            label = label.to(device)
            for channel in data:
                data[channel] = data[channel].to(device)
            case_id = case_ids_list[batch_idx]
            with torch.inference_mode():
                # Pass drop_prob parameter
                if drop_prob is not None:
                    results = model(data, label, drop_prob=drop_prob)
                else:
                    results = model(data, label)
                Y_prob = results['probabilities']
                Y_hat = results['predictions']
            
            results['labels'] = label
            loss = self.loss_fn(results['logits'], results['labels'], results)
            logger.log_batch(Y_hat, label, Y_prob, loss)
            
            patient_results.update({case_id: {'case_id': np.array(case_id), 'prob': Y_prob.cpu().numpy(), 'label': label.item()}})
        
        # Calculate AUC loss at test end
        if hasattr(model, 'group_loss_fn') and hasattr(model, 'group_logits') and model.group_logits:
            results['group_loss'] = model.group_loss_fn(results)
            logger.batch_log['loss'] += results['group_loss']
        
        test_loss = logger.batch_log['loss']/len(loader)
        test_acc = logger.get_overall_accuracy()
        
        if hasattr(model, 'verbose_items'):
            results['is_epoch'] = True
            results['total_loss'] = test_loss
            items = model.verbose_items(results)
            if len(items) > 0:
                print('- ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
        
        labels = torch.cat(logger.batch_log['labels'], dim=0)
        prob = torch.cat(logger.batch_log['probs'], dim=0)

        if self.model_config['n_classes'] == 2:
            auroc = TM_AUROC(task='binary').to(prob.device)
            auc = float(auroc(prob[:, 1], labels.long()).item())
        else:
            auroc = TM_AUROC(task='multiclass', num_classes=self.model_config['n_classes'], average='macro').to(prob.device)
            auc = float(auroc(prob, labels.long()).item())
        
        print('\nTest Set, test_loss: {:.4f}, test_accuracy: {:.4f}, auc: {:.4f}'.format(test_loss, test_acc, auc))

        return patient_results, test_acc, auc, logger

    def evaluate_fold(self,
                      datasets: Tuple[Any, Any, Any],
                      fold_idx: int,
                      checkpoint_path: str,
                      drop_prob: Optional[float] = None) -> Tuple[Dict, float, Optional[float], float, Optional[float]]:
        """
        Evaluation-only interface: Load specified checkpoint, evaluate on test set of given datasets.

        Args:
            datasets: (train_dataset, val_dataset, test_dataset) tuple, test set will be used for evaluation
            fold_idx: Current fold index (for logging/compatibility interface)
            checkpoint_path: Model weights path (recommended: s_{fold}_checkpoint.pt saved by train_fold)
            drop_prob: Modality dropout probability (0.0-1.0), passed to model during forward

        Returns:
            (results_dict, test_auc, None, test_acc, None) aligned with train_fold results format (validation metrics set to None)
        """
        print(f"\n[Evaluate] Fold {fold_idx} | checkpoint: {checkpoint_path}")

        # Re-initialize model for each evaluation (don't reuse previous model state)
        model = self._init_model()
        print(f"🔧 Creating new model instance, id={id(model)}")
        self.loss_fn = model.loss_fn  # Update loss_fn to current model's
        
        # Load checkpoint (consistent with training load method)
        state = torch.load(checkpoint_path, map_location=device)
        print(f"📦 Checkpoint loaded successfully, state_dict keys count: {len(state.keys())}")
        
        if hasattr(model, 'transfer_layer') and hasattr(model, 'create_transfer_layer'):
            # Find all transfer_layer channels from checkpoint
            transfer_layer_channels = {}
            for key in state.keys():
                if 'transfer_layer.' in key:
                    # Extract channel name and weight type, e.g., "transfer_layer.clinical=val.weight" -> ("clinical=val", "weight")
                    parts = key.split('.')
                    if len(parts) >= 3:
                        channel_name = parts[1]  # e.g., "clinical=val"
                        weight_type = parts[2]  # "weight" 或 "bias"
                        
                        if channel_name not in transfer_layer_channels:
                            transfer_layer_channels[channel_name] = {}
                        transfer_layer_channels[channel_name][weight_type] = state[key]
            
            # Create corresponding transfer_layer based on weights in checkpoint
            if hasattr(model, 'output_dim'):
                output_dim = model.output_dim
                print(f"🔧 Pre-creating {len(transfer_layer_channels)} transfer_layers to match checkpoint...")
                for channel_name, weights in transfer_layer_channels.items():
                    if channel_name not in model.transfer_layer:
                        # Infer input_dim from weight shape: weight shape is [output_dim, input_dim]
                        if 'weight' in weights:
                            weight_tensor = weights['weight']
                            if len(weight_tensor.shape) == 2:
                                input_dim = weight_tensor.shape[1]  # Second dimension is input_dim
                                # Create transfer_layer
                                transfer_layer = model.create_transfer_layer(input_dim)
                                model.transfer_layer[channel_name] = transfer_layer
                                print(f"   ✅ Created transfer_layer.{channel_name} (input_dim={input_dim}, output_dim={output_dim})")
                            else:
                                print(f"   ⚠️ Cannot infer input_dim for {channel_name}: abnormal weight shape {weight_tensor.shape}")
                        else:
                            print(f"   ⚠️ Missing {channel_name}.weight in checkpoint, cannot create transfer_layer")
        
        # Analyze weight types in checkpoint
        transfer_layer_keys = [k for k in state.keys() if 'transfer_layer.' in k]
        core_keys = [k for k in state.keys() if 'transfer_layer.' not in k]
        
        print(f"📊 Checkpoint weights analysis:")
        print(f"   Core weights: {len(core_keys)} items")
        print(f"   Transfer_layer weights: {len(transfer_layer_keys)} items")
        
        # Now all required transfer_layers are created, try strict=True to ensure perfect match
        # If there are still mismatches, fall back to strict=False
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
            print(f"✅ Successfully loaded all weights using strict=True (perfect match)")
        except RuntimeError as e:
            # If strict=True fails, use strict=False but report in detail
            print(f"⚠️ strict=True loading failed: {e}")
            print(f"🔧 Falling back to strict=False loading...")
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        
        # Check if all core weights are loaded
        model_core_keys = set([k for k in model.state_dict().keys() if 'transfer_layer.' not in k])
        checkpoint_core_keys = set(core_keys)
        loaded_core_keys = model_core_keys & checkpoint_core_keys
        
        if missing_keys:
            missing_core = [k for k in missing_keys if 'transfer_layer.' not in k]
            missing_transfer = [k for k in missing_keys if 'transfer_layer.' in k]
            if missing_core:
                print(f"⚠️ Warning: Missing core weights (may cause performance degradation): {len(missing_core)} items")
                for key in missing_core[:5]:
                    print(f"    - {key}")
                if len(missing_core) > 5:
                    print(f"    ... 还有 {len(missing_core) - 5} 个")
            if missing_transfer:
                print(f"ℹ️ Info: Missing transfer_layer weights (will be created dynamically during forward): {len(missing_transfer)} items")
        
        if unexpected_keys:
            unexpected_transfer = [k for k in unexpected_keys if 'transfer_layer.' in k]
            unexpected_other = [k for k in unexpected_keys if 'transfer_layer.' not in k]
            if unexpected_transfer:
                print(f"ℹ️ Info: Extra transfer_layer weights in checkpoint (ignored, does not affect evaluation): {len(unexpected_transfer)} items")
            if unexpected_other:
                print(f"⚠️ Warning: Unexpected other weights in checkpoint: {len(unexpected_other)} items")
                for key in unexpected_other[:5]:
                    print(f"    - {key}")
        
        # Validate core weights loading status
        print(f"✅ Core weights loaded: {len(loaded_core_keys)}/{len(checkpoint_core_keys)} items")
        if len(loaded_core_keys) == len(checkpoint_core_keys):
            print(f"✅ All core weights successfully loaded")
        else:
            print(f"⚠️ Warning: Some core weights not loaded: {len(checkpoint_core_keys) - len(loaded_core_keys)} items")
        
        # Set to evaluation mode
        model.eval()

        # Only construct test set data loader
        _, _, test_split = datasets
        test_loader = get_split_loader(test_split, training=False, weighted=False, batch_size=1)

        # Evaluate (pass drop_prob)
        results_dict, test_acc, test_auc, _ = self._evaluate_model(test_loader, model, drop_prob=drop_prob)
        return results_dict, float(test_auc), None, float(test_acc), None

    def evaluate_with_checkpoint(self,
                                 datasets: Tuple[Any, Any, Any],
                                 fold_idx: int,
                                 checkpoint_path: str,
                                 drop_prob: Optional[float] = None) -> Tuple[Dict, float, Optional[float], float, Optional[float]]:
        """
        Compatible name: directly calls evaluate_fold.
        
        Args:
            datasets: Dataset tuple
            fold_idx: Fold index
            checkpoint_path: Checkpoint path
            drop_prob: Modality dropout probability (0.0-1.0), passed to model during forward
        """
        return self.evaluate_fold(datasets=datasets, fold_idx=fold_idx, checkpoint_path=checkpoint_path, drop_prob=drop_prob)
