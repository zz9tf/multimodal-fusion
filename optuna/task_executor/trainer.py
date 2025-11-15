"""
通用训练器类
支持不同的模型类型和训练配置
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

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 只导入必要的模块，避免依赖 utils.utils
from models.model_factory import ModelFactory
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import AUROC as TM_AUROC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_serializable(obj: Any) -> Any:
    """通用JSON安全序列化转换器

    - 将 numpy 标量转换为 Python 标量
    - 将 numpy 数组转换为列表
    - 将 torch 张量移动到 CPU 并转换为列表
    - 其他不可序列化对象转换为字符串
    """
    # numpy 标量
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if obj is np.nan:
        return None
    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # torch 张量
    if torch.is_tensor(obj):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            return str(obj)
    # 其他常见不可序列化类型兜底为字符串
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	"""
	保存数据集分割信息（使用patient_id/case_id而非索引，确保可复现性）
	
	关键修复：从Subset对象中提取原始数据集的case_ids，使用实际的case_id而非索引
	这样即使数据集顺序不同，也能通过case_id正确匹配划分
	"""
	try:
		# 获取每个分割的case_ids（从Subset中提取原始数据集的case_ids）
		splits = []
		for i, dataset in enumerate(split_datasets):
			if hasattr(dataset, 'case_ids'):
				# 直接是MultimodalDataset对象
				splits.append(pd.Series(dataset.case_ids))
			elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
				# 是Subset对象，需要从原始数据集提取case_ids
				base_dataset = dataset.dataset
				indices = dataset.indices
				
				if hasattr(base_dataset, 'case_ids'):
					# 从原始数据集的case_ids中提取对应的case_id
					base_case_ids = base_dataset.case_ids
					if isinstance(base_case_ids, list):
						case_ids = [base_case_ids[idx] for idx in indices]
					else:
						# 如果是其他类型（如numpy array），转换为list
						case_ids = [base_case_ids[idx] for idx in indices]
					splits.append(pd.Series(case_ids))
				else:
					# fallback: 使用索引
					splits.append(pd.Series([f"sample_{j}" for j in indices]))
			else:
				# fallback: 使用索引
				splits.append(pd.Series([f"sample_{j}" for j in range(len(dataset))]))
		
		if not boolean_style:
			# 创建DataFrame，每列是一个分割的case_ids
			# 使用最长的分割作为DataFrame的长度，较短的用NaN填充
			max_len = max(len(s) for s in splits) if splits else 0
			
			# 创建字典，每个键对应一个分割的case_ids
			data_dict = {}
			for i, col_key in enumerate(column_keys):
				if i < len(splits):
					case_ids = splits[i].tolist()
					# 填充NaN使其长度一致
					while len(case_ids) < max_len:
						case_ids.append(None)
					data_dict[col_key] = case_ids
				else:
					data_dict[col_key] = [None] * max_len
			
			df = pd.DataFrame(data_dict)
			# 移除全NaN的行
			df = df.dropna(how='all')
		else:
			df = pd.concat(splits, ignore_index = True, axis=0)
			index = df.values.tolist()
			one_hot = np.eye(len(split_datasets)).astype(bool)
			bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
			df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

		df.to_csv(filename, index=False)
		print(f"✅ 保存分割信息到: {filename} (使用case_id)")
	except Exception as e:
		print(f"⚠️ 保存分割信息失败: {e}")
		import traceback
		traceback.print_exc()
		# 创建一个简单的分割记录
		split_info = {
			'split_type': column_keys,
			'train_size': len(split_datasets[0]) if len(split_datasets) > 0 else 0,
			'val_size': len(split_datasets[1]) if len(split_datasets) > 1 else 0,
			'test_size': len(split_datasets[2]) if len(split_datasets) > 2 else 0
		}
		pd.DataFrame([split_info]).to_csv(filename, index=False)
		print(f"✅ 保存简化分割信息到: {filename}")

def print_network(model: nn.Module):
    """打印网络结构和参数统计"""
    print("=" * 50)
    print("Model Architecture:")
    print("=" * 50)
    print(model)
    print("=" * 50)
    
    # 计算参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 50)

def get_optim(model: nn.Module, opt: str, lr: float, reg: float) -> torch.optim.Optimizer:
    """获取优化器"""
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
    获取学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_config: 调度器配置字典
        
    Returns:
        学习率调度器或None
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
        T_0 = scheduler_config.get('T_0', 10)  # 第一个重启周期长度
        T_mult = scheduler_config.get('T_mult', 2)  # 周期长度倍增因子
        eta_min = scheduler_config.get('eta_min', 0.0)  # 最小学习率
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
        print(f"⚠️ 未知的调度器类型: {scheduler_type}")
        return None

def get_split_loader(dataset, training=False, weighted=False, batch_size=1, generator=None):
    """获取数据加载器
    
    Args:
        dataset: 数据集
        training: 是否为训练模式
        weighted: 是否使用加权采样
        batch_size: batch大小
        generator: 随机数生成器（用于确保采样顺序一致）
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
    """为平衡类别创建权重"""
    N = float(len(dataset))
    
    # 获取标签，适配MultimodalDataset格式
    labels = []
    unique_labels = set()
    
    for i in range(len(dataset)):
        # 处理Subset对象
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'get_label'):
            # 通过Subset的indices获取原始数据集的索引
            original_idx = dataset.indices[i]
            label = dataset.dataset.get_label(original_idx)
        else:
            # 直接处理MultimodalDataset
            label = dataset.get_label(i)
        
        unique_labels.add(label)
        labels.append(label)
    
    # 使用数据集的标签映射
    if hasattr(dataset, 'label_to_int'):
        label_to_int = dataset.label_to_int
    else:
        # 如果没有标签映射，创建默认映射
        label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    
    # 将字符串标签转换为数字
    numeric_labels = [label_to_int[label] for label in labels]
    labels = np.array(numeric_labels)
    
    class_counts = np.bincount(labels)
    class_weights = N / class_counts
    weights = [class_weights[labels[i]] for i in range(len(dataset))]
    return torch.DoubleTensor(weights)


class Logger:
    """
    统一的训练指标记录器
    整合了准确率统计、训练日志记录和最佳指标跟踪功能
    """
    
    def __init__(self, n_classes: int, log_dir: str = None, fold: int = 0):
        """
        初始化指标记录器
        
        Args:
            n_classes: 类别数量
            log_dir: 日志保存目录（可选）
            fold: fold索引
        """
        self.n_classes = n_classes
        self.log_dir = log_dir
        self.fold = fold
        
        # 类别统计
        self.batch_log = {
            'class_stats': [{"count": 0, "correct": 0} for _ in range(self.n_classes)],
            'labels': [],
            'probs': [],
            'loss': 0.0
        }
        
        # 训练日志
        self.epoch_logs = []
        self.best_metrics = {
            'best_val_loss': float('inf'),
            'best_val_acc': 0.0,
            'best_val_auc': 0.0,
            'best_epoch': 0
        }
        
        # 初始化文件记录（如果提供了log_dir）
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
        """重置当前epoch的统计"""
        self.batch_log = {
            'class_stats': [{"count": 0, "correct": 0} for _ in range(self.n_classes)],
            'labels': [],
            'probs': [],
            'loss': 0.0
        }
    
    def log_batch(self, Y_hat, Y, Y_prob, loss):
        """
        记录批次预测结果，loss，labels，probs
        
        Args:
            Y_hat: 预测结果 (int, tensor, 或 array)
            Y: 真实标签 (int, tensor, 或 array)
            Y_prob: 预测概率 (tensor, 或 array)
            loss: 损失值 (tensor)
        """
        # 统一转为Tensor（不转numpy），用于后续torch.cat
        if not torch.is_tensor(Y_hat):
            Y_hat = torch.as_tensor(Y_hat)
        if not torch.is_tensor(Y):
            Y = torch.as_tensor(Y)
        if not torch.is_tensor(Y_prob):
            Y_prob = torch.as_tensor(Y_prob)

        # 统计分类正确数
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

        # 追加到日志（保持为Tensor）
        self.batch_log['labels'].append(Y)
        self.batch_log['probs'].append(Y_prob)
        self.batch_log['loss'] += float(loss.item())
    
    def get_class_accuracy(self, class_idx: int) -> Tuple[Optional[float], int, int]:
        """
        获取指定类别的准确率
        
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
        """获取整体准确率"""
        total_correct = sum(stat["correct"] for stat in self.batch_log['class_stats'])
        total_count = sum(stat["count"] for stat in self.batch_log['class_stats'])
        
        if total_count == 0:
            return 0.0
        return float(total_correct) / total_count
    
    def log_epoch(self, epoch: int, train_metrics: Dict, val_metrics: Dict, lr: float):
        """
        记录epoch指标
        
        Args:
            epoch: 当前epoch
            train_metrics: 训练指标字典
            val_metrics: 验证指标字典  
            lr: 学习率
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
        
        # 写入CSV文件（如果启用了文件记录）
        if self.log_dir:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch_log['epoch'], epoch_log['train_loss'], epoch_log['train_acc'], 
                    epoch_log['train_auc'], epoch_log['val_loss'], epoch_log['val_acc'], 
                    epoch_log['val_auc'], epoch_log['learning_rate'], epoch_log['timestamp']
                ])
        
        # 更新最佳指标
        self._update_best_metrics(epoch, val_metrics)
        
        # 打印进度
        print(f"📊 Epoch {epoch:3d} | "
              f"Train: Loss={train_metrics.get('loss', 0.0):.4f}, "
              f"Acc={train_metrics.get('acc', 0.0):.4f}, "
              f"AUC={train_metrics.get('auc', 0.0):.4f} | "
              f"Val: Loss={val_metrics.get('loss', 0.0):.4f}, "
              f"Acc={val_metrics.get('acc', 0.0):.4f}, "
              f"AUC={val_metrics.get('auc', 0.0):.4f}")
    
    def _update_best_metrics(self, epoch: int, val_metrics: Dict):
        """更新最佳指标"""
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
        """保存训练总结"""
        summary = {
            'fold': self.fold,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.epoch_logs),
            'final_epoch': self.epoch_logs[-1] if self.epoch_logs else None,
            'test_metrics': test_metrics
        }
        
        if self.log_dir:
            # 保存JSON总结
            summary_path = os.path.join(self.log_dir, f'fold_{self.fold}_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=to_serializable)
        
        # 打印总结
        print(f"\n🎯 Fold {self.fold} 训练总结:")
        print(f"   最佳验证损失: {self.best_metrics['best_val_loss']:.4f} (Epoch {self.best_metrics['best_epoch']})")
        print(f"   最佳验证准确率: {self.best_metrics['best_val_acc']:.4f}")
        print(f"   最佳验证AUC: {self.best_metrics['best_val_auc']:.4f}")
        
        if test_metrics:
            print(f"   测试准确率: {test_metrics.get('acc', 0.0):.4f}")
            print(f"   测试AUC: {test_metrics.get('auc', 0.0):.4f}")
        
        return summary

class EarlyStopping:
    """
    早停机制
    
    支持根据任意指标（score）进行早停，可以是 loss、AUC、accuracy 等
    通过 mode 参数指定是最大化还是最小化指标
    """
    
    def __init__(self, 
                 patience: int = 20, 
                 stop_epoch: int = 50, 
                 verbose: bool = False,
                 mode: str = 'max',
                 min_delta: float = 0.0):
        """
        初始化早停机制
        
        Args:
            patience: 容忍多少个 epoch 没有改善
            stop_epoch: 最早在第几个 epoch 之后才允许早停
            verbose: 是否打印详细信息
            mode: 'max' 表示最大化指标（如 AUC、accuracy），'min' 表示最小化指标（如 loss）
            min_delta: 改善的最小阈值，只有超过这个阈值才认为是改善
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # 根据模式设置初始最佳值
        if self.mode == 'max':
            self.best_score = -np.Inf
        elif self.mode == 'min':
            self.best_score = np.Inf
        else:
            raise ValueError(f"mode 必须是 'max' 或 'min'，当前为: {mode}")

    def __call__(self, epoch: int, score: float, model: nn.Module, ckpt_name: str = 'checkpoint.pt') -> bool:
        """
        检查是否应该早停
        
        Args:
            epoch: 当前 epoch 编号
            score: 当前指标值（可以是 loss、AUC、accuracy 等）
            model: 模型对象
            ckpt_name: 检查点保存路径
            
        Returns:
            是否应该早停
        """
        # 判断是否改善
        if self.mode == 'max':
            # 最大化模式：score 越大越好
            is_better = score > (self.best_score + self.min_delta)
        else:
            # 最小化模式：score 越小越好
            is_better = score < (self.best_score - self.min_delta)
        
        if is_better:
            # 有改善，更新最佳值并保存模型
            self.best_score = score
            self.save_checkpoint(score, model, ckpt_name)
            self.counter = 0
        else:
            # 没有改善，增加计数器
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            
            # 检查是否应该早停
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        
        return self.early_stop

    def save_checkpoint(self, score: float, model: nn.Module, ckpt_name: str):
        """
        保存模型检查点
        
        Args:
            score: 当前指标值
            model: 模型对象
            ckpt_name: 检查点保存路径
        """
        if self.verbose:
            mode_str = 'increased' if self.mode == 'max' else 'decreased'
            print(f'Validation score {mode_str} ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)

class Trainer:
    """
    通用训练器类
    支持不同的模型类型和训练配置
    """
    
    def __init__(self, 
                 configs: Dict,
                 log_dir: str = None):
        """
        初始化训练器
        
        Args:
            configs: 配置字典
            log_dir: 日志保存目录
        """
        self.model_config = configs['model_config']
        self.experiment_config = configs['experiment_config']
        self.results_dir = self.experiment_config['results_dir']
        self.log_dir = log_dir or './logs'
        
        # 验证配置完整性
        required_training_params = ['max_epochs', 'lr', 'reg', 'opt', 'early_stopping', 'batch_size']
        missing_training_params = [param for param in required_training_params if param not in self.experiment_config]
        if missing_training_params:
            raise ValueError(f"训练配置缺少必需参数: {missing_training_params}")
        
        # 从配置中提取参数
        self.max_epochs = self.experiment_config['max_epochs']
        self.lr = self.experiment_config['lr']
        self.reg = self.experiment_config['reg']
        self.opt = self.experiment_config['opt']
        self.early_stopping = self.experiment_config['early_stopping']
        self.batch_size = self.experiment_config['batch_size']
        
        # 初始化模型和损失函数
        self.model = None
        self.loss_fn = None
        self.scheduler = None

    def _init_model(self) -> nn.Module:
        """初始化模型"""
        # 从model_config中获取参数并构建配置
        config = self.model_config.copy()
        
        # 使用模型工厂创建模型
        model = ModelFactory.create_model(config)
        
        return model.to(device)
    
    def train_fold(self, 
                   datasets: Tuple[Any, Any, Any],
                   fold_idx: int) -> Tuple[Dict, float, float, float, float]:
        """
        Level 1: Fold训练主入口
        
        Args:
            datasets: (train_dataset, val_dataset, test_dataset)
            fold_idx: fold索引
            
        Returns:
            (results_dict, test_auc, val_auc, test_acc, val_acc)
        """
        print(f'\nTraining Fold {fold_idx}!')
        
        # 创建目录和日志记录器
        metrics_logger = Logger(self.model_config['n_classes'], self.log_dir, fold_idx)

        # 保存数据集分割
        train_split, val_split, test_split = datasets
        save_splits(datasets, ['train', 'val', 'test'], 
                   os.path.join(self.results_dir, 'splits_{}.csv'.format(fold_idx)))
        
        print(f"Training on {len(train_split)} samples")
        print(f"Validating on {len(val_split)} samples")
        print(f"Testing on {len(test_split)} samples")

        # 初始化模型和损失函数
        model = self._init_model()
        self.loss_fn = model.loss_fn
        print_network(model)
        optimizer = get_optim(model, self.opt, self.lr, self.reg)
        
        # 初始化学习率调度器
        scheduler_config = self.experiment_config.get('scheduler_config', {})
        self.scheduler = get_scheduler(optimizer, scheduler_config)
        if self.scheduler:
            print(f"🎯 使用学习率调度器: {scheduler_config.get('type', 'unknown')}")
        
        # 初始化数据加载器
        seed = self.experiment_config['seed']
        train_loader = get_split_loader(train_split, training=True, weighted=True, batch_size=1, generator=torch.Generator().manual_seed(seed))
        val_loader = get_split_loader(val_split, training=False, weighted=False, batch_size=1, generator=torch.Generator().manual_seed(seed))
        test_loader = get_split_loader(test_split, training=False, weighted=False, batch_size=1, generator=torch.Generator().manual_seed(seed))

        # 初始化早停
        # 从配置中获取早停参数，支持自定义指标和模式
        early_stopping_config = self.experiment_config.get('early_stopping_config', {})
        
        if self.early_stopping:
            # 如果 early_stopping 是字典，使用字典中的配置；否则使用默认配置
            if isinstance(self.early_stopping, dict):
                config = {**early_stopping_config, **self.early_stopping}
            else:
                config = early_stopping_config
            
            # 获取配置参数，使用默认值
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
            # 保存 metric 配置，用于后续选择指标
            early_stopping_obj.metric = metric
        else:
            early_stopping_obj = None
        
        # 2. 训练
        for epoch in range(self.max_epochs):
            # 训练和验证
            train_metrics = self._train_single_epoch(epoch, train_loader, optimizer, model, metrics_logger)
            val_metrics, stop = self._validate_single_epoch(fold_idx, epoch, val_loader, model, early_stopping_obj)
            
            # 记录日志
            metrics_logger.log_epoch(epoch, train_metrics, val_metrics, optimizer.param_groups[0]['lr'])
            
            # 更新学习率调度器
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    # ReduceLROnPlateau需要验证损失
                    self.scheduler.step(val_metrics['loss'])
                else:
                    # 其他调度器使用epoch
                    self.scheduler.step()
            
            if stop: 
                break
        
        # 3. 最终评估和返回结果
        # 保存模型
        checkpoint_path = os.path.join(self.results_dir, "s_{}_checkpoint.pt".format(fold_idx))
        if self.early_stopping:
            model.load_state_dict(torch.load(checkpoint_path))
        else:
            torch.save(model.state_dict(), checkpoint_path)

        # 最终评估
        _, val_accuracy, val_auc, _ = self._evaluate_model(val_loader, model)
        results_dict, test_accuracy, test_auc, eval_logger = self._evaluate_model(test_loader, model)
        
        print('Val accuracy: {:.4f}, ROC AUC: {:.4f}'.format(val_accuracy, val_auc))
        print('Test accuracy: {:.4f}, ROC AUC: {:.4f}'.format(test_accuracy, test_auc))

        # 打印各类别准确率
        for i in range(self.model_config['n_classes']):
            acc, correct, count = eval_logger.get_class_accuracy(i)
            print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        # 保存训练总结
        metrics_logger.save_summary({
            'acc': test_accuracy,
            'auc': test_auc,
            'loss': 1-test_accuracy
        })
            
        return results_dict, test_auc, val_auc, test_accuracy, val_accuracy

    def _train_single_epoch(self, epoch: int, loader: DataLoader, optimizer: torch.optim.Optimizer, model: nn.Module, logger: Logger) -> Dict:
        """
        Level 3: 标准模型单个epoch训练
        """
        model.train()
        
        # 🔧 重置epoch统计信息，确保每个epoch的统计是独立的
        logger.reset_epoch_stats()

        print('\n')
        batch_size = self.experiment_config['batch_size']
        total_loss = 0
        for batch_idx, (data, label) in enumerate(loader):
            # 标签已经是tensor，直接移动到设备
            label = label.to(device)
            
            # data 现在是字典格式，每个channel包含一个张量
            # 需要将每个channel的张量移动到设备上
            for channel in data:
                data[channel] = data[channel].to(device)
            results = model(data, label)
            Y_prob = results['probabilities']
            Y_hat = results['predictions']
            
            # 计算损失
            results['labels'] = label
            loss = self.loss_fn(results['logits'], results['labels'], results)
            total_loss += loss
            # 记录指标
            logger.log_batch(Y_hat, label, Y_prob, loss)
            
            if (batch_idx + 1) % batch_size == 0:
                # 反向传播
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
            # 计算剩余batch的数量
            remaining_batches = len(loader) % batch_size
            # 反向传播
            if hasattr(model, 'group_loss_fn'):
                results['group_loss'] = model.group_loss_fn(results)
                total_loss += results['group_loss']
            total_loss = total_loss / remaining_batches  # 使用剩余batch数量进行平均
            results['total_loss'] = total_loss.item()
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if hasattr(model, 'verbose_items'):
                items = model.verbose_items(results)
                if len(items) > 0:
                    print('Final batch: ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
            total_loss = 0
        # 计算平均指标
        train_loss = logger.batch_log['loss'] / len(loader)

        print('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}'.format(epoch, train_loss, logger.get_overall_accuracy()))
        if hasattr(model, 'verbose_items'):
            results['is_epoch'] = True
            items = model.verbose_items(results)
            if len(items) > 0:
                print('- ' + ' '.join([f'{key}: {value:.4f}' for key, value in items]))
        
        # 计算并返回指标
        return self._calculate_epoch_metrics(logger)

    def _calculate_epoch_metrics(self, logger: Dict) -> Dict:
        """计算epoch指标"""
        n_classes = self.model_config['n_classes']
        
        # 计算准确率
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

        # 计算AUC - 使用 torchmetrics（Tensor/GPU 原生）
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
        """验证函数"""
        model.eval()
        n_classes = self.model_config['n_classes']
        logger = Logger(n_classes=n_classes)
        
        # 重置模型的group_logits和group_labels，确保验证时从干净状态开始
        if hasattr(model, 'group_logits'):
            model.group_logits = []
        if hasattr(model, 'group_labels'):
            model.group_labels = []
        
        with torch.no_grad():
            for batch_idx, (data, label) in enumerate(loader):
                label = label.to(device)
                
                # data 现在是字典格式，每个channel包含一个张量
                # 需要将每个channel的张量移动到设备上
                for channel in data:
                    data[channel] = data[channel].to(device)

                results = model(data, label)
                Y_prob = results['probabilities']
                Y_hat = results['predictions']

                results['labels'] = label
                loss = self.loss_fn(results['logits'], results['labels'], results)
                logger.log_batch(Y_hat, label, Y_prob, loss)
        
        # 在验证结束时计算AUC损失
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
            # 根据配置的 metric 选择使用的指标
            metric_name = getattr(early_stopping, 'metric', 'auc')
            if metric_name == 'loss':
                score = val_loss
            elif metric_name == 'auc':
                score = auc
            elif metric_name == 'acc' or metric_name == 'accuracy':
                score = val_acc
            else:
                # 默认使用 auc
                score = auc
                print(f"⚠️ 警告: 未知的早停指标 '{metric_name}'，使用默认值 'auc'")
            
            early_stopping(epoch, score, model, 
                         ckpt_name=os.path.join(self.results_dir, "s_{}_checkpoint.pt".format(cur)))
            
            if early_stopping.early_stop:
                print("Early stopping")
                return val_metrics, True

        return val_metrics, False

    def _evaluate_model(self, loader: DataLoader, model: nn.Module, drop_prob: Optional[float] = None) -> Tuple[Dict, float, float, Logger]:
        """
        模型评估总结
        
        Args:
            loader: 数据加载器
            model: 模型
            drop_prob: 模态丢弃概率（0.0-1.0），在 forward 时传入模型
        """
        model.eval()
        logger = Logger(n_classes=self.model_config['n_classes'])

        # 重置模型的group_logits和group_labels，确保测试时从干净状态开始
        if hasattr(model, 'group_logits'):
            model.group_logits = []
        if hasattr(model, 'group_labels'):
            model.group_labels = []

        dataset_ref = loader.dataset
        case_ids_list: List[str]
        if hasattr(dataset_ref, 'case_ids'): # 直接数据集（拥有 case_ids 属性）
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
                # 传入 drop_prob 参数
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
        
        # 在测试结束时计算AUC损失
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
        仅评测接口：加载指定checkpoint，在给定datasets的测试集上评测。

        Args:
            datasets: (train_dataset, val_dataset, test_dataset) 元组，测试集将被用于评测
            fold_idx: 当前fold索引（用于日志打印/兼容接口）
            checkpoint_path: 模型权重路径（推荐为 train_fold 保存的 s_{fold}_checkpoint.pt）
            drop_prob: 模态丢弃概率（0.0-1.0），在 forward 时传入模型

        Returns:
            (results_dict, test_auc, None, test_acc, None) 与 train_fold 结果形式对齐（验证指标置为 None）
        """
        print(f"\n[Evaluate] Fold {fold_idx} | checkpoint: {checkpoint_path}")

        # 每次评测都重新初始化模型（不复用之前的模型状态）
        model = self._init_model()
        print(f"🔧 创建新模型实例，id={id(model)}")
        self.loss_fn = model.loss_fn  # 更新 loss_fn 为当前模型的
        
        # 加载checkpoint（与训练时的load方式一致）
        state = torch.load(checkpoint_path, map_location=device)
        print(f"📦 checkpoint加载成功，state_dict keys数量: {len(state.keys())}")
        
        if hasattr(model, 'transfer_layer') and hasattr(model, 'create_transfer_layer'):
            # 从checkpoint中找到所有transfer_layer的通道
            transfer_layer_channels = {}
            for key in state.keys():
                if 'transfer_layer.' in key:
                    # 提取通道名和权重类型，例如 "transfer_layer.clinical=val.weight" -> ("clinical=val", "weight")
                    parts = key.split('.')
                    if len(parts) >= 3:
                        channel_name = parts[1]  # 例如 "clinical=val"
                        weight_type = parts[2]  # "weight" 或 "bias"
                        
                        if channel_name not in transfer_layer_channels:
                            transfer_layer_channels[channel_name] = {}
                        transfer_layer_channels[channel_name][weight_type] = state[key]
            
            # 根据checkpoint中的权重创建对应的transfer_layer
            if hasattr(model, 'output_dim'):
                output_dim = model.output_dim
                print(f"🔧 预创建 {len(transfer_layer_channels)} 个transfer_layer以匹配checkpoint...")
                for channel_name, weights in transfer_layer_channels.items():
                    if channel_name not in model.transfer_layer:
                        # 从weight的形状推断input_dim: weight形状是 [output_dim, input_dim]
                        if 'weight' in weights:
                            weight_tensor = weights['weight']
                            if len(weight_tensor.shape) == 2:
                                input_dim = weight_tensor.shape[1]  # 第二维是input_dim
                                # 创建transfer_layer
                                transfer_layer = model.create_transfer_layer(input_dim)
                                model.transfer_layer[channel_name] = transfer_layer
                                print(f"   ✅ 创建 transfer_layer.{channel_name} (input_dim={input_dim}, output_dim={output_dim})")
                            else:
                                print(f"   ⚠️ 无法推断 {channel_name} 的input_dim: weight形状异常 {weight_tensor.shape}")
                        else:
                            print(f"   ⚠️ checkpoint中缺少 {channel_name}.weight，无法创建transfer_layer")
        
        # 分析checkpoint中的权重类型
        transfer_layer_keys = [k for k in state.keys() if 'transfer_layer.' in k]
        core_keys = [k for k in state.keys() if 'transfer_layer.' not in k]
        
        print(f"📊 checkpoint权重分析:")
        print(f"   核心权重: {len(core_keys)} 个")
        print(f"   transfer_layer权重: {len(transfer_layer_keys)} 个")
        
        # 现在所有需要的transfer_layer都已创建，尝试使用strict=True确保完全匹配
        # 如果还有不匹配，再降级到strict=False
        try:
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=True)
            print(f"✅ 使用strict=True成功加载所有权重（完全匹配）")
        except RuntimeError as e:
            # 如果strict=True失败，使用strict=False但会详细报告
            print(f"⚠️ strict=True加载失败: {e}")
            print(f"🔧 降级到strict=False加载...")
            missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
        
        # 检查核心权重是否都加载了
        model_core_keys = set([k for k in model.state_dict().keys() if 'transfer_layer.' not in k])
        checkpoint_core_keys = set(core_keys)
        loaded_core_keys = model_core_keys & checkpoint_core_keys
        
        if missing_keys:
            missing_core = [k for k in missing_keys if 'transfer_layer.' not in k]
            missing_transfer = [k for k in missing_keys if 'transfer_layer.' in k]
            if missing_core:
                print(f"⚠️ 警告：缺少以下核心权重（可能导致性能下降）: {len(missing_core)} 个")
                for key in missing_core[:5]:
                    print(f"    - {key}")
                if len(missing_core) > 5:
                    print(f"    ... 还有 {len(missing_core) - 5} 个")
            if missing_transfer:
                print(f"ℹ️ 信息：缺少以下transfer_layer权重（将在forward时动态创建）: {len(missing_transfer)} 个")
        
        if unexpected_keys:
            unexpected_transfer = [k for k in unexpected_keys if 'transfer_layer.' in k]
            unexpected_other = [k for k in unexpected_keys if 'transfer_layer.' not in k]
            if unexpected_transfer:
                print(f"ℹ️ 信息：checkpoint中有额外的transfer_layer权重（已忽略，不影响评测）: {len(unexpected_transfer)} 个")
            if unexpected_other:
                print(f"⚠️ 警告：checkpoint中有意外的其他权重: {len(unexpected_other)} 个")
                for key in unexpected_other[:5]:
                    print(f"    - {key}")
        
        # 验证核心权重加载情况
        print(f"✅ 核心权重加载: {len(loaded_core_keys)}/{len(checkpoint_core_keys)} 个")
        if len(loaded_core_keys) == len(checkpoint_core_keys):
            print(f"✅ 所有核心权重已成功加载")
        else:
            print(f"⚠️ 警告：部分核心权重未加载: {len(checkpoint_core_keys) - len(loaded_core_keys)} 个")
        
        # 设置为评估模式
        model.eval()

        # 仅构造测试集数据加载器
        _, _, test_split = datasets
        test_loader = get_split_loader(test_split, training=False, weighted=False, batch_size=1)

        # 评测（传入 drop_prob）
        results_dict, test_acc, test_auc, _ = self._evaluate_model(test_loader, model, drop_prob=drop_prob)
        return results_dict, float(test_auc), None, float(test_acc), None

    def evaluate_with_checkpoint(self,
                                 datasets: Tuple[Any, Any, Any],
                                 fold_idx: int,
                                 checkpoint_path: str,
                                 drop_prob: Optional[float] = None) -> Tuple[Dict, float, Optional[float], float, Optional[float]]:
        """
        兼容名：直接调用 evaluate_fold。
        
        Args:
            datasets: 数据集元组
            fold_idx: fold索引
            checkpoint_path: checkpoint路径
            drop_prob: 模态丢弃概率（0.0-1.0），在 forward 时传入模型
        """
        return self.evaluate_fold(datasets=datasets, fold_idx=fold_idx, checkpoint_path=checkpoint_path, drop_prob=drop_prob)
