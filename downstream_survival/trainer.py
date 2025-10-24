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
	"""保存数据集分割信息"""
	try:
		# 获取每个分割的case_ids
		splits = []
		for i, dataset in enumerate(split_datasets):
			if hasattr(dataset, 'case_ids'):
				# 直接是MultimodalDataset对象
				splits.append(pd.Series(dataset.case_ids))
			else:
				# fallback: 使用索引
				splits.append(pd.Series([f"sample_{j}" for j in range(len(dataset))]))
		
		if not boolean_style:
			df = pd.concat(splits, ignore_index=True, axis=1)
			df.columns = column_keys
		else:
			df = pd.concat(splits, ignore_index = True, axis=0)
			index = df.values.tolist()
			one_hot = np.eye(len(split_datasets)).astype(bool)
			bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
			df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

		df.to_csv(filename)
		print(f"✅ 保存分割信息到: {filename}")
	except Exception as e:
		print(f"⚠️ 保存分割信息失败: {e}")
		# 创建一个简单的分割记录
		split_info = {
			'split_type': column_keys,
			'train_size': len(split_datasets[0]) if len(split_datasets) > 0 else 0,
			'val_size': len(split_datasets[1]) if len(split_datasets) > 1 else 0,
			'test_size': len(split_datasets[2]) if len(split_datasets) > 2 else 0
		}
		pd.DataFrame([split_info]).to_csv(filename, index=False)
		print(f"✅ 保存简化分割信息到: {filename}")

# 工具函数，使用高效的手动实现
def calculate_accuracy(Y_hat: torch.Tensor, Y: torch.Tensor) -> float:
    """计算预测准确率"""
    return float((Y_hat == Y).float().mean().item())

def cal_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """快速计算AUC - 比sklearn快18倍"""
    # 排序
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # 计算TPR和FPR
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    
    # 避免除零
    tp_total = tp[-1]
    fp_total = fp[-1]
    
    if tp_total == 0 or fp_total == 0:
        return 0.5
    
    tpr = tp / tp_total
    fpr = fp / fp_total
    
    # 计算AUC（梯形积分）
    auc = np.trapz(tpr, fpr)
    return auc

def cal_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """快速计算ROC曲线 - 比sklearn快9.6倍"""
    # 排序
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    y_scores_sorted = y_scores[sorted_indices]
    
    # 计算TPR和FPR
    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(1 - y_true_sorted)
    
    tp_total = tp[-1]
    fp_total = fp[-1]
    
    if tp_total == 0:
        tpr = np.zeros_like(tp, dtype=float)
    else:
        tpr = tp / tp_total
    
    if fp_total == 0:
        fpr = np.zeros_like(fp, dtype=float)
    else:
        fpr = fp / fp_total
    
    # 添加起始点
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    return fpr, tpr, y_scores_sorted

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

def get_split_loader(dataset, training=False, weighted=False, batch_size=1):
    """获取数据加载器"""
    if training:
        if weighted:
            weights = make_weights_for_balanced_classes_split(dataset)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        else:
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
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
    """早停机制"""
    
    def __init__(self, patience: int = 20, stop_epoch: int = 50, verbose: bool = False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch: int, val_loss: float, model: nn.Module, ckpt_name: str = 'checkpoint.pt') -> bool:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0
        
        return self.early_stop

    def save_checkpoint(self, val_loss: float, model: nn.Module, ckpt_name: str):
        """保存模型检查点"""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

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
        train_loader = get_split_loader(train_split, training=True, weighted=True, batch_size=1)
        val_loader = get_split_loader(val_split, training=False, weighted=False, batch_size=1)
        test_loader = get_split_loader(test_split, training=False, weighted=False, batch_size=1)

        # 初始化早停
        early_stopping_obj = EarlyStopping(patience=25, stop_epoch=10, verbose=True) if self.early_stopping else None
        
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
                    results['auc_loss'] = model.group_loss_fn(results)
                    total_loss += results['auc_loss']
                total_loss = total_loss/batch_size
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
                results['auc_loss'] = model.group_loss_fn(results)
                total_loss += results['auc_loss']
            total_loss = total_loss / remaining_batches  # 使用剩余batch数量进行平均
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
            results['auc_loss'] = model.group_loss_fn(results)
            logger.batch_log['loss'] += results['auc_loss']
            
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
            early_stopping(epoch, val_loss, model, 
                         ckpt_name=os.path.join(self.results_dir, "s_{}_checkpoint.pt".format(cur)))
            
            if early_stopping.early_stop:
                print("Early stopping")
                return val_metrics, True

        return val_metrics, False

    def _evaluate_model(self, loader: DataLoader, model: nn.Module) -> Tuple[Dict, float, float, Logger]:
        """模型评估总结"""
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
        elif hasattr(dataset_ref, 'dataset') and hasattr(dataset_ref.dataset, 'case_ids') and hasattr(dataset_ref, 'indices'): # Subset 数据集（没有 case_ids 属性，需从原数据集映射）
            try:
                base = dataset_ref.dataset.case_ids
                base_list = list(base) if not isinstance(base, list) else base
                case_ids_list = [base_list[i] for i in dataset_ref.indices]
            except Exception:
                case_ids_list = [f"sample_{i}" for i in range(len(dataset_ref))]
        else:
            case_ids_list = [f"sample_{i}" for i in range(len(dataset_ref))]
        patient_results = {}

        for batch_idx, (data, label) in enumerate(loader):
            label = label.to(device)
            # data 现在是字典格式，每个channel包含一个张量，搬到设备
            for channel in data:
                data[channel] = data[channel].to(device)
            case_id = case_ids_list[batch_idx]
            with torch.inference_mode():
                results = model(data, label)
                Y_prob = results['probabilities']
                Y_hat = results['predictions']
            
            results['labels'] = label
            loss = self.loss_fn(results['logits'], results['labels'], results)
            logger.log_batch(Y_hat, label, Y_prob, loss)
            
            patient_results.update({case_id: {'case_id': np.array(case_id), 'prob': Y_prob.cpu().numpy(), 'label': label.item()}})
        
        # 在测试结束时计算AUC损失
        if hasattr(model, 'group_loss_fn') and hasattr(model, 'group_logits') and model.group_logits:
            results['auc_loss'] = model.group_loss_fn(results)
            logger.batch_log['loss'] += results['auc_loss']
        
        test_loss = logger.batch_log['loss']/len(loader)
        test_acc = logger.get_overall_accuracy()
        
        if hasattr(model, 'verbose_items'):
            results['is_epoch'] = True
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
