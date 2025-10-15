"""
多模态对齐训练器模块
从UNI_finetune.py中分离出来的训练器类
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import torch.nn.functional as F
import time
import numpy as np

# 导入模型
from alignment_model import MultiModalAlignmentModel

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiModalAlignmentTrainer:
    """
    多模态对齐训练器 - 用于多个modality之间的alignment
    """
    
    def __init__(self, 
                 model: MultiModalAlignmentModel,
                 device: str = "cuda",
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5,
                 loss_type: str = "rank1",
                 tau1: float = 0.1,
                 tau2: float = 0.1,
                 lambda1: float = 1.0,
                 lambda2: float = 0.1,
                 mismatch_ratio: float = 1.0,
                 modality_names_for_mismatch: Optional[List[str]] = None,
                 val_max_batches: Optional[int] = None,
                 loss2_chunk_size: Optional[int] = None,
                 verbose_timing: bool = False,
                 early_stopping_patience: int = 10,
                 early_stopping_min_delta: float = 1e-4):
        """
        初始化训练器
        
        Args:
            model: 多模态对齐模型
            device: 训练设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            loss_type: 损失函数类型，"volume", "rank1"，默认 "rank1"
            tau1: 温度参数
            tau2: 温度参数
            lambda1: 损失函数参数
            lambda2: 损失函数参数
            mismatch_ratio: mismatch与match的比例，1.0表示1:1，2.0表示2:1
            modality_names_for_mismatch: 可选，指定用于mismatch的模态名称，默认使用 model.modality_names
            val_max_batches: 验证时最大批次数
            loss2_chunk_size: loss2分块大小
            verbose_timing: 是否启用详细性能分析
            early_stopping_patience: early stopping的耐心值（验证loss不改善的步数）
            early_stopping_min_delta: early stopping的最小改善阈值
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.tau1 = tau1
        self.tau2 = tau2
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mismatch_ratio = mismatch_ratio
        self.modality_names_for_mismatch = modality_names_for_mismatch or model.modality_names
        self.val_max_batches = val_max_batches
        self.loss2_chunk_size = loss2_chunk_size
        self.verbose_timing = verbose_timing
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        
        # Early stopping 相关状态
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False
        
        # 性能分析相关（仅在verbose_timing=True时启用）
        if self.verbose_timing:
            self.timing_stats = {
                'data_loading': [],
                'forward_pass': [],
                'loss_computation': [],
                'loss1_computation': [],
                'loss2_computation': [],
                'loss3_computation': [],
                'backward_pass': [],
                'optimizer_step': [],
                'total_step': []
            }
        else:
            self.timing_stats = None
        
        if loss_type not in ["volume", "rank1"]:
            raise ValueError(f"不支持的损失类型: {loss_type}，支持的类型: 'volume', 'rank1'")
        self.loss_type = loss_type
        
        alignment_params = []
        for modality_name in model.modality_names:
            alignment_params.extend(list(model.alignment_layers[modality_name].parameters()))
        
        self.optimizer = optim.AdamW(
            alignment_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100, eta_min=1e-6
        )
        
        logger.info(f"✅ 多模态对齐训练器初始化完成")
        logger.info(f"   - 设备: {device}")
        logger.info(f"   - 学习率: {learning_rate}")
        logger.info(f"   - 权重衰减: {weight_decay}")
        logger.info(f"   - 优化器: AdamW")
        logger.info(f"   - 调度器: CosineAnnealingLR")
        logger.info(f"   - 损失函数: {loss_type.upper()} Loss(rank1)")
        logger.info(f"   - Mismatch比例: {mismatch_ratio}:1")
        logger.info(f"   - 模态数量: {model.num_modalities}")
        logger.info(f"   - 模态名称: {model.modality_names}")
        if self.val_max_batches is not None:
            logger.info(f"   - 每次验证最多批次数: {self.val_max_batches}")
        if self.loss2_chunk_size is not None:
            logger.info(f"   - loss2 分块大小: {self.loss2_chunk_size}")
        if self.verbose_timing:
            logger.info(f"   - 详细性能分析: 启用")
        if self.early_stopping_patience > 0:
            logger.info(f"   - Early Stopping: 启用 (patience={self.early_stopping_patience}, min_delta={self.early_stopping_min_delta})")
    
    def _compute_loss_with_metrics(self, aligned_features: Dict[str, torch.Tensor], 
                                    aligned_negatives: Optional[Dict[str, torch.Tensor]] = None):
        """
        计算损失并返回 SVD 特征值用于监控
        
        Returns:
            loss: 损失值
            svd_values: SVD 特征值 Tensor[num_modalities]，如果不适用则返回 None
        """
        if self.loss_type == "volume":
            return self._compute_volume_loss_with_metrics(aligned_features)
        elif self.loss_type == "rank1":
            return self._compute_rank1_loss_with_metrics(aligned_features, aligned_negatives)
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")
        
    def _volume_computation(self, language, *inputs):
        """
        General function to compute volume for contrastive learning loss functions.
        Compute the volume metric for each vector in language batch and all the other modalities listed in *inputs.

        Args:
        - language (torch.Tensor): Tensor of shape (batch_size1, dim)
        - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

        Returns:
        - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
        """
        batch_size1 = language.shape[0]
        batch_size2 = inputs[0].shape[0]

        # Compute pairwise dot products for language with itself
        ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

        # Compute pairwise dot products for language with each input
        l_inputs = [language @ input.T for input in inputs]

        # Compute pairwise dot products for each input with themselves and with each other
        input_dot_products = []
        for i, input1 in enumerate(inputs):
            row = []
            for j, input2 in enumerate(inputs):
                dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
                row.append(dot_product)
            input_dot_products.append(row)

        # Stack the results to form the Gram matrix for each pair
        G = torch.stack([
            torch.stack([ll] + l_inputs, dim=-1),
            *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
        ], dim=-2)
        
        evals = torch.linalg.eigvalsh(G.to(torch.float64))        # [B1,B2,K], 升序
        evals = evals.clamp_min(0.0).to(G.dtype)

        # Compute the determinant for each Gram matrix
        gram_det = torch.det(G.float())

        # Compute the square root of the absolute value of the determinants
        res = torch.sqrt(torch.abs(gram_det))
        return res, evals
    
    def _compute_volume_loss_with_metrics(self, aligned_features: dict):
        """
        aligned_features: dict[str, Tensor]，每个 Tensor 形状 [B, D]，索引 i 处各模态一一对应
        返回: vol: [B]（每个样本一个 体积 或 log-体积）, svd_values: [K]
        """
        feature_list = list(aligned_features.values())
        volume, evals = self._volume_computation(feature_list[0], *feature_list[1:])
        # 2) 温度缩放并取负号作为 logits（越小越好交给 CE）
        logits_ab = - volume / self.tau1                        # anchor->others
        logits_ba = - volume.t() / self.tau1                    # others->anchor

        # 3) 目标指定对角
        B = volume.size(0)
        targets = torch.arange(B, device=volume.device)

        # 4) 交叉熵（可带 label_smoothing）
        loss = (F.cross_entropy(logits_ab, targets, label_smoothing=0.1) \
            + F.cross_entropy(logits_ba, targets, label_smoothing=0.1)) / 2
        loss = loss.mean()
        
        # 将特征值从 [B1, B2, K] 按批维聚合为 [K] 并按降序排序（用于日志展示）
        svd_values, _ = torch.sort(evals.mean(dim=(0, 1)).detach(), descending=True)
        return loss, svd_values
    
    def _compute_rank1_loss_with_metrics(self, aligned_features: Dict[str, torch.Tensor], 
                                          aligned_negatives: Optional[Dict[str, torch.Tensor]] = None):
        """
        计算 rank1 损失并返回 SVD 特征值（带详细时间分析）
        
        Returns:
            loss: 损失值
            svd_values: SVD 特征值 Tensor[num_modalities]
        """
        # 1. SVD 计算和 loss1
        loss1_start = time.perf_counter() if self.verbose_timing else None
        
        feature_list = list(aligned_features.values())
        features = torch.stack(feature_list, dim=-1)  # [batch_size, feature_dim, num_modalities]
        
        # L2 归一化：x <- x / (||x||_2 + ε)
        eps = 1e-8
        l2_norm = torch.norm(features, p=2, dim=1, keepdim=True)  # [batch_size, 1, num_modalities]
        features = features / (l2_norm + eps)
        
        # U: [batch_size, feature_dim, num_modalities]
        # S(diag): [batch_size, num_modalities]
        # _: [batch_size, feature_dim, num_modalities]
        U, S, _ = torch.linalg.svd(features)
        
        # 📊 记录 SVD 特征值：对 batch 维度求平均（用于记录单个 batch 的代表值）
        svd_values = S.mean(dim=0).detach()  # [num_modalities]
        
        loss1 = F.cross_entropy(S / self.tau1, torch.zeros(S.shape[0]).to(S.device).long())
        
        if self.verbose_timing:
            loss1_time = time.perf_counter() - loss1_start
            self.timing_stats['loss1_computation'].append(loss1_time)
        
        # 2. loss2 计算
        loss2_start = time.perf_counter() if self.verbose_timing else None
        
        U1 = U[:, :, 0] # dominate projection [batch_size, feature_dim]
        # 组内矩阵计算：按 loss2_chunk_size 将 batch 分组，仅组内做 softmax/CE
        batch_count = U1.shape[0]
        if self.loss2_chunk_size is None or self.loss2_chunk_size >= batch_count:
            loss2 = F.cross_entropy((U1 @ U1.T) / self.tau2, torch.arange(batch_count, device=U1.device).long())
        else:
            c = max(1, int(self.loss2_chunk_size))
            full = (batch_count // c) * c
            loss2_sum = U1.new_tensor(0.0)
            if full > 0:
                groups = U1[:full].view(-1, c, U1.shape[1])  # [G, c, D]
                logits_gc = torch.einsum('gxd,gyd->gxy', groups, groups) / self.tau2  # [G, c, c]
                targets_gc = torch.arange(c, device=U1.device).expand(logits_gc.shape[0], c) # [G, c]
                loss2_sum = loss2_sum + F.cross_entropy(
                    logits_gc.reshape(-1, c), targets_gc.reshape(-1), reduction='sum'
                )
            if full < batch_count:
                tail = U1[full:]
                c_tail = tail.shape[0]
                logits_tail = (tail @ tail.T) / self.tau2
                targets_tail = torch.arange(c_tail, device=U1.device)
                loss2_sum = loss2_sum + F.cross_entropy(logits_tail, targets_tail, reduction='sum')
            loss2 = loss2_sum / batch_count

        if self.verbose_timing:
            loss2_time = time.perf_counter() - loss2_start
            self.timing_stats['loss2_computation'].append(loss2_time)

        if self.lambda2 == 0:
            return loss1 + self.lambda1 * loss2, svd_values

        # 3. loss3 (loss_IM) 计算
        loss3_start = time.perf_counter() if self.verbose_timing else None
        
        batch_size = feature_list[0].shape[0]
        positive_labels = torch.ones(batch_size, device=self.device)
        
        def fuse(feat_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            # 将多模态特征拼接为单向量 [N, d*K]
            return torch.cat(list(feat_dict.values()), dim=1)

        if aligned_negatives is None:
            raise RuntimeError("Negative features not provided by dataset. Ensure DataLoader yields 'features_neg' per batch.")
        neg_fused = fuse(aligned_negatives)

        pos_fused = fuse(aligned_features)
        all_features = torch.cat([pos_fused, neg_fused], dim=0)
        negative_labels = torch.zeros(neg_fused.shape[0], device=self.device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        pred_M = self.model.mlp_predictor(all_features)
        loss_IM = F.binary_cross_entropy(pred_M.squeeze(), all_labels)
        
        if self.verbose_timing:
            loss3_time = time.perf_counter() - loss3_start
            self.timing_stats['loss3_computation'].append(loss3_time)
        
        total_loss = loss1 + self.lambda1 * loss2 + self.lambda2 * loss_IM
        return total_loss, svd_values
    
    def _get_next_batch(self, train_iter, train_loader):
        """获取下一个batch，如果dataloader用完则重新开始"""
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            self.scheduler.step()  # 每个epoch结束更新学习率
        return batch
    
    def _update_progress_bar(self, progress_bar, loss: float, svd_values: Optional[torch.Tensor]):
        """更新进度条显示（当前 batch 的原始值）"""
        postfix_dict = {'loss': f'{loss:.4f}'}
        if svd_values is not None:
            # 显示前3个特征值（已经是 batch 内平均过的值）
            for i in range(min(3, len(svd_values))):
                postfix_dict[f'σ{i+1}'] = f'{svd_values[i].item():.3f}'
        progress_bar.set_postfix(postfix_dict)
        progress_bar.update(1)
    
    def _run_validation(self, val_loader: DataLoader, val_max_batches: Optional[int] = None) -> Tuple[float, Optional[torch.Tensor]]:
        """
        运行验证并返回平均loss和平均SVD值
        
        Returns:
            (avg_loss, avg_svd): 平均验证损失和平均SVD特征值
        """
        self.model.eval()
        val_result = self._validate_full(val_loader, val_max_batches)
        
        self.model.train()
        
        # 计算平均 loss
        avg_loss = sum(val_result['batch_losses']) / len(val_result['batch_losses'])
        
        # 计算平均 SVD 值
        avg_svd = None
        if val_result['batch_svd_values']:
            avg_svd = torch.stack(val_result['batch_svd_values']).mean(dim=0)
        
        return avg_loss, avg_svd
    
    def _save_checkpoint(self, save_path: str, step: int, val_loss: float):
        """保存模型checkpoint"""
        torch.save({
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
        }, save_path)
        logger.info(f"✅ [Step {step}] 保存最佳模型 (val_loss: {val_loss:.4f})")
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """
        检查是否应该early stopping
        
        Args:
            val_loss: 当前验证损失
            
        Returns:
            bool: True表示应该early stopping，False表示继续训练
        """
        if self.early_stopping_patience <= 0:
            return False
        
        # 检查是否有改善
        if val_loss < self.best_val_loss - self.early_stopping_min_delta:
            # 有改善，重置计数器
            self.best_val_loss = val_loss
            self.early_stopping_counter = 0
            logger.info(f"🎯 [Early Stop] 验证损失改善: {val_loss:.4f} (最佳: {self.best_val_loss:.4f})")
            return False
        else:
            # 没有改善，增加计数器
            self.early_stopping_counter += 1
            logger.info(f"⏳ [Early Stop] 验证损失无改善: {val_loss:.4f} (最佳: {self.best_val_loss:.4f}, 计数: {self.early_stopping_counter}/{self.early_stopping_patience})")
            
            # 检查是否达到patience
            if self.early_stopping_counter >= self.early_stopping_patience:
                self.early_stopping_triggered = True
                logger.info(f"🛑 [Early Stop] 触发早停！验证损失连续 {self.early_stopping_patience} 次验证无改善")
                return True
        
        return False
    
    def _log_progress(self, global_step: int, max_steps: int, log_interval: int, history: Dict):
        """打印训练进度日志"""
        # 计算最近 log_interval 步的平均
        recent_losses = history['train_losses'][-log_interval:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        log_msg = f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f}"
        
        # SVD 值
        if history['train_svd_values']:
            recent_svd = history['train_svd_values'][-log_interval:]
            if recent_svd:
                avg_svd = torch.stack(recent_svd).mean(dim=0)
                svd_msg = [f"σ{i+1}={avg_svd[i].item():.3f}" for i in range(min(3, len(avg_svd)))]
                log_msg += f" | SVD: [{', '.join(svd_msg)}]"
        
        # 最近的验证loss
        if history['val_losses']:
            log_msg += f" | Val: {history['val_losses'][-1]:.4f}"
        
        logger.info(log_msg)
    
    def _train_step(self, batch) -> Tuple[float, Optional[torch.Tensor]]:
        """
        执行单个训练步骤（可选性能分析）
        
        Args:
            batch: 数据批次
            
        Returns:
            (loss, svd_values): 损失值和SVD特征值
        """
        if self.verbose_timing:
            step_start = time.perf_counter()
            
            # 1. 数据加载和处理
            data_start = time.perf_counter()
            if isinstance(batch, dict) and 'features' in batch:
                pos_dict = {k: v.to(self.device) for k, v in batch['features'].items()}
                neg_dict = {k: v.to(self.device) for k, v in batch.get('features_neg', {}).items()} if 'features_neg' in batch else None
                pos_mask = {k: v.to(self.device).float() for k, v in batch.get('mask', {}).items()} if 'mask' in batch else None
                neg_mask = {k: v.to(self.device).float() for k, v in batch.get('mask_neg', {}).items()} if 'mask_neg' in batch else None
            else:
                pos_dict = {k: v.to(self.device) for k, v in batch.items()}
                neg_dict = None
                pos_mask = None
                neg_mask = None
            data_time = time.perf_counter() - data_start
            
            self.optimizer.zero_grad()
            
            # 2. 前向传播
            forward_start = time.perf_counter()
            aligned_pos = self.model(pos_dict)
            if pos_mask is not None:
                for m, x in aligned_pos.items():
                    if m in pos_mask:
                        aligned_pos[m] = x * pos_mask[m].unsqueeze(1)
            
            # 负样本
            if self.lambda2 == 0:
                aligned_neg = None
            else:
                if neg_dict is None:
                    raise RuntimeError("Dataset must provide features_neg for negative samples.")
                aligned_neg = self.model(neg_dict)
                if aligned_neg is not None and neg_mask is not None:
                    for m, x in aligned_neg.items():
                        if m in neg_mask:
                            aligned_neg[m] = x * neg_mask[m].unsqueeze(1)
            forward_time = time.perf_counter() - forward_start
            
            # 3. 损失计算
            loss_start = time.perf_counter()
            loss, svd_values = self._compute_loss_with_metrics(aligned_pos, aligned_negatives=aligned_neg)
            loss_time = time.perf_counter() - loss_start
            
            # 4. 反向传播
            backward_start = time.perf_counter()
            loss.backward()
            backward_time = time.perf_counter() - backward_start
            
            # 5. 优化器步骤
            optimizer_start = time.perf_counter()
            self.optimizer.step()
            optimizer_time = time.perf_counter() - optimizer_start
            
            total_time = time.perf_counter() - step_start
            
            # 记录时间统计
            self.timing_stats['data_loading'].append(data_time)
            self.timing_stats['forward_pass'].append(forward_time)
            self.timing_stats['loss_computation'].append(loss_time)
            self.timing_stats['backward_pass'].append(backward_time)
            self.timing_stats['optimizer_step'].append(optimizer_time)
            self.timing_stats['total_step'].append(total_time)
            
            return loss.item(), svd_values
        else:
            # 标准训练步骤（无性能分析）
            if isinstance(batch, dict) and 'features' in batch:
                pos_dict = {k: v.to(self.device) for k, v in batch['features'].items()}
                neg_dict = {k: v.to(self.device) for k, v in batch.get('features_neg', {}).items()} if 'features_neg' in batch else None
                pos_mask = {k: v.to(self.device).float() for k, v in batch.get('mask', {}).items()} if 'mask' in batch else None
                neg_mask = {k: v.to(self.device).float() for k, v in batch.get('mask_neg', {}).items()} if 'mask_neg' in batch else None
            else:
                pos_dict = {k: v.to(self.device) for k, v in batch.items()}
                neg_dict = None
                pos_mask = None
                neg_mask = None
            
            self.optimizer.zero_grad()
            
            # 前向传播
            aligned_pos = self.model(pos_dict)
            if pos_mask is not None:
                for m, x in aligned_pos.items():
                    if m in pos_mask:
                        aligned_pos[m] = x * pos_mask[m].unsqueeze(1)
            
            # 负样本
            if self.lambda2 == 0:
                aligned_neg = None
            else:
                if neg_dict is None:
                    raise RuntimeError("Dataset must provide features_neg for negative samples.")
                aligned_neg = self.model(neg_dict)
                if aligned_neg is not None and neg_mask is not None:
                    for m, x in aligned_neg.items():
                        if m in neg_mask:
                            aligned_neg[m] = x * neg_mask[m].unsqueeze(1)
            
            # 计算损失
            loss, svd_values = self._compute_loss_with_metrics(aligned_pos, aligned_negatives=aligned_neg)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            return loss.item(), svd_values
    
    def print_timing_analysis(self, num_steps: int = 100):
        """
        打印性能分析报告
        
        Args:
            num_steps: 用于分析的步数（取最近的N步）
        """
        if not self.verbose_timing or not self.timing_stats or not self.timing_stats['total_step']:
            logger.info("📊 性能分析未启用或无数据")
            return
            
        logger.info("=" * 80)
        logger.info("📊 训练性能分析报告")
        logger.info("=" * 80)
        
        # 取最近的N步进行分析
        recent_steps = min(num_steps, len(self.timing_stats['total_step']))
        
        for stage, times in self.timing_stats.items():
            if not times:
                continue
                
            recent_times = times[-recent_steps:]
            avg_time = np.mean(recent_times)
            std_time = np.std(recent_times)
            min_time = np.min(recent_times)
            max_time = np.max(recent_times)
            
            # 计算占总时间的百分比
            total_avg = np.mean(self.timing_stats['total_step'][-recent_steps:])
            percentage = (avg_time / total_avg) * 100 if total_avg > 0 else 0
            
            stage_name = {
                'data_loading': '数据加载',
                'forward_pass': '前向传播',
                'loss_computation': '损失计算',
                'loss1_computation': 'Loss1计算',
                'loss2_computation': 'Loss2计算',
                'loss3_computation': 'Loss3计算',
                'backward_pass': '反向传播',
                'optimizer_step': '优化器更新',
                'total_step': '总步时间'
            }.get(stage, stage)
            
            logger.info(f"🔹 {stage_name:12} | "
                       f"平均: {avg_time*1000:6.2f}ms | "
                       f"标准差: {std_time*1000:5.2f}ms | "
                       f"范围: [{min_time*1000:5.2f}, {max_time*1000:5.2f}]ms | "
                       f"占比: {percentage:5.1f}%")
        
        # 性能瓶颈分析
        logger.info("\n🔍 性能瓶颈分析:")
        bottleneck_stages = []
        for stage in ['data_loading', 'forward_pass', 'loss_computation', 'loss1_computation', 'loss2_computation', 'loss3_computation', 'backward_pass', 'optimizer_step']:
            if stage in self.timing_stats and self.timing_stats[stage]:
                recent_times = self.timing_stats[stage][-recent_steps:]
                avg_time = np.mean(recent_times)
                bottleneck_stages.append((stage, avg_time))
        
        # 按耗时排序
        bottleneck_stages.sort(key=lambda x: x[1], reverse=True)
        
        stage_names = {
            'data_loading': '数据加载',
            'forward_pass': '前向传播', 
            'loss_computation': '损失计算',
            'loss1_computation': 'Loss1计算',
            'loss2_computation': 'Loss2计算',
            'loss3_computation': 'Loss3计算',
            'backward_pass': '反向传播',
            'optimizer_step': '优化器更新'
        }
        
        for i, (stage, avg_time) in enumerate(bottleneck_stages):
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            logger.info(f"  {emoji} {stage_names.get(stage, stage)}: {avg_time*1000:.2f}ms")
        
        logger.info("=" * 80)
    
    @torch.no_grad()
    def _validate_full(self, val_loader: DataLoader, val_max_batches: Optional[int] = None) -> Dict[str, any]:
        """
        验证模型，返回每个batch的metrics
        
        Returns:
            {
                'batch_losses': List[float],
                'batch_svd_values': List[Tensor],
                'batch_indices': List[int],
            }
        """
        self.model.eval()
        batch_losses = []
        batch_svd_values = []
        batch_indices = []
        
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation", mininterval=1.0)):
            if val_max_batches is not None and batch_idx >= val_max_batches:
                break
            if isinstance(batch, dict) and 'features' in batch:
                pos_dict = {k: v.to(self.device) for k, v in batch['features'].items()}
                neg_dict = {k: v.to(self.device) for k, v in batch.get('features_neg', {}).items()} if 'features_neg' in batch else None
                pos_mask = {k: v.to(self.device).float() for k, v in batch.get('mask', {}).items()} if 'mask' in batch else None
                neg_mask = {k: v.to(self.device).float() for k, v in batch.get('mask_neg', {}).items()} if 'mask_neg' in batch else None
            else:
                pos_dict = {k: v.to(self.device) for k, v in batch.items()}
                neg_dict = None
                pos_mask = None
                neg_mask = None
            
            aligned_pos = self.model(pos_dict)
            if pos_mask is not None:
                for m, x in aligned_pos.items():
                    if m in pos_mask:
                        aligned_pos[m] = x * pos_mask[m].unsqueeze(1)
            
            if self.lambda2 == 0:
                aligned_neg = None
            else:
                if neg_dict is None:
                    raise RuntimeError("Dataset must provide features_neg for negative samples.")
                aligned_neg = self.model(neg_dict)
                if aligned_neg is not None and neg_mask is not None:
                    for m, x in aligned_neg.items():
                        if m in neg_mask:
                            aligned_neg[m] = x * neg_mask[m].unsqueeze(1)
            
            loss, batch_svd = self._compute_loss_with_metrics(aligned_pos, aligned_negatives=aligned_neg)
            
            # 记录当前 batch 的 metrics
            batch_losses.append(loss.item())
            if batch_svd is not None:
                batch_svd_values.append(batch_svd)
            batch_indices.append(batch_idx)
        
        # 返回每个 batch 的原始 metrics
        return {
            'batch_losses': batch_losses,
            'batch_svd_values': batch_svd_values,
            'batch_indices': batch_indices,
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              max_steps: int = 10000,
              save_path: str = "best_model.pth",
              log_interval: int = 100,
              val_interval: int = 500) -> Dict[str, List[float]]:
        """
        训练模型（Step 模式 - 现代做法）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            max_steps: 最大训练步数（每个 batch 算一步）
            save_path: 最佳模型保存路径
            log_interval: 每隔多少步记录日志
            val_interval: 每隔多少步进行验证
            
        Returns:
            训练历史记录字典
            
        Example:
            trainer.train(train_loader, val_loader, max_steps=100000, log_interval=1000, val_interval=5000)
        """
        # 重置early stopping状态
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_triggered = False
        
        history = {
            'train_losses': [],        # 每个 step 的 loss
            'train_svd_values': [],    # 每个 step 的 SVD 值
            'val_losses': [],          # 验证的 loss（在 val_interval 时记录）
            'val_svd_values': [],      # 验证的 SVD 值
            'val_steps': [],           # 在哪些 step 进行了验证
        }
        
        logger.info("=" * 80)
        logger.info(f"🚀 开始训练 - 共 {max_steps} 步")
        logger.info(f"   - 日志间隔: 每 {log_interval} 步")
        logger.info(f"   - 验证间隔: 每 {val_interval} 步")
        logger.info("=" * 80)
        
        self.model.train()
        global_step = 0
        train_iter = iter(train_loader)
        
        progress_bar = tqdm(total=max_steps, desc="Training")
        
        while global_step < max_steps:
            # 获取下一个 batch
            batch = self._get_next_batch(train_iter, train_loader)
            
            # 执行训练步骤
            loss, svd_values = self._train_step(batch)
            
            # 📊 记录 metrics
            history['train_losses'].append(loss)
            if svd_values is not None:
                history['train_svd_values'].append(svd_values)
            
            global_step += 1
            
            # 更新进度条
            self._update_progress_bar(progress_bar, loss, svd_values)
            
            # 🔍 定期验证
            if val_loader is not None and global_step % val_interval == 0:
                val_loss, val_svd = self._run_validation(val_loader, self.val_max_batches)
                history['val_losses'].append(val_loss)
                history['val_steps'].append(global_step)
                if val_svd is not None:
                    history['val_svd_values'].append(val_svd)
                
                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self._save_checkpoint(save_path, global_step, val_loss)
                
                # 检查early stopping
                if self._check_early_stopping(val_loss):
                    logger.info(f"🛑 [Step {global_step}] Early stopping触发，训练提前结束")
                    break
            
            # 📝 定期日志输出
            if global_step % log_interval == 0:
                self._log_progress(global_step, max_steps, log_interval, history)
        
        progress_bar.close()
        
        # 训练结束后的总结
        if self.early_stopping_triggered:
            logger.info("=" * 80)
            logger.info("🛑 训练因Early Stopping提前结束")
            logger.info(f"   - 最佳验证损失: {self.best_val_loss:.4f}")
            logger.info(f"   - 总训练步数: {global_step}/{max_steps}")
            logger.info(f"   - 节省步数: {max_steps - global_step}")
            logger.info("=" * 80)
        else:
            logger.info("=" * 80)
            logger.info("✅ 训练正常完成！")
            logger.info(f"   - 最佳验证损失: {self.best_val_loss:.4f}")
            logger.info(f"   - 总训练步数: {global_step}/{max_steps}")
            logger.info("=" * 80)

        # 打印性能分析报告（仅在启用时）
        if self.verbose_timing:
            self.print_timing_analysis()
        
        return history