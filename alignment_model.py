"""
多模态对齐模块
直接读取已编码的特征数据，进行线性变换和对齐训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalAlignmentModel(nn.Module):
    """
    多模态对齐模型 - 直接对已编码特征进行线性变换和对齐
    """
    
    def __init__(self, 
                 modality_names: List[str],
                 feature_dim: int = 1024,
                 num_layers: int = 1):
        """
        初始化多模态对齐模型
        
        Args:
            modality_names: 模态名称列表，如 ['CD3', 'CD8', 'CD28']
            feature_dim: 特征维度（所有模态统一维度）
            num_layers: 对齐层的层数，默认为 1
        """
        super(MultiModalAlignmentModel, self).__init__()
        
        self.modality_names = modality_names
        self.num_modalities = len(modality_names)
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # 为每个模态构建对齐层
        self.alignment_layers = nn.ModuleDict()
        # MLP predictor 接收拼接后的特征 (num_modalities * feature_dim)
        self.mlp_predictor = MLPMatchPredictor(feature_dim=self.num_modalities * feature_dim, hidden_dim=512)
        
        for modality_name in modality_names:
            self.alignment_layers[modality_name] = self._build_alignment_layer(
                feature_dim=feature_dim,
                num_layers=num_layers
            )
        
        logger.info(f"✅ 多模态对齐模型初始化完成")
        logger.info(f"   - 模态数量: {self.num_modalities}")
        logger.info(f"   - 模态名称: {modality_names}")
        logger.info(f"   - 特征维度: {feature_dim}")
        logger.info(f"   - 对齐层数: {num_layers}")
    
    def _build_alignment_layer(self, 
                               feature_dim: int, 
                               num_layers: int = 1) -> nn.Module:
        """
        构建对齐层 - 支持多层网络结构（纯线性层叠加）
        
        Args:
            feature_dim: 输入和输出特征维度
            num_layers: 层数
            
        Returns:
            nn.Sequential: 多层线性网络模块
        """
        layers = []
        
        for i in range(num_layers):
            layers.append(nn.Linear(feature_dim, feature_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, modality_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 高效并行对已编码特征进行线性对齐变换
        
        Args:
            modality_data: 字典，键为模态名称，值为已编码的特征
                          例如: {'CD3': CD3_features, 'CD8': CD8_features, 'CD28': CD28_features}
            
        Returns:
            对齐后的特征字典，键为模态名称，值为对齐后的特征
        """
        aligned_features = {}
        
        # 验证所有模态都在模型中
        for modality_name in modality_data.keys():
            if modality_name not in self.modality_names:
                raise ValueError(f"未知的模态: {modality_name}")
        
        for modality_name, features in modality_data.items():
            aligned_features[modality_name] = self.alignment_layers[modality_name](features)
        
        return aligned_features


class MLPMatchPredictor(nn.Module):
    """
    MLP匹配预测器 - 预测多模态特征是否匹配
    两层MLP网络，用于判断特征是否来自同一个样本
    """
    
    def __init__(self, 
                 feature_dim: int = 1024,
                 hidden_dim: int = 512,
                 dropout: float = 0.1):
        super(MLPMatchPredictor, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 两层MLP网络
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 输出0-1之间的概率
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)
    

def main():
    """
    主函数 - 示例用法和模型测试
    """
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建多模态对齐模型
    model = MultiModalAlignmentModel(
        modality_names=['CD3', 'CD8', 'CD28'],
        feature_dim=1024
    )
    
    logger.info("✅ 多模态对齐模块初始化完成")
    
    # 🧪 测试MLP匹配预测器
    logger.info("🧪 测试MLP匹配预测器...")
    
    # 模拟数据
    batch_size = 32
    feature_dim = 1024
    
    # 1. 测试单模态MLP预测器
    mlp_predictor = MLPMatchPredictor(feature_dim=feature_dim, hidden_dim=512)
    test_features = torch.randn(batch_size, feature_dim, device=device)
    predictions = mlp_predictor(test_features)
    logger.info(f"   📊 MLP预测器输出形状: {predictions.shape}")
    logger.info(f"   📊 预测概率范围: [{predictions.min().item():.3f}, {predictions.max().item():.3f}]")
    
    # 2. 测试多模态MLP预测器
    multi_modal_predictor = MultiModalMatchPredictor(
        modality_names=['CD3', 'CD8', 'CD28'],
        feature_dim=feature_dim,
        fusion_method="concat"
    )
    
    test_modality_data = {
        'CD3': torch.randn(batch_size, feature_dim, device=device),
        'CD8': torch.randn(batch_size, feature_dim, device=device),
        'CD28': torch.randn(batch_size, feature_dim, device=device)
    }
    
    multi_predictions = multi_modal_predictor(test_modality_data)
    logger.info(f"   📊 多模态MLP预测器输出形状: {multi_predictions.shape}")
    logger.info(f"   📊 多模态预测概率范围: [{multi_predictions.min().item():.3f}, {multi_predictions.max().item():.3f}]")
    
    # 3. 测试二分类预测
    binary_predictions = multi_modal_predictor.predict_match(test_modality_data, threshold=0.5)
    logger.info(f"   📊 二分类预测形状: {binary_predictions.shape}")
    logger.info(f"   📊 匹配样本数量: {binary_predictions.sum().item()}/{batch_size}")
    
    logger.info("✅ 所有模型测试完成")
    logger.info("💡 使用示例:")
    logger.info("   from Finetune_trainer import MultiModalAlignmentTrainer")
    logger.info("   trainer = MultiModalAlignmentTrainer(model, device)")
    logger.info("   history = trainer.train(train_loader, val_loader)")


if __name__ == "__main__":
    main()
