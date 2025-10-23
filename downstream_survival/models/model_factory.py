"""
模型工厂类
根据配置创建相应的模型实例
支持统一的模型接口和训练流程
"""

from .clam import CLAM
from .auc_clam import AUC_CLAM
from .mil import MIL_fc
from .clam_svd_loss import CLAM_SVD_LOSS
from .gate_shared_mil import GateSharedMIL
from .gate_mil import GateMIL
from .gate_auc_mil import GateAUCMIL
from .base_model import BaseModel
from typing import Dict, Any, Type

class ModelFactory:
    """
    模型工厂类
    
    支持创建统一接口的模型实例，便于统一训练流程
    所有模型都继承自BaseModel并实现统一的forward接口
    """
    
    # 模型类型映射
    MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
        'mil': MIL_fc,
        'clam': CLAM,
        'auc_clam': AUC_CLAM,
        'clam_svd_loss': CLAM_SVD_LOSS,
        'gate_shared_mil': GateSharedMIL,
        'gate_mil': GateMIL,
        'gate_auc_mil': GateAUCMIL
    }
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        根据配置创建模型实例
        
        Args:
            config: 模型配置字典，必须包含 'model_type' 字段
            
        Returns:
            BaseModel: 统一接口的模型实例
            
        Raises:
            ValueError: 配置错误时抛出
        """
        if 'model_type' not in config:
            raise ValueError("配置中缺少 'model_type' 参数")
        
        model_type = config['model_type']
        
        # 验证模型类型
        if model_type not in ModelFactory.MODEL_REGISTRY:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {list(ModelFactory.MODEL_REGISTRY.keys())}")
        
        model_class = ModelFactory.MODEL_REGISTRY[model_type]
        
        # 创建模型实例
        try:
            model = model_class(config)
            return model
        except Exception as e:
            raise ValueError(f"创建模型失败: {str(e)}")
    
    @staticmethod
    def create_model_with_validation(config: Dict[str, Any]) -> BaseModel:
        """
        创建模型并进行配置验证
        
        Args:
            config: 模型配置字典
            
        Returns:
            BaseModel: 验证通过的模型实例
        """
        # 验证配置
        ModelFactory.validate_model_config(config)
        
        # 创建模型
        model = ModelFactory.create_model(config)
        
        return model
    
    @staticmethod
    def get_supported_models() -> list:
        """
        获取支持的模型类型列表
        
        Returns:
            list: 支持的模型类型列表
        """
        return list(ModelFactory.MODEL_REGISTRY.keys())
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """
        验证模型配置的完整性和正确性
        
        Args:
            config: 模型配置字典
            
        Returns:
            bool: 验证通过返回True
            
        Raises:
            ValueError: 配置验证失败时抛出
        """
        if 'model_type' not in config:
            raise ValueError("配置中缺少 'model_type' 参数")
        
        model_type = config['model_type']
        if model_type not in ModelFactory.MODEL_REGISTRY:
            raise ValueError(f"不支持的模型类型: {model_type}，支持的类型: {ModelFactory.get_supported_models()}")
        
        # 根据模型类型验证特定参数
        required_params = ['n_classes', 'input_dim', 'dropout', 'base_loss_fn']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"模型配置缺少必需参数: {missing_params}")
        
        # 验证参数值的合理性
        if config['n_classes'] < 2:
            raise ValueError(f"类别数量必须 >= 2，当前: {config['n_classes']}")
        
        if config['input_dim'] <= 0:
            raise ValueError(f"输入维度必须 > 0，当前: {config['input_dim']}")
        
        if not 0 <= config['dropout'] <= 1:
            raise ValueError(f"dropout率必须在[0,1]范围内，当前: {config['dropout']}")

        return True
    
    @staticmethod
    def get_model_info(model: BaseModel) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model: 模型实例
            
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return model.get_model_info()
