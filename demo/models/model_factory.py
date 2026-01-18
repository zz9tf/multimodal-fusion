"""
Model factory class for SVD + Drop Modality Demo
Creates SVD-based multimodal fusion models

This demo version includes only the core SVD models for demonstration.
"""

from .clam_mlp import ClamMLP
from .svd_gate_random_clam import SVDGateRandomClam
from .deep_supervise_svd_gate_random import DeepSuperviseSVDGateRandomClam
from .base_model import BaseModel
from typing import Dict, Any, Type

class ModelFactory:
    """
    Model factory class

    Supports creating model instances with unified interface for unified training process
    All models inherit from BaseModel and implement unified forward interface
    """
    
    # Model type mapping - Demo version with core SVD models
    MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
        'clam_mlp': ClamMLP,
        'svd_gate_random_clam': SVDGateRandomClam,
        'deep_supervise_svd_gate_random': DeepSuperviseSVDGateRandomClam,
    }
    
    @staticmethod
    def create_model(config: Dict[str, Any]) -> BaseModel:
        """
        Create model instance based on configuration

        Args:
            config: Model configuration dictionary, must contain 'model_type' field

        Returns:
            BaseModel: Model instance with unified interface

        Raises:
            ValueError: Thrown when configuration error occurs
        """
        if 'model_type' not in config:
            raise ValueError("Missing 'model_type' parameter in configuration")

        model_type = config['model_type']

        # Validate model type
        if model_type not in ModelFactory.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}, supported types: {sorted(ModelFactory.MODEL_REGISTRY.keys())}")

        model_class = ModelFactory.MODEL_REGISTRY[model_type]

        # Create model instance
        try:
            model = model_class(config)
            return model
        except Exception as e:
            raise ValueError(f"Failed to create model: {str(e)}")
    
    @staticmethod
    def create_model_with_validation(config: Dict[str, Any]) -> BaseModel:
        """
        Create model and perform configuration validation

        Args:
            config: Model configuration dictionary

        Returns:
            BaseModel: Model instance that passed validation
        """
        # Validate configuration
        ModelFactory.validate_model_config(config)

        # Create model
        model = ModelFactory.create_model(config)
        
        return model
    
    @staticmethod
    def get_supported_models() -> list:
        """
        Get list of supported model types

        Returns:
            list: List of supported model types (sorted alphabetically)
        """
        return sorted(ModelFactory.MODEL_REGISTRY.keys())
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> bool:
        """
        Validate model configuration completeness and correctness

        Args:
            config: Model configuration dictionary

        Returns:
            bool: Returns True if validation passes

        Raises:
            ValueError: Thrown when configuration validation fails
        """
        if 'model_type' not in config:
            raise ValueError("Missing 'model_type' parameter in configuration")
        
        model_type = config['model_type']
        if model_type not in ModelFactory.MODEL_REGISTRY:
            raise ValueError(f"Unsupported model type: {model_type}, supported types: {ModelFactory.get_supported_models()}")
        
        # 根据模型类型验证特定参数
        required_params = ['n_classes', 'input_dim', 'dropout', 'base_loss_fn']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Model configuration missing required parameters: {missing_params}")
        
        # 验证参数值的合理性
        if config['n_classes'] < 2:
            raise ValueError(f"Number of classes must be >= 2, current: {config['n_classes']}")
        
        if config['input_dim'] <= 0:
            raise ValueError(f"Input dimension must be > 0, current: {config['input_dim']}")
        
        if not 0 <= config['dropout'] <= 1:
            raise ValueError(f"Dropout rate must be in [0,1] range, current: {config['dropout']}")

        return True
    
    @staticmethod
    def get_model_info(model: BaseModel) -> Dict[str, Any]:
        """
        Get model information
        
        Args:
            model: 模型实例
            
        Returns:
            Dict[str, Any]: 模型信息字典
        """
        return model.get_model_info()
