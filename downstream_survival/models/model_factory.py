"""
Model factory class
Creates corresponding model instances based on configuration
Supports unified model interface and training process
"""

from .clam import CLAM
from .auc_clam import AUC_CLAM
from .mil import MIL_fc
from .clam_mlp_detach import ClamMLPDetach
from .clam_mlp import ClamMLP
from .svd_gate_random_clam import SVDGateRandomClam
from .svd_gate_random_clam_detach import SVDGateRandomClamDetach
from .gate_shared_mil import GateSharedMIL
from .gate_mil import GateMIL
from .gate_auc_mil import GateAUCMIL
from .gate_mil_detach import GateMILDetach
from .clip_gate_random_clam import ClipGateRandomClam
from .clip_gate_random_clam_detach import ClipGateRandomClamDetach
from .deep_supervise_svd_gate_random import DeepSuperviseSVDGateRandomClam
from .deep_supervise_svd_gate_random_detach import DeepSuperviseSVDGateRandomClamDetach
from .svd_pool import SVDPool
from .mdlm import MDLM
from .ps3 import PS3
from .fbp import FBP
from .mfmf import MFMF
from .base_model import BaseModel
from typing import Dict, Any, Type

class ModelFactory:
    """
    Model factory class

    Supports creating model instances with unified interface for unified training process
    All models inherit from BaseModel and implement unified forward interface
    """
    
    # Model type mapping
    MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
        'mil': MIL_fc,
        'clam': CLAM,
        'auc_clam': AUC_CLAM,
        'clam_mlp': ClamMLP,
        'clam_mlp_detach': ClamMLPDetach,
        'svd_gate_random_clam': SVDGateRandomClam,
        'svd_gate_random_clam_detach': SVDGateRandomClamDetach,
        'clip_gate_random_clam': ClipGateRandomClam,
        'clip_gate_random_clam_detach': ClipGateRandomClamDetach,
        'gate_shared_mil': GateSharedMIL,
        'gate_mil': GateMIL,
        'gate_auc_mil': GateAUCMIL,
        'gate_mil_detach': GateMILDetach,
        'deep_supervise_svd_gate_random': DeepSuperviseSVDGateRandomClam,
        'deep_supervise_svd_gate_random_detach': DeepSuperviseSVDGateRandomClamDetach,
        'svd_pool': SVDPool,
        'mdlm': MDLM,
        'ps3': PS3,
        'fbp': FBP,
        'mfmf': MFMF
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
        
        # Validate specific parameters based on model type
        required_params = ['n_classes', 'input_dim', 'dropout', 'base_loss_fn']
        missing_params = [param for param in required_params if param not in config]
        if missing_params:
            raise ValueError(f"Model configuration missing required parameters: {missing_params}")
        
        # Validate parameter value rationality
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
            model: Model instance
            
        Returns:
            Dict[str, Any]: Model information dictionary
        """
        return model.get_model_info()
