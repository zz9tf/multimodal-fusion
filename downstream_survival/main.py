#!/usr/bin/env python3
"""
Multimodal Survival Status Prediction Main Program
Focused on WSI + TMA multimodal survival status prediction task
"""

from __future__ import print_function

import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import re
import pickle

# Add project path
import sys
# Add which folder your main.py is located as root_dir
root_dir = '/home/zheng/zheng/multimodal-fusion/downstream_survival'
sys.path.append(root_dir)

# Internal imports
from trainer import Trainer
from datasets.multimodal_dataset import MultimodalDataset

def save_pkl(filename, data):
    """Save pickle file"""
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pkl(filename):
    """Load pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def _get_model_specific_config(args):
    """Get specific configuration based on model type"""
    model_type = args.model_type
    
    mil_config = {
        'model_size': args.model_size,
        'return_features': args.return_features,
    }
    clam_config = {
        'gate': args.gate,
        'base_weight': args.base_weight,
        'inst_loss_fn': args.inst_loss_fn,
        'model_size': args.model_size,
        'subtyping': args.subtyping,
        'inst_number': args.inst_number,
        'return_features': args.return_features,
        'attention_only': args.attention_only
    }
    auc_config = {
        'auc_loss_weight': args.auc_loss_weight,
    }
    transfer_layer_config = {
        'output_dim': args.output_dim,
    }
    svd_config = {
        'enable_svd': args.enable_svd,
        'alignment_layer_num': args.alignment_layer_num,
        'lambda1': args.lambda1,
        'lambda2': args.lambda2,
        'tau1': args.tau1,
        'tau2': args.tau2,
        'return_svd_features': args.return_svd_features,
    }
    
    clip_config = {
        'alignment_layer_num': args.alignment_layer_num,
        'enable_clip': args.enable_clip,
        'clip_init_tau': args.clip_init_tau,
    }
    
    dynamic_gate_config = {
        'enable_dynamic_gate': args.enable_dynamic_gate,
        'confidence_weight': args.confidence_weight,
        'feature_weight_weight': args.feature_weight_weight,
    }
    
    random_loss_config = {
        'enable_random_loss': args.enable_random_loss,
        'weight_random_loss': args.weight_random_loss,
    }
    pooling_config = {
        'pooling_strategy': args.pooling_strategy,
    }
    
    # Parse fusion_blocks_sequence from JSON string
    if isinstance(args.fusion_blocks_sequence, str):
        try:
            fusion_blocks_sequence = json.loads(args.fusion_blocks_sequence)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse fusion_blocks_sequence as JSON: {e}")
    else:
        raise ValueError(f"fusion_blocks_sequence must be a JSON string, got {type(args.fusion_blocks_sequence)}")
    
    attention_block_config = {
        'fusion_blocks_sequence': fusion_blocks_sequence,
        'attention_num_heads': args.attention_num_heads,
    }
    
    if model_type == 'mil':
        return {
            **mil_config,
        }
    elif model_type == 'clam':
        return {
            **clam_config,
        }
    elif model_type == 'auc_clam':
        return {
            **clam_config,
            **auc_config,
        }
    elif model_type == 'clam_mlp':
        return {
            **clam_config,
            **transfer_layer_config
        }
    elif model_type == 'clam_mlp_detach':
        return {
            **clam_config,
            **transfer_layer_config,
        }
    elif model_type == 'svd_gate_random_clam':
        return {
            **clam_config,
            **transfer_layer_config,
            **svd_config,
            **dynamic_gate_config,
            **random_loss_config,
        }
    elif model_type == 'svd_gate_random_clam_detach':
        return {
            **clam_config,
            **transfer_layer_config,
            **svd_config,
            **dynamic_gate_config,
            **random_loss_config,
        }
    elif model_type == 'deep_supervise_svd_gate_random':
        return {
            **clam_config,
            **transfer_layer_config,
            **svd_config,
            **dynamic_gate_config,
            **random_loss_config,
        }
    elif model_type == 'deep_supervise_svd_gate_random_detach':
        return {
            **clam_config,
            **transfer_layer_config,
            **svd_config,
            **dynamic_gate_config,
            **random_loss_config,
        }
    elif model_type == 'clip_gate_random_clam':
        return {
            **clam_config,
            **transfer_layer_config,
            **clip_config,
            **dynamic_gate_config,
            **random_loss_config,
        }
    elif model_type == 'clip_gate_random_clam_detach':
        return {
            **clam_config,
            **transfer_layer_config,
            **clip_config,
            **dynamic_gate_config,
            **random_loss_config,
        }   
    elif model_type == 'gate_shared_mil':
        return {
            **mil_config,
            **dynamic_gate_config,
        }
    elif model_type == 'gate_mil':
        return {
            **mil_config,
            **dynamic_gate_config,
        }
    elif model_type == 'gate_auc_mil':
        return {
            **mil_config,
            **dynamic_gate_config,
            **auc_config,
        }
    elif model_type == 'gate_mil_detach':
        return {
            **mil_config,
            **dynamic_gate_config,
        }
    elif model_type == 'mdlm':
        return {
            **clam_config,
        }
    elif model_type == 'ps3':
        return {
            **clam_config,
        }
    elif model_type == 'fbp':
        return {
            **clam_config,
        }
    elif model_type == 'mfmf':
        return {
            **clam_config,
            **attention_block_config,
        }
    elif model_type == 'svd_pool':
        return {
            **clam_config,
            **svd_config,
            **pooling_config,
        }
    else:
        # Return empty configuration for other model types, can be extended as needed
        return {}

def _parse_aligned_channels(aligned_channels_list):
    """Parse aligned channel parameters"""
    if not aligned_channels_list:
        return {}
    
    align_channels = {}
    for item in aligned_channels_list:
        if '=' in item:
            key, value = item.split('=', 1)
            align_channels[key] = value
        else:
            # If no equals sign, assume key=value are the same
            align_channels[item] = item
    
    return align_channels

def seed_torch(seed=7):
    """Set random seed"""
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_dataset_split(dataset_split_path):
    """
    Load dataset split information from JSON file

    Args:
        dataset_split_path (str): Path to dataset split JSON file

    Returns:
        dict: Dictionary containing train/test splits, format {'train': [patient_ids], 'test': [patient_ids]}
    """
    if not os.path.exists(dataset_split_path):
        raise FileNotFoundError(f"Dataset split file does not exist: {dataset_split_path}")
    
    with open(dataset_split_path, 'r') as f:
        split_data = json.load(f)
    
    # Convert JSON data to train/test split
    train_patients = []
    test_patients = []
    
    for item in split_data:
        patient_id = item['patient_id']
        dataset_type = item['dataset']
        
        if dataset_type == 'training':
            train_patients.append(patient_id)
        elif dataset_type == 'test':
            test_patients.append(patient_id)
    
    return {
        'train': train_patients,
        'test': test_patients
    }

def create_k_fold_splits(dataset, k=10, seed=42, fixed_test_split=None):
    """
    Create k-fold cross-validation splits

    Args:
        dataset: Dataset object
        k (int): Number of folds
        seed (int): Random seed
        fixed_test_split (dict, optional): Fixed test split, format {'train': [patient_ids], 'test': [patient_ids]}

    Returns:
        list: List containing train/val/test indices for each fold
    """
    
    
    # Get labels and patient IDs for all samples (optimized: use dataset mappings directly to avoid reading HDF5)
    labels = []
    patient_ids = []

    # If dataset has case_ids and case_to_label mappings, use them directly (avoid reading HDF5 files)
    if hasattr(dataset, 'case_ids') and hasattr(dataset, 'case_to_label'):
        for i, case_id in enumerate(dataset.case_ids):
            label = dataset.case_to_label[case_id]
            labels.append(label)
            patient_ids.append(case_id)
    else:
        # Fallback: read samples one by one (slower, may get stuck)
        for i in range(len(dataset)):
            # Get sample data
            sample = dataset[i]

            # Get label from dataset
            if hasattr(dataset, 'get_label'):
                label = dataset.get_label(i)
            elif isinstance(sample, dict) and 'label' in sample:
                label = sample['label']
            else:
                # Assume tuple format (data, label)
                _, label = sample

            labels.append(label)

            # Get patient ID (assume dataset has get_patient_id method, or get from sample)
            if hasattr(dataset, 'get_patient_id'):
                patient_id = dataset.get_patient_id(i)
            elif isinstance(sample, dict) and 'patient_id' in sample:
                patient_id = sample['patient_id']
            else:
                # If no patient ID, use index as ID
                patient_id = str(i)
            patient_ids.append(patient_id)
    
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)
    
    splits = []
    
    if fixed_test_split is not None:
        # Build mapping from numeric ID to sample indices, compatible with different formats like "patient_002", "002", 2, etc.
        numeric_id_to_indices = {}
        for idx, pid in enumerate(patient_ids):
            num_id = _extract_numeric_id(pid)
            if num_id is None:
                continue
            numeric_id_to_indices.setdefault(num_id, []).append(idx)

        # Use fixed test split
        print(f"🔒 Using fixed test split")
        print(f"📊 Fixed training set patient count: {len(fixed_test_split['train'])}")
        print(f"📊 Fixed test set patient count: {len(fixed_test_split['test'])}")
        
        # Find indices corresponding to test set
        test_indices = []
        missing_test_ids = []
        for test_patient_id in fixed_test_split['test']:
            num_id = _extract_numeric_id(test_patient_id)
            cand = numeric_id_to_indices.get(num_id, []) if num_id is not None else []
            if len(cand) > 0:
                test_indices.extend(cand)
            else:
                missing_test_ids.append(test_patient_id)

        test_indices = np.array(test_indices, dtype=int)
        if len(missing_test_ids) > 0:
            print(f"⚠️ Fixed test set has {len(missing_test_ids)} IDs not found in dataset, for example: {missing_test_ids[:5]}")
        
        # Find indices corresponding to training set
        train_indices = []
        missing_train_ids = []
        for train_patient_id in fixed_test_split['train']:
            num_id = _extract_numeric_id(train_patient_id)
            cand = numeric_id_to_indices.get(num_id, []) if num_id is not None else []
            if len(cand) > 0:
                train_indices.extend(cand)
            else:
                missing_train_ids.append(train_patient_id)

        train_indices = np.array(train_indices, dtype=int)
        if len(missing_train_ids) > 0:
            print(f"⚠️ Fixed training set has {len(missing_train_ids)} IDs not found in dataset, for example: {missing_train_ids[:5]}")

        if train_indices.size == 0:
            available_sample = patient_ids[:5].tolist()
            raise ValueError(
                "Fixed training set split failed to match any entries with sample IDs in the dataset.\n"
                f"Please check if ID naming is consistent (case/prefix/suffix/type), or if data sources correspond.\n"
                f"Example - First 5 available IDs in dataset: {available_sample}\n"
                f"Example - First 5 unmatched IDs in fixed training set: {missing_train_ids[:5]}"
            )
        
        # Perform k-fold cross-validation on training set
        train_labels = labels[train_indices]
        
        # Create stratified k-fold split
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(skf.split(train_indices, train_labels)):
            # Convert to actual indices
            actual_train_idx = train_indices[fold_train_idx]
            actual_val_idx = train_indices[fold_val_idx]
            
            splits.append({
                'train': actual_train_idx,
                'val': actual_val_idx,
                'test': test_indices  # Test set always the same
            })
    else:
        # Original split method: further divide test set into validation and test sets
        print(f"🔄 Using traditional k-fold split")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        
        splits = []
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(range(len(dataset)), labels)):
            # Further divide test set into validation and test sets
            # 🔧 Ensure test_idx is sorted consistently to avoid different results each time
            test_idx_sorted = np.sort(test_idx)
            test_labels = labels[test_idx_sorted]
            val_test_skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
            val_idx, test_idx_final = next(val_test_skf.split(test_idx_sorted, test_labels))
            
            # Convert to actual indices
            val_idx = test_idx_sorted[val_idx]
            test_idx_final = test_idx_sorted[test_idx_final]
            
            splits.append({
                'train': train_idx,
                'val': val_idx, 
                'test': test_idx_final
            })
    
    return splits

def _extract_numeric_id(id_value):
    """Convert different forms of patient ID to numeric ID for robust matching."""
    try:
        if isinstance(id_value, (int, np.integer)):
            return int(id_value)
        if id_value is None:
            return None
        s = str(id_value)
        m = re.findall(r"\d+", s)
        if not m:
            return None
        return int(m[-1])
    except Exception:
        return None

def parse_channels(channels):
    """
    Parse channels list, mapping simplified channel names to complete HDF5 paths

    Supported channel types:
    - WSI: 'wsi' -> 'wsi=features', 'wsi=reconstructed_features'
    - TMA Features: 'tma', 'cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1'
    - TMA Patches: 'tma_patches', 'cd163_patches', 'cd3_patches', etc.
    - Clinical: 'clinical', 'clinical_ori', 'clinical_mask', 'clinical_ori_mask'
    - Pathological: 'pathological', 'pathological_ori', 'pathological_mask', 'pathological_ori_mask'
    - Blood: 'blood', 'blood_ori', 'blood_mask', 'blood_ori_mask'
    - ICD: 'icd', 'icd_ori', 'icd_mask', 'icd_ori_mask'
    - TMA Cell Density: 'tma_cell_density', 'tma_cell_density_ori', 'tma_cell_density_mask', 'tma_cell_density_ori_mask'

    Args:
        channels (List[str]): List of channel names

    Returns:
        List[str]: List of parsed complete channel paths

    Raises:
        ValueError: When input channel names are invalid
    """
    if not channels:
        return []
    
    # TMA channel definitions
    TMA_CHANNELS = ['cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1']

    # Supported channel type mappings
    CHANNEL_MAPPINGS = {
        # WSI channels
        'wsi': ['wsi=features', 'wsi=reconstructed_features'],

        # TMA Features channels
        'tma': [f'tma={channel}=features' for channel in TMA_CHANNELS],

        # TMA Patches channels
        'tma_patches': [f'tma={channel}=patches' for channel in TMA_CHANNELS],

        # Clinical channels
        'clinical': ['clinical=val'],
        'clinical_ori': ['clinical=ori_val'],
        'clinical_mask': ['clinical=val', 'clinical=mask'],
        'clinical_ori_mask': ['clinical=ori_val', 'clinical=mask'],

        # Pathological channels
        'pathological': ['pathological=val'],
        'pathological_ori': ['pathological=ori_val'],
        'pathological_mask': ['pathological=val', 'pathological=mask'],
        'pathological_ori_mask': ['pathological=ori_val', 'pathological=mask'],

        # Blood channels
        'blood': ['blood=val'],
        'blood_ori': ['blood=ori_val'],
        'blood_mask': ['blood=val', 'blood=mask'],
        'blood_ori_mask': ['blood=ori_val', 'blood=mask'],

        # ICD channels
        'icd': ['icd=val'],
        'icd_ori': ['icd=ori_val'],
        'icd_mask': ['icd=val', 'icd=mask'],
        'icd_ori_mask': ['icd=ori_val', 'icd=mask'],

        # TMA Cell Density channels
        'tma_cell_density': ['tma_cell_density=val'],
        'tma_cell_density_ori': ['tma_cell_density=ori_val'],
        'tma_cell_density_mask': ['tma_cell_density=val', 'tma_cell_density=mask'],
        'tma_cell_density_ori_mask': ['tma_cell_density=ori_val', 'tma_cell_density=mask'],
    }
    
    # Add mappings for individual TMA channels
    for channel in TMA_CHANNELS:
        CHANNEL_MAPPINGS[channel] = [f'tma={channel}=features']
        CHANNEL_MAPPINGS[f'{channel}_patches'] = [f'tma={channel}=patches']
    
    parsed_channels = []
    invalid_channels = []
    
    for channel in channels:
        if channel in CHANNEL_MAPPINGS:
            parsed_channels.extend(CHANNEL_MAPPINGS[channel])
        elif '=' in channel:  # Already in complete path format
            parsed_channels.append(channel)
        else:
            invalid_channels.append(channel)
    
    # Validate invalid channels
    if invalid_channels:
        available_channels = list(CHANNEL_MAPPINGS.keys())
        raise ValueError(
            f"❌ Invalid channel names: {invalid_channels}\n"
            f"📋 Supported channel types: {available_channels}\n"
            f"💡 Tip: Channel names are case-insensitive, support single channels or combinations"
        )
    
    return parsed_channels

def get_available_channels():
    """
    Get list of all available channel types

    Returns:
        Dict[str, List[str]]: Dictionary of available channels grouped by category
    """
    TMA_CHANNELS = ['cd163', 'cd3', 'cd56', 'cd68', 'cd8', 'he', 'mhc1', 'pdl1']
    
    return {
        'WSI channels': ['wsi'],
        'TMA Features channels': ['tma'] + TMA_CHANNELS,
        'TMA Patches channels': ['tma_patches'] + [f'{ch}_patches' for ch in TMA_CHANNELS],
        'Clinical channels': ['clinical', 'clinical_ori', 'clinical_mask', 'clinical_ori_mask'],
        'Pathological channels': ['pathological', 'pathological_ori', 'pathological_mask', 'pathological_ori_mask'],
        'Blood channels': ['blood', 'blood_ori', 'blood_mask', 'blood_ori_mask'],
        'ICD channels': ['icd', 'icd_ori', 'icd_mask', 'icd_ori_mask'],
        'TMA Cell Density channels': ['tma_cell_density', 'tma_cell_density_ori', 'tma_cell_density_mask', 'tma_cell_density_ori_mask']
    }

def print_available_channels():
    """
    Print all available channel types for debugging and help
    """
    channels = get_available_channels()
    print("🔍 Available channel types:")
    print("=" * 50)
    
    for category, channel_list in channels.items():
        print(f"\n📁 {category}:")
        for channel in channel_list:
            print(f"  • {channel}")
    
    print("\n💡 Usage examples:")
    print("  • Single channels: ['wsi', 'clinical']")
    print("  • Combined channels: ['tma', 'blood_mask']")
    print("  • Complete paths: ['wsi=features', 'clinical=val']")

def main(args, configs):
    """Main function"""
    # Get parameters from configuration
    experiment_config = configs['experiment_config']
    
    # Load dataset
    print('\nLoad Dataset')
    if not experiment_config['data_root_dir']:
        raise ValueError('data_root_dir is required')
    if not os.path.exists(experiment_config['data_root_dir']):
        raise ValueError('data_root_dir does not exist')
    
    print('data_root_dir: ', os.path.abspath(experiment_config['data_root_dir']))

    # Create multimodal dataset
    print(f"Target channels: {experiment_config['target_channels']}")
    
    # Build channels list
    channels = args.target_channels
    
    # Test parse_channels function
    try:
        parsed_channels = parse_channels(channels)
        print(f"✅ Successfully parsed channels: {len(parsed_channels)} channels")
        print(f"📋 Original channels: {channels}")
        print(f"🔗 Parsed channels: {parsed_channels}")
    except ValueError as e:
        print(f"❌ Channel parsing error: {e}")
        print_available_channels()
        return
    
    # Build align_channels mapping
    align_channels = _parse_aligned_channels(args.aligned_channels)
    
    print(f"Channels: {channels}")
    print(f"Align channels: {align_channels}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultimodalDataset(
        csv_path=experiment_config['csv_path'],
        data_root_dir=experiment_config['data_root_dir'],
        channels=channels,
        align_channels=align_channels,
        alignment_model_path=experiment_config['alignment_model_path'],
        device=device,
        print_info=True,
        preload_all=False,
    )
    
    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # Create k-fold splits
    print(f'\nCreating {args.k}-fold cross-validation splits...')
    print(f"🔧 Split mode: {args.split_mode}")
    
    # Check if using fixed test split
    fixed_test_split = None
    if args.split_mode == 'fixed':
        if not args.dataset_split_path:
            raise ValueError("❌ When using fixed test split mode, --dataset_split_path parameter must be provided")
        print(f"📁 Loading fixed test split: {args.dataset_split_path}")
        fixed_test_split = load_dataset_split(args.dataset_split_path)
        print(f"✅ Successfully loaded fixed test split")
    elif args.split_mode == 'random':
        print(f"🎲 Using random split mode")
    else:
        raise ValueError(f"❌ Unsupported split mode: {args.split_mode}")
    
    splits = create_k_fold_splits(dataset, k=args.k, seed=args.seed, fixed_test_split=fixed_test_split)
    print(f'✅ Created {len(splits)} folds')

    # Determine fold range (support starting from start_k_fold)
    start = int(args.start_k_fold) if hasattr(args, 'start_k_fold') and args.start_k_fold is not None else 0
    end = int(args.k)
    if start < 0 or start >= end:
        raise ValueError(f"❌ start_k_fold out of bounds: start={start}, k={end}. Allowed range: 0 <= start < k")
    print(f"\n➡️ Will run from fold {start} to fold {end-1}")

    # Initialize trainer
    trainer = Trainer(
        configs=configs,
        log_dir=os.path.join(args.results_dir, 'training_logs')
    )

    # Store results
    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    
    # Train each fold
    for i in folds:
        print(f'\n{"="*60}')
        print(f'Training Fold {i+1}/{args.k}')
        print(f'{"="*60}')
        
        # Get current fold split
        split = splits[i]
        train_idx = split['train']
        val_idx = split['val']
        test_idx = split['test']
        
        print(f'Train samples: {len(train_idx)}')
        print(f'Val samples: {len(val_idx)}')
        print(f'Test samples: {len(test_idx)}')
        
        # Create subset datasets
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        
        datasets = (train_dataset, val_dataset, test_dataset)
        
        # Train using trainer
        results, test_auc, val_auc, test_acc, val_acc = trainer.train_fold(
            datasets=datasets,
            fold_idx=i
        )
        
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        
        # Save results
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)
        
        print(f'Fold {i+1} completed - Test AUC: {test_auc:.4f}, Val AUC: {val_auc:.4f}')

    # Save final results
    final_df = pd.DataFrame({
        'folds': folds, 
        'test_auc': all_test_auc, 
        'val_auc': all_val_auc, 
        'test_acc': all_test_acc, 
        'val_acc': all_val_acc
    })

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))
    
    # Save detailed training data for subsequent plotting
    detailed_results = {
        'configurations': configs,
        'fold_results': {
            'folds': folds.tolist(),
            'test_auc': all_test_auc,
            'val_auc': all_val_auc, 
            'test_acc': all_test_acc,
            'val_acc': all_val_acc
        },
        'summary_stats': {
            'mean_test_auc': float(np.mean(all_test_auc)),
            'std_test_auc': float(np.std(all_test_auc)),
            'mean_val_auc': float(np.mean(all_val_auc)),
            'std_val_auc': float(np.std(all_val_auc)),
            'mean_test_acc': float(np.mean(all_test_acc)),
            'std_test_acc': float(np.std(all_test_acc)),
            'mean_val_acc': float(np.mean(all_val_acc)),
            'std_val_acc': float(np.std(all_val_acc))
        }
    }
    
    # Save detailed results for plotting
    detailed_save_name = 'detailed_results_for_plotting.json'
    with open(os.path.join(args.results_dir, detailed_save_name), 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print final statistics
    print(f'\n{"="*60}')
    print('FINAL RESULTS SUMMARY')
    print(f'{"="*60}')
    print(f'Mean Test AUC: {np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}')
    print(f'Mean Val AUC: {np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}')
    print(f'Mean Test Acc: {np.mean(all_test_acc):.4f} ± {np.std(all_test_acc):.4f}')
    print(f'Mean Val Acc: {np.mean(all_val_acc):.4f} ± {np.std(all_val_acc):.4f}')
    print(f'Results saved to: {os.path.join(args.results_dir, save_name)}')
    print(f'Detailed results for plotting: {os.path.join(args.results_dir, detailed_save_name)}')


if __name__ == "__main__":
    # Parameter parsing
    parser = argparse.ArgumentParser(description='Multimodal survival status prediction configuration')

    # Data-related parameters
    parser.add_argument('--data_root_dir', type=str, default=None,
                        help='Data root directory')
    parser.add_argument('--results_dir', default='./results',
                        help='Results save directory (default: ./results)')
    parser.add_argument('--csv_path', type=str, default='dataset_csv/survival_status_labels.csv',
                        help='CSV file path')
    # Alignment model related parameters
    parser.add_argument('--alignment_model_path', type=str, default=None,
                        help='Pre-trained alignment model path (providing this parameter will automatically enable alignment functionality)')
    # Multimodal related parameters
    parser.add_argument('--target_channels', type=str, nargs='+',
                        default=['CD3', 'CD8', 'CD56', 'CD68', 'CD163', 'HE', 'MHC1', 'PDL1'],
                        help='Target channels')
    parser.add_argument('--aligned_channels', type=str, nargs='*',
                        default=None,
                        help='Alignment targets, format: channel_to_align1=align_channel_name1 channel_to_align2=align_channel_name2 ...')
    # Experiment related parameters
    parser.add_argument('--exp_code', type=str,
                        help='Experiment code for saving results')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--start_k_fold', type=int, default=0,
                        help='Starting fold number (default: 0)')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of folds (default: 10)')
    parser.add_argument('--split_mode', type=str, choices=['random', 'fixed'], default='random',
                        help='Dataset split mode: random=random split(80% train, 10% val, 10% test), fixed=fixed test split (default: random)')
    parser.add_argument('--dataset_split_path', type=str, default=None,
                        help='Fixed test split JSON file path (only used when split_mode=fixed)')
    parser.add_argument('--max_epochs', type=int, default=200,
                        help='Maximum training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--reg', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--opt', type=str, choices=['adam', 'sgd'], default='adam',
                        help='Optimizer type')
    parser.add_argument('--early_stopping', action='store_true', default=False,
                        help='Enable early stopping')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--lr_scheduler', type=str,
                        choices=['none', 'cosine', 'cosine_warm_restart', 'step', 'plateau', 'exponential'],
                        default='none',
                        help='Learning rate scheduler type (default: none)')
    parser.add_argument('--lr_scheduler_params', type=str, default='{}',
                        help='Learning rate scheduler parameters (JSON string, default: {})')

    # Model related parameters
    parser.add_argument('--model_type', type=str, choices=[
        'mil', 'clam', 'auc_clam', 'clam_mlp', 'clam_mlp_detach', 'svd_gate_random_clam', 'svd_gate_random_clam_detach', 
        'gate_shared_mil', 'gate_mil_detach', 'gate_mil', 'gate_auc_mil', 'clip_gate_random_clam', 'clip_gate_random_clam_detach',
        'deep_supervise_svd_gate_random', 'deep_supervise_svd_gate_random_detach', 'svd_pool',
        'mdlm', 'ps3', 'fbp', 'mfmf'
        ], 
                        default='clam', help='Model type (default: clam)')
    parser.add_argument('--input_dim', type=int, default=1024,
                        help='Input dimension')
    parser.add_argument('--dropout', type=float, default=0.25, 
                        help='Dropout rate')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--base_loss_fn', type=str, choices=['svm', 'ce'], default='ce',
                        help='Slide-level classification loss function (default: ce)')

    # CLAM related parameters
    parser.add_argument('--gate', action='store_true', default=True, 
                        help='CLAM: Use gated attention mechanism')
    parser.add_argument('--base_weight', type=float, default=0.7,
                        help='CLAM: bag-level loss weight coefficient (default: 0.7)')
    parser.add_argument('--inst_loss_fn', type=str, choices=['svm', 'ce', None], default=None,
                        help='CLAM: instance-level clustering loss function (default: None)')
    parser.add_argument('--model_size', type=str, 
                        choices=['small', 'big', '128*64', '64*32', '32*16', '16*8', '8*4', '4*2', '2*1'], 
                        default='small', help='Model size')
    parser.add_argument('--subtyping', action='store_true', default=False, 
                        help='Subtyping problem')
    parser.add_argument('--inst_number', type=int, default=8, 
                        help='CLAM: positive and negative sample sampling count')
    parser.add_argument('--channels_used_in_model', type=str, nargs='+', 
                        default=['wsi', 'tma', 'clinical', 'pathological', 'blood', 'icd', 'tma_cell_density'],
                        help='Channels to be used in the model')
    parser.add_argument('--return_features', action='store_true', default=False, 
                        help='MIL & CLAM: return features')
    parser.add_argument('--attention_only', action='store_true', default=False, 
                        help='CLAM: return only attention')

    # Transfer layer
    parser.add_argument('--output_dim', type=int, default=128, 
                        help='Transfer layer: modality-unified output dimension')

    # SVD related parameters
    parser.add_argument('--enable_svd', action='store_true', default=False, 
                        help='SVD: Enable SVD')
    parser.add_argument('--alignment_layer_num', type=int, default=2,
                        help='SVD: alignment layer count')
    parser.add_argument('--lambda1', type=float, default=1.0,
                        help='SVD: alignment loss weight')
    parser.add_argument('--lambda2', type=float, default=0.0,
                        help='SVD: alignment loss weight')
    parser.add_argument('--tau1', type=float, default=0.1,
                        help='SVD: alignment loss weight')
    parser.add_argument('--tau2', type=float, default=0.05,
                        help='SVD: alignment loss weight')
    parser.add_argument('--return_svd_features', action='store_true', default=False, 
                        help='SVD: return SVD features')

    # CLIP related parameters
    parser.add_argument('--enable_clip', action='store_true', default=False, 
                        help='CLIP: Enable CLIP')
    parser.add_argument('--clip_init_tau', type=float, default=0.07,
                        help='CLIP: initial tau')

    # Dynamic Gate related parameters
    parser.add_argument('--enable_dynamic_gate', action='store_true', default=False, 
                        help='Dynamic Gate: Enable dynamic gating')
    parser.add_argument('--confidence_weight', type=float, default=1.0,
                        help='Dynamic Gate: confidence weight')
    parser.add_argument('--feature_weight_weight', type=float, default=1.0,
                        help='Dynamic Gate: feature weight weight')

    # AUC related parameters
    parser.add_argument('--auc_loss_weight', type=float, default=1.0,
                        help='AUC: AUC loss weight')

    # Random Loss related parameters
    parser.add_argument('--enable_random_loss', action='store_true', default=False, 
                        help='Random Loss: Enable random loss')
    parser.add_argument('--weight_random_loss', type=float, default=0.1, 
                        help='Random Loss: random loss weight')
    
    # Attention related parameters
    parser.add_argument('--attention_num_heads', type=int, default=8,
                        help='Attention: number of attention heads')
    parser.add_argument('--fusion_blocks_sequence', type=str,
                        default='[{"q": "other", "kv": "tma"}, {"q": "result", "kv": "wsi"}, {"q": "reconstruct", "kv": "result"}]',
                        help='Attention: fusion block sequence (JSON string)')
    
    # Pooling related parameters
    parser.add_argument('--pooling_strategy', type=str, choices=['mean', 'max', 'sum'], default='mean',
                        help='Pooling strategy: mean=mean pooling, max=max pooling, sum=sum pooling')
    # Parse arguments
    args = parser.parse_args()
    args.target_channels = parse_channels(args.target_channels)
    args.aligned_channels = parse_channels(args.aligned_channels)
    args.channels_used_in_model = parse_channels(args.channels_used_in_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set random seed
    seed_torch(args.seed)

    # Create results directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # Create timestamped results directory
    args.results_dir = os.path.join(
        args.results_dir, 
        datetime.now().strftime("%Y%m%d-%H%M%S") + '_' + str(args.exp_code) + '_s{}'.format(args.seed)
    )
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    # Create concise classification configuration dictionary
    configs = {
        'experiment_config': {
            'data_root_dir': args.data_root_dir,
            'results_dir': args.results_dir,
            'csv_path': args.csv_path,
            'alignment_model_path': args.alignment_model_path,
            'target_channels': args.target_channels,
            'aligned_channels': args.aligned_channels,
            'exp_code': args.exp_code,
            'seed': args.seed,
            'num_splits': args.k,
            'split_mode': args.split_mode,
            'dataset_split_path': args.dataset_split_path,
            'max_epochs': args.max_epochs,
            'lr': args.lr,
            'reg': args.reg,
            'opt': args.opt,
            'early_stopping': args.early_stopping,
            'batch_size': args.batch_size,
            'scheduler_config': {
                'type': args.lr_scheduler if args.lr_scheduler != 'none' else None,
                **(json.loads(args.lr_scheduler_params) if args.lr_scheduler_params else {})
            }
        },
        
        'model_config': {
            'model_type': args.model_type,
            'input_dim': args.input_dim,
            'dropout': args.dropout,
            'n_classes': args.n_classes,
            'base_loss_fn': args.base_loss_fn,
            'channels_used_in_model': args.channels_used_in_model,
            **_get_model_specific_config(args)
        }
    }

    # Save classification configuration
    with open(args.results_dir + '/configs_{}.json'.format(args.exp_code), 'w') as f:
        json.dump(configs, f, indent=2)

    # Print concise configuration
    print("################# Configuration ###################")
    print(f"\n📋 EXPERIMENT CONFIG:")
    for key, val in configs['experiment_config'].items():
        print(f"  {key}: {val}")

    print(f"\n📋 MODEL CONFIG:")
    for key, val in configs['model_config'].items():
        print(f"  {key}: {val}")
    results = main(args, configs)
    print("finished!")
    print("end script")