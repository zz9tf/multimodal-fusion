"""
TMA 对齐加载模块
仅保留用于对齐加载 tma_uni_tile_1024_{marker}.npz 的 Dataset
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Union, Tuple
import logging
import re
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class TMANpzAlignedDataset(Dataset):
    """
    TMA NPZ 多 marker 对齐数据集

    该数据集用于按规范化键 (block, x, y, patient) 对齐多个
    tma_uni_tile_1024_{marker}.npz 文件中的 1024 维向量。默认仅保留“所有 marker 都存在”的交集，自动跳过缺失组。

    属性：
        modality_to_path: modality 名到 NPZ 路径的映射
        modality_to_npz:  已打开的 np.load 句柄（mmap 只读）
        normalized_keys: 用于索引的规范化键列表（仅交集）
        normalized_to_raw_key: {modality: {normalized_tuple: raw_key}}
        modality_names: 参与对齐的 modality 列表
    """

    def __init__(self,
                 base_dir: str,
                 modality_names: List[str],
                 filename_template: str = 'tma_uni_tile_1024_{marker}.npz',
                 align_mode: str = 'intersection',
                 return_key: bool = False):
        """
        初始化数据集

        Args:
            base_dir: 存放各 marker NPZ 的目录
            modality_names:  需要对齐加载的 modality_names 列表（如 ['CD3','CD8',...']）
            filename_template: 文件名模板，默认 'tma_uni_tile_1024_{marker}.npz'
            align_mode: 对齐模式，'intersection'（仅公共规范化键）或 'union'（全部规范化键，缺失时补零）
            return_key: __getitem__ 是否返回键名（返回规范化键），便于调试与追踪
        """
        super().__init__()
        if align_mode not in ('intersection', 'union'):
            raise ValueError("align_mode 必须为 'intersection' 或 'union'")

        self.base_dir = base_dir
        self.modality_names = list(modality_names)
        self.align_mode = align_mode
        # 强制不排序、不返回键
        self.return_key = False

        # 编译规范化键解析正则
        self._norm_pat = re.compile(r"_block(\d+)_x(\d+)_y(\d+)_patient(\w+)$")

        # 构建路径映射并打开文件（mmap 只读，节省内存）
        self.modality_to_path: Dict[str, str] = {}
        self.modality_to_npz: Dict[str, np.lib.npyio.NpzFile] = {}

        for modality in self.modality_names:
            path = os.path.join(base_dir, filename_template.format(marker=modality))
            if not os.path.exists(path):
                logging.warning(f"NPZ 文件不存在: {path}")
            self.modality_to_path[modality] = path
            if os.path.exists(path):
                self.modality_to_npz[modality] = np.load(path, allow_pickle=True, mmap_mode='r')

        # 为每个 modality 建立 规范化键 -> 原始键 的映射
        self.normalized_to_raw_key: Dict[str, Dict[Tuple[int,int,int,str], str]] = {}
        normalized_sets: Dict[str, set] = {}
        for modality, npz in self.modality_to_npz.items():
            mapping: Dict[Tuple[int,int,int,str], str] = {}
            for raw_key in npz.keys():
                m = self._norm_pat.search(raw_key)
                if not m:
                    continue
                norm = (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
                mapping[norm] = raw_key
            self.normalized_to_raw_key[modality] = mapping
            normalized_sets[modality] = set(mapping.keys())
            logging.info(f"{modality}: normalized={len(mapping)}")

        if not normalized_sets:
            raise RuntimeError("未找到可用的规范化键集合")

        # 计算规范化键的对齐集合
        if self.align_mode == 'intersection':
            normalized_keys = set.intersection(*normalized_sets.values())
        else:
            normalized_keys = set().union(*normalized_sets.values())

        # 展开 patch 级别的数据：将每个 (n_patches, 1024) 展开为 n_patches 个独立样本
        # 直接修改 normalized_keys，包含 patch_id
        self.normalized_keys = self._expand_patch_keys(normalized_keys)

        # 统计（按 tma_counter 的口径：仅 intersection 与最大样本量）
        inter_size = len(set.intersection(*normalized_sets.values()))
        max_marker, max_count = None, -1
        for m, s in normalized_sets.items():
            if len(s) > max_count:
                max_marker, max_count = m, len(s)
        
        total_samples = len(self.normalized_keys)
        logging.info(
            f"TMA 对齐(规范化): modality_names={self.modality_names} | mode={self.align_mode} | original_keys={len(normalized_keys)} | "
            f"intersection={inter_size} | max_modality={max_marker} | max_samples={max_count} | "
            f"expanded_samples={total_samples}")

    def _expand_patch_keys(self, normalized_keys: List[Tuple[int,int,int,str]]) -> List[Tuple[int,int,int,str,int]]:
        """
        展开 patch 级别的键：将每个 (n_patches, 1024) 展开为 n_patches 个独立样本
        
        Args:
            normalized_keys: 原始规范化键列表
            
        Returns:
            List[Tuple[int,int,int,str,int]]: [(block, x, y, patient, patch_id), ...]
        """
        expanded_keys = []
        
        for norm_key in normalized_keys:
            # 获取第一个模态的 patch 数量
            n_patches = None
            for modality in self.modality_names:
                raw_key = self.normalized_to_raw_key.get(modality, {}).get(norm_key)
                npz = self.modality_to_npz.get(modality)
                if raw_key is not None and npz is not None:
                    vec = npz[raw_key]
                    if vec.shape == (1024,):
                        # Tile 级别：reshape 为 (1, 1024)
                        n_patches = 1
                    elif len(vec.shape) == 2 and vec.shape[1] == 1024:
                        # Patch 级别：直接使用
                        n_patches = vec.shape[0]
                    else:
                        raise ValueError(f"{modality} 键 {raw_key} 的向量形状不支持，实际为 {vec.shape}")
                    break
            
            if n_patches is not None:
                # 为每个 patch 创建一个展开的键，直接包含 patch_id
                for patch_id in range(n_patches):
                    expanded_keys.append(norm_key + (patch_id,))
            else:
                raise ValueError(f"未找到 {norm_key} 的向量")
        
        return expanded_keys

    def __len__(self) -> int:
        """返回样本数量（展开后的数量）"""
        return len(self.normalized_keys)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Tuple[int,int,int,str,int], Dict[str, torch.Tensor]]]:
        """
        获取单条样本（支持 patch 展开）

        返回：
            {
              'features': {modality: Tensor[1024]},
              'mask': {modality: Tensor[bool]},
              'key': 规范化键 (block,x,y,patient,patch_id)
            }
        """
        # 获取展开后的键，现在直接包含 patch_id
        expanded_key = self.normalized_keys[index]
        norm_key = expanded_key[:4]  # (block, x, y, patient)
        patch_id = expanded_key[4]   # patch_id
        
        features: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}

        for modality in self.modality_names:
            raw_key = self.normalized_to_raw_key.get(modality, {}).get(norm_key)
            npz = self.modality_to_npz.get(modality)
            if raw_key is None or npz is None:
                # intersection 模式下不会出现缺失；union 模式允许缺失时补零
                features[modality] = torch.zeros(1024, dtype=torch.float32)
                mask[modality] = torch.tensor(False)
                continue
            vec = npz[raw_key]
            
            # 统一处理：都按 (n_patches, 1024) 格式处理
            if vec.shape == (1024,):
                # Tile 级别：reshape 为 (1, 1024)
                vec = vec.reshape(1, 1024)
            elif len(vec.shape) == 2 and vec.shape[1] == 1024:
                # Patch 级别：直接使用
                pass
            else:
                raise ValueError(f"{modality} 键 {raw_key} 的向量形状不支持，实际为 {vec.shape}")
            
            # 选择特定的 patch
            if patch_id >= vec.shape[0]:
                raise ValueError(f"patch_id {patch_id} 超出范围 {vec.shape[0]}")
            
            patch_vec = vec[patch_id]  # 选择第 patch_id 个 patch
            if patch_vec.dtype != np.float32:
                patch_vec = patch_vec.astype(np.float32, copy=False)
            features[modality] = torch.from_numpy(patch_vec)
            mask[modality] = torch.tensor(True)

        sample: Dict[str, Union[Dict[str, torch.Tensor], Tuple[int,int,int,str,int]]] = {
            'features': features,
            'mask': mask,
        }
        if self.return_key:
            sample['key'] = expanded_key
        return sample

    def close(self) -> None:
        """
        关闭已打开的 NPZ 句柄
        """
        for npz in self.modality_to_npz.values():
            try:
                npz.close()
            except Exception:
                pass

    def stats(self) -> Dict[str, int]:
        """
        返回各 modality 的规范化键数量统计
        """
        return {m: len(self.normalized_to_raw_key.get(m, {})) for m in self.modality_names}

    # === Split：按 ID 列表切分为子数据集（只读视图） ===
    def _key_to_id(self, norm_key: Tuple[int,int,int,str,int], id_type: str) -> Union[str, Tuple[int,int,int,str], Tuple[int,int,int,str,int]]:
        if id_type == 'patient':
            return norm_key[3]  # 患者ID
        elif id_type == 'tuple':
            return norm_key  # 完整键 (block,x,y,patient,patch_id)
        elif id_type == 'spatial':
            return norm_key[:4]  # 空间键 (block,x,y,patient)，不包含patch_id
        else:
            raise ValueError("id_type 仅支持 'patient'、'tuple' 或 'spatial'")

    def split_by_ids(self,
                     id_groups: Dict[str, List[Union[str, Tuple[int,int,int,str], Tuple[int,int,int,str,int]]]],
                     id_type: str = 'patient') -> Dict[str, 'TMANpzAlignedView']:
        """
        根据给定 ID 列表切分为多个子集。
        - id_type='patient': 以患者ID切分，ids 为如 '296' 的字符串
        - id_type='spatial': 以空间键切分，ids 为 (block,x,y,patient) 元组
        - id_type='tuple':   以完整键切分，ids 为 (block,x,y,patient,patch_id) 元组
        返回 {split_name: TMANpzAlignedView}
        """
        key_to_id = {k: self._key_to_id(k, id_type) for k in self.normalized_keys}
        out: Dict[str, TMANpzAlignedView] = {}
        for name, ids in id_groups.items():
            id_set = set(ids)
            subset_keys = [k for k in self.normalized_keys if key_to_id[k] in id_set]
            out[name] = TMANpzAlignedView(parent=self, normalized_keys=subset_keys)
        return out

class GlobalMismatchSampler:
    """
    基于整个 Dataset 规范化键的全局负样本采样器。

    用法：
        sampler = GlobalMismatchSampler(dataset_normalized_keys)
        indices_dict = sampler.sample(num_pairs, num_modalities, device)
        # indices_dict: {modality_name: LongTensor[num_pairs]}，可用于索引全局池中的样本

    说明：
        - 为提高覆盖率，采样在整个 normalized_keys 范围进行，而非受限于单个 batch。
        - 保证不同模态在每个位置的组合尽量唯一；若重复则进行修复。
    """
    def __init__(self, normalized_keys: List[Tuple[int,int,int,str,int]]):
        self.pool_size = len(normalized_keys)
        if self.pool_size == 0:
            raise ValueError("normalized_keys 为空，无法构建全局采样器")

    def sample(self, num_pairs: int, modality_names: List[str], device: str) -> Dict[str, torch.Tensor]:
        """
        在全局范围为每个模态采样索引，并修复重复组合。
        Returns:
            {modality: LongTensor[num_pairs]}
        """
        indices: Dict[str, torch.Tensor] = {}
        for name in modality_names:
            indices[name] = torch.randint(0, self.pool_size, (num_pairs,), device=device)
        # 修复重复组合
        modality_names_list = list(modality_names)
        combos = {}
        for i in range(num_pairs):
            combo = str([indices[m][i].item() for m in modality_names_list])
            combos.setdefault(combo, []).append(i)
        for _, positions in combos.items():
            if len(positions) > 1:
                for pos in positions[1:]:
                    # 重新随机直到唯一
                    max_attempts = 50
                    for _ in range(max_attempts):
                        m = modality_names_list[torch.randint(0, len(modality_names_list), (1,), device=device).item()]
                        old = indices[m][pos].item()
                        new = torch.randint(0, self.pool_size, (1,), device=device).item()
                        if new == old:
                            continue
                        indices[m][pos] = new
                        new_combo = str([indices[x][pos].item() for x in modality_names_list])
                        if new_combo not in combos:
                            break
        return indices

class TMANpzAlignedWithNegDataset(TMANpzAlignedDataset):
    """
    预生成全局负样本组合的 Dataset：
    - 继承自 TMANpzAlignedDataset（按规范化键交集对齐，跳过缺失）
    - 初始化时基于 normalized_keys 全局采样，为每个样本预存一组 mismatch 组合（每个模态各一个索引）
    - __getitem__ 返回正样本 features 与预生成的负样本 features_neg，训练时直接读取即可
    """

    def __init__(self,
                 base_dir: str,
                 modality_names: List[str],
                 filename_template: str = 'tma_uni_tile_1024_{marker}.npz',
                 align_mode: str = 'intersection',
                 return_key: bool = False,
                 mismatch_ratio: float = 1.0,
                 seed: Optional[int] = 42):
        super().__init__(
            base_dir=base_dir,
            modality_names=modality_names,
            filename_template=filename_template,
            align_mode=align_mode,
            return_key=return_key,
        )
        self.mismatch_ratio = float(mismatch_ratio)
        self._rng = np.random.RandomState(seed if seed is not None else None)

        # 构造全局采样器并预生成每个样本的负样本索引组合
        # 使用展开后的键
        self._global_sampler = GlobalMismatchSampler(self.normalized_keys)
        # 一次性生成全局负样本池：大小 = len(keys) * ratio（向上取整）
        self._neg_pool: List[Dict[str, int]] = []  # [n_pool][modality] -> neg_idx
        self._build_negative_pool()

    def _build_negative_pool(self) -> None:
        # 使用展开后的总样本数
        total = len(self.normalized_keys)
        modality_names = self.modality_names
        n_pool = int(np.ceil(total * max(0.0, self.mismatch_ratio)))
        n_pool = max(n_pool, total if self.mismatch_ratio <= 0 else 1)
        idx_dict = self._global_sampler.sample(n_pool, modality_names, device='cpu')
        self._neg_pool = []
        for i in range(n_pool):
            combo: Dict[str, int] = {}
            for m in modality_names:
                neg_global_idx = int(idx_dict[m][i].item())
                # 避免与同位置正样本偶然一致：无法知道正样本索引，此处仅确保多样性
                if neg_global_idx >= total:
                    neg_global_idx = neg_global_idx % total
                combo[m] = neg_global_idx
            self._neg_pool.append(combo)

    def resample_negatives(self, seed: Optional[int] = None) -> None:
        """重建全局负样本池。"""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self._build_negative_pool()

    def get_negatives_for_batch(self, batch_id: int, batch_size: int, ratio: float) -> List[Dict[str, int]]:
        """
        根据 batch_id 与 batch_size、ratio 返回足量的负样本组合（从全局池切片）。
        返回的是规范化键索引组合，训练端可据此从 features 池取向量。
        """
        need = int(np.ceil(batch_size * max(0.0, ratio)))
        if need <= 0:
            return []
        start = (batch_id * need) % max(1, len(self._neg_pool))
        out = []
        for i in range(need):
            out.append(self._neg_pool[(start + i) % len(self._neg_pool)])
        return out

    # === Split：带负样本池版本的切分 ===
    def split_by_ids_with_neg(self,
                              id_groups: Dict[str, List[Union[str, Tuple[int,int,int,str], Tuple[int,int,int,str,int]]]],
                              id_type: str = 'patient',
                              mismatch_ratio: Optional[float] = None,
                              seed: Optional[int] = None) -> Dict[str, 'TMANpzAlignedWithNegView']:
        """
        根据给定 ID 列表切分为多个子集，并为每个子集重建其全局负样本池。
        mismatch_ratio/seed 不传则分别沿用父数据集的 ratio 与默认 42。
        - id_type='patient': 以患者ID切分，ids 为如 '296' 的字符串
        - id_type='spatial': 以空间键切分，ids 为 (block,x,y,patient) 元组
        - id_type='tuple':   以完整键切分，ids 为 (block,x,y,patient,patch_id) 元组
        返回 {split_name: TMANpzAlignedWithNegView}
        """
        key_to_id = {k: self._key_to_id(k, id_type) for k in self.normalized_keys}
        out: Dict[str, TMANpzAlignedWithNegView] = {}
        ratio = self.mismatch_ratio if mismatch_ratio is None else float(mismatch_ratio)
        sd = 42 if seed is None else int(seed)
        for name, ids in id_groups.items():
            id_set = set(ids)
            subset_keys = [k for k in self.normalized_keys if key_to_id[k] in id_set]
            out[name] = TMANpzAlignedWithNegView(parent=self, normalized_keys=subset_keys,
                                                 mismatch_ratio=ratio, seed=sd)
        return out


class TMANpzAlignedView(Dataset):
    """
    只读子视图：复用父数据集的句柄与映射，仅替换 normalized_keys，支持 patch 展开。
    """
    def __init__(self, parent: TMANpzAlignedDataset, normalized_keys: List[Tuple[int,int,int,str,int]]):
        self.parent = parent
        # 子视图内使用稳定排序，保证复现
        self.normalized_keys = sorted(normalized_keys)
        self.modality_names = parent.modality_names
        
        # 展开 patch 级别的数据
        self.normalized_keys = self._expand_patch_keys(self.normalized_keys)

    def _expand_patch_keys(self, normalized_keys: List[Tuple[int,int,int,str]]) -> List[Tuple[int,int,int,str,int]]:
        """展开 patch 级别的键"""
        expanded_keys = []
        
        for norm_key in normalized_keys:
            # 获取第一个模态的 patch 数量
            n_patches = None
            for modality in self.modality_names:
                raw_key = self.parent.normalized_to_raw_key.get(modality, {}).get(norm_key)
                npz = self.parent.modality_to_npz.get(modality)
                if raw_key is not None and npz is not None:
                    vec = npz[raw_key]
                    if vec.shape == (1024,):
                        # Tile 级别：reshape 为 (1, 1024)
                        n_patches = 1
                    elif len(vec.shape) == 2 and vec.shape[1] == 1024:
                        # Patch 级别：直接使用
                        n_patches = vec.shape[0]
                    else:
                        raise ValueError(f"{modality} 键 {raw_key} 的向量形状不支持，实际为 {vec.shape}")
                    break
            
            if n_patches is not None:
                # 为每个 patch 创建一个展开的键，直接包含 patch_id
                for patch_id in range(n_patches):
                    expanded_keys.append(norm_key + (patch_id,))
            else:
                # 如果没有找到任何数据，创建一个默认的
                expanded_keys.append(norm_key + (0,))
        
        return expanded_keys

    def __len__(self) -> int:
        return len(self.normalized_keys)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        # 获取展开后的键，现在直接包含 patch_id
        expanded_key = self.normalized_keys[index]
        norm_key = expanded_key[:4]  # (block, x, y, patient)
        patch_id = expanded_key[4]   # patch_id
        
        features: Dict[str, torch.Tensor] = {}
        mask: Dict[str, torch.Tensor] = {}
        
        for modality in self.modality_names:
            raw_key = self.parent.normalized_to_raw_key.get(modality, {}).get(norm_key)
            npz = self.parent.modality_to_npz.get(modality)
            if raw_key is None or npz is None:
                features[modality] = torch.zeros(1024, dtype=torch.float32)
                mask[modality] = torch.tensor(False)
                continue
            vec = npz[raw_key]
            
            # 统一处理：都按 (n_patches, 1024) 格式处理
            if vec.shape == (1024,):
                # Tile 级别：reshape 为 (1, 1024)
                vec = vec.reshape(1, 1024)
            elif len(vec.shape) == 2 and vec.shape[1] == 1024:
                # Patch 级别：直接使用
                pass
            else:
                raise ValueError(f"{modality} 键 {raw_key} 的向量形状不支持，实际为 {vec.shape}")
            
            # 选择特定的 patch
            if patch_id >= vec.shape[0]:
                raise ValueError(f"patch_id {patch_id} 超出范围 {vec.shape[0]}")
            
            patch_vec = vec[patch_id]  # 选择第 patch_id 个 patch
            if patch_vec.dtype != np.float32:
                patch_vec = patch_vec.astype(np.float32, copy=False)
            features[modality] = torch.from_numpy(patch_vec)
            mask[modality] = torch.tensor(True)
        
        return {'features': features, 'mask': mask}


class TMANpzAlignedWithNegView(TMANpzAlignedView):
    """
    含全局负样本池的只读子视图：对所给 subset 重建负样本池，支持批量负样本读取。
    """
    def __init__(self, parent: TMANpzAlignedWithNegDataset,
                 normalized_keys: List[Tuple[int,int,int,str,int]],
                 mismatch_ratio: float = 1.0,
                 seed: Optional[int] = 42):
        super().__init__(parent=parent, normalized_keys=normalized_keys)
        self.parent = parent
        self.mismatch_ratio = float(mismatch_ratio)
        # 使用展开后的键来构建采样器（在 super().__init__ 之后，normalized_keys 已经被设置）
        self._sampler = GlobalMismatchSampler(self.normalized_keys)
        self._neg_pool: List[Dict[str, int]] = []
        self._build_pool(seed=seed)

    def _build_pool(self, seed: Optional[int]) -> None:
        # 使用展开后的总样本数
        total = len(self.normalized_keys)
        modality_names = self.modality_names
        n_pool = int(np.ceil(total * max(0.0, self.mismatch_ratio)))
        n_pool = max(n_pool, total if self.mismatch_ratio <= 0 else 1)
        # 固定 seed 的全局索引采样（通过 torch.Generator 也可，但这里沿用 RandomState 控制在 sampler 内部）
        idx_dict = self._sampler.sample(n_pool, modality_names, device='cpu')
        self._neg_pool = []
        for i in range(n_pool):
            combo: Dict[str, int] = {}
            for m in modality_names:
                neg_idx = int(idx_dict[m][i].item()) % total
                combo[m] = neg_idx
            self._neg_pool.append(combo)

    def get_negatives_for_batch(self, batch_id: int, batch_size: int, ratio: float) -> List[Dict[str, int]]:
        """
        根据 batch_id 和 ratio 返回负样本索引组合列表
        
        Args:
            batch_id: 当前批次ID
            batch_size: 批次大小
            ratio: 负样本比例
            
        Returns:
            List[Dict[str, int]]: 负样本组合列表，每个字典为 {modality: neg_idx}
        """
        need = int(np.ceil(batch_size * max(0.0, ratio)))
        if need <= 0:
            return []
        start = (batch_id * need) % max(1, len(self._neg_pool))
        return [self._neg_pool[(start + i) % len(self._neg_pool)] for i in range(need)]
    
    def build_negative_tensors(self, neg_combos: List[Dict[str, int]]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        将负样本索引组合物化为张量
        
        Args:
            neg_combos: 负样本组合列表 [{modality: neg_idx}, ...]
            
        Returns:
            (features_neg, mask_neg): 负样本特征和掩码，每个为 {modality: Tensor[n_neg, 1024]}
        """
        if not neg_combos:
            # 返回空张量
            empty_features = {m: torch.zeros(0, 1024, dtype=torch.float32) for m in self.modality_names}
            empty_masks = {m: torch.zeros(0, dtype=torch.bool) for m in self.modality_names}
            return empty_features, empty_masks
        
        n_neg = len(neg_combos)
        features_neg: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modality_names}
        mask_neg: Dict[str, List[torch.Tensor]] = {m: [] for m in self.modality_names}
        
        for combo in neg_combos:
            for modality in self.modality_names:
                neg_idx = combo[modality]
                
                # 解析负样本索引：使用展开后的键
                expanded_key = self.normalized_keys[neg_idx]
                norm_key = expanded_key[:4]  # (block, x, y, patient)
                patch_id = expanded_key[4]   # patch_id
                
                raw_key = self.parent.normalized_to_raw_key.get(modality, {}).get(norm_key)
                npz = self.parent.modality_to_npz.get(modality)
                
                if raw_key is None or npz is None:
                    features_neg[modality].append(torch.zeros(1024, dtype=torch.float32))
                    mask_neg[modality].append(torch.tensor(False))
                else:
                    vec = npz[raw_key]
                    
                    # 统一处理：都按 (n_patches, 1024) 格式处理
                    if vec.shape == (1024,):
                        # Tile 级别：reshape 为 (1, 1024)
                        vec = vec.reshape(1, 1024)
                    elif len(vec.shape) == 2 and vec.shape[1] == 1024:
                        # Patch 级别：直接使用
                        pass
                    else:
                        raise ValueError(f"{modality} 键 {raw_key} 的向量形状不支持，实际为 {vec.shape}")
                    
                    # 选择特定的 patch
                    if patch_id >= vec.shape[0]:
                        raise ValueError(f"patch_id {patch_id} 超出范围 {vec.shape[0]}")
                    
                    patch_vec = vec[patch_id]  # 选择第 patch_id 个 patch
                    if patch_vec.dtype != np.float32:
                        patch_vec = patch_vec.astype(np.float32, copy=False)
                    features_neg[modality].append(torch.from_numpy(patch_vec))
                    mask_neg[modality].append(torch.tensor(True))
        
        # 堆叠为 [n_neg, 1024]
        features_stacked = {m: torch.stack(features_neg[m]) for m in self.modality_names}
        masks_stacked = {m: torch.stack(mask_neg[m]) for m in self.modality_names}
        
        return features_stacked, masks_stacked


def build_collate_fn(view: TMANpzAlignedWithNegView, ratio: float):
    """
    构建用于 DataLoader 的 collate_fn，负责堆叠正样本并生成负样本
    
    Args:
        view: TMANpzAlignedWithNegView 实例
        ratio: 负样本比例（负样本数 = ceil(batch_size * ratio)）
        
    Returns:
        collate_fn: 可用于 DataLoader 的 collate 函数
        
    用法:
        >>> train_loader = DataLoader(
        ...     train_view, 
        ...     batch_size=128, 
        ...     shuffle=True,
        ...     collate_fn=build_collate_fn(train_view, ratio=1.0)
        ... )
    """
    # 使用闭包捕获 batch_id 计数器
    batch_counter = {'count': 0}
    
    def collate_fn(batch_list: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        DataLoader 的 collate 函数：堆叠正样本 + 生成负样本
        
        Args:
            batch_list: __getitem__ 返回的样本列表
            
        Returns:
            {
                'features': {modality: Tensor[batch_size, 1024]},
                'mask': {modality: Tensor[batch_size]},
                'features_neg': {modality: Tensor[n_neg, 1024]},
                'mask_neg': {modality: Tensor[n_neg]}
            }
        """
        batch_size = len(batch_list)
        
        # 📦 堆叠正样本
        pos_features: Dict[str, List[torch.Tensor]] = {m: [] for m in view.modality_names}
        pos_masks: Dict[str, List[torch.Tensor]] = {m: [] for m in view.modality_names}
        
        for sample in batch_list:
            for modality in view.modality_names:
                pos_features[modality].append(sample['features'][modality])
                pos_masks[modality].append(sample['mask'][modality])
        
        features_stacked = {m: torch.stack(pos_features[m]) for m in view.modality_names}
        masks_stacked = {m: torch.stack(pos_masks[m]) for m in view.modality_names}
        
        # 🔄 生成负样本（根据 batch_id 和 ratio）
        batch_id = batch_counter['count']
        batch_counter['count'] += 1
        
        neg_combos = view.get_negatives_for_batch(batch_id, batch_size, ratio)
        features_neg, masks_neg = view.build_negative_tensors(neg_combos)
        
        return {
            'features': features_stacked,
            'mask': masks_stacked,
            'features_neg': features_neg,
            'mask_neg': masks_neg,
        }
    
    return collate_fn


def create_tma_aligned_with_neg_dataset(
    base_dir: str,
    modality_names: List[str],
    align_mode: str = 'intersection',
    filename_template: str = 'tma_uni_tile_1024_{marker}.npz',
    mismatch_ratio: float = 1.0,
    seed: Optional[int] = 42,
) -> TMANpzAlignedWithNegDataset:
    """
    创建带全局负样本池的 TMA 对齐数据集（不排序、不返回键）。

    参数:
        base_dir: NPZ 目录（如 /home/zheng/.../TMA_Core_encodings）
        modality_names:  参与对齐的 modality_names 列表
        align_mode: 对齐模式，默认 'intersection'（仅保留8-marker完整组）
        filename_template: 文件名模板
        mismatch_ratio: 负样本池大小系数（约= len(keys) * ratio）
        seed: 负样本池随机种子

    返回:
        TMANpzAlignedWithNegDataset 实例
    """
    return TMANpzAlignedWithNegDataset(
        base_dir=base_dir,
        modality_names=modality_names,
        filename_template=filename_template,
        align_mode=align_mode,
        return_key=False,
        mismatch_ratio=mismatch_ratio,
        seed=seed,
    )