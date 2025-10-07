"""
TMA特征提取模块 - 使用UNI模型
支持对TMA_TumorCenter_Cores_ori目录中的PNG图像进行特征提取
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict
import logging

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_pngs(tile_dir: Path) -> List[Path]:
    """
    列出目录中的PNG文件，按确定性顺序排序
    
    Args:
        tile_dir: 包含PNG瓦片的目录
        
    Returns:
        有序的PNG路径列表
    """
    return sorted([p for p in tile_dir.iterdir() if p.suffix.lower() == ".png"])


def build_uni_model(device: str = "cpu"):
    """
    通过timm从HuggingFace hub加载UNI编码器
    
    Args:
        device: "cpu" 或 "cuda" (需要兼容的torch/CUDA)
        
    Returns:
        (model, transform)
    """
    if timm is None:
        raise RuntimeError("timm/torch不可用。请安装 torch>=2.0, timm>=0.9.8")

    # 从本地文件读取并设置Hugging Face token
    try:
        token_file = Path("/home/zheng/zheng/bin/.commands/.env")
        if token_file.exists():
            with open(token_file, 'r') as f:
                content = f.read()
                for line in content.split('\n'):
                    if line.startswith('HF_TOKEN='):
                        token = line.split('=', 1)[1].strip()
                        if token:
                            os.environ['HF_TOKEN'] = token
                            logger.info("✅ Hugging Face token已设置")
                            break
        else:
            logger.warning(f"Token文件不存在: {token_file}")
    except Exception as e:
        logger.warning(f"读取token文件失败: {e}")

    # 创建UNI模型 - 输出维度固定为1024
    model = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
        num_classes=0,  # 设置为0以获取特征而不是分类输出
    )
    
    # 获取模型配置并创建transform
    # UNI模型会自动将输入图像调整到224x224
    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**cfg)
    
    model.eval()
    model.to(device)
    
    logger.info(f"✅ UNI模型加载完成 - 输出维度: 1024")
    return model, transform


def load_image(path: Path) -> Image.Image:
    """
    加载RGB PIL图像
    
    Args:
        path: 图像路径
        
    Returns:
        RGB PIL图像
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def extract_patches_from_image(
    img: Image.Image, 
    patch_size: int = 256, 
    stride: int = 128,
    white_threshold: float = None,
    min_content_ratio: float = None
) -> List[Image.Image]:
    """
    从图像中提取patches，可选择过滤掉包含太多白色/空白区域的patches
    
    Args:
        img: 输入图像
        patch_size: patch尺寸
        stride: 步长
        white_threshold: 白色像素阈值 (0-1)，为None时不进行过滤
        min_content_ratio: 最小内容比例，为None时不进行过滤
        
    Returns:
        patch列表
    """
    width, height = img.size
    patches = []
    
    # 如果图像尺寸小于patch_size，直接调整图像大小
    if width < patch_size or height < patch_size:
        logger.info(f"图像尺寸 {width}x{height} 小于patch尺寸 {patch_size}，直接调整图像大小")
        img_resized = img.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
        patches.append(img_resized)
        return patches
    
    # 计算可以提取的patch数量
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            # 提取patch
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            
            # 如果设置了threshold参数，则进行patch验证
            if white_threshold is not None and min_content_ratio is not None:
                # 检查patch是否包含足够的内容（非白色区域）
                if _is_patch_valid(patch, white_threshold, min_content_ratio):
                    patches.append(patch)
                else:
                    logger.debug(f"跳过包含太多空白区域的patch: ({x}, {y})")
            else:
                # 不进行验证，直接添加所有patches
                patches.append(patch)
    
    return patches


def _is_patch_valid(patch: Image.Image, white_threshold: float, min_content_ratio: float) -> bool:
    """
    检查patch是否有效（不包含太多白色/空白区域）
    
    Args:
        patch: 图像patch
        white_threshold: 白色像素阈值
        min_content_ratio: 最小内容比例
        
    Returns:
        是否有效
    """
    # 转换为numpy数组
    img_array = np.array(patch)
    
    # 计算白色像素比例（RGB都接近255）
    white_pixels = np.all(img_array >= white_threshold * 255, axis=2)
    white_ratio = np.mean(white_pixels)
    
    # 内容比例 = 1 - 白色比例
    content_ratio = 1 - white_ratio
    
    return content_ratio >= min_content_ratio

def extract_uni_features_from_patches(
    patches: List[Image.Image],
    model,
    transform,
    device: str = "cpu",
    batch_size: int = 32
) -> np.ndarray:
    """
    使用UNI为patch列表提取特征
    
    Args:
        patches: patch图像列表
        model: UNI模型
        transform: 来自timm配置的图像变换
        device: "cpu" 或 "cuda"
        batch_size: 推理的批大小
        
    Returns:
        形状为 (N, 1024) 的数组 - UNI固定输出1024维
    """
    feats: List[np.ndarray] = []
    
    with torch.inference_mode():
        for i in tqdm(range(0, len(patches), batch_size), desc="提取UNI特征", unit="batch"):
            batch_images = []
            for patch in patches[i : i + batch_size]:
                # 应用变换 - UNI会自动调整到224x224
                transformed_img = transform(patch).unsqueeze(0)
                batch_images.append(transformed_img)
            
            # 拼接批次
            batch_t = torch.cat(batch_images, dim=0).to(device)
            
            # 提取特征 - UNI输出固定为1024维
            emb = model(batch_t)  # [B, 1024]
            feats.append(emb.detach().cpu().numpy())
            
            # 清理内存
            del batch_t, emb, batch_images
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    
    return np.concatenate(feats, axis=0)


def extract_uni_features_from_image(
    img_path: Path,
    model,
    transform,
    device: str = "cpu",
    batch_size: int = 32,
    patch_size: int = 256,
    stride: int = 128,
    white_threshold: float = 0.8,
    min_content_ratio: float = 0.1,
) -> np.ndarray:
    """
    从单个图像提取UNI特征（通过patches）
    
    Args:
        img_path: 图像路径
        model: UNI模型
        transform: 图像变换
        device: 设备
        batch_size: 批大小
        patch_size: patch尺寸
        stride: 步长
        
    Returns:
        形状为 (N, 1024) 的数组 - UNI固定输出1024维
    """
    # 加载图像
    img = load_image(img_path)
    
    # 提取patches
    patches = extract_patches_from_image(img, patch_size, stride, white_threshold, min_content_ratio)
    
    if not patches:
        logger.warning(f"从 {img_path} 中没有提取到patches")
        return np.zeros((0, 1024), dtype=np.float32)
    
    # 提取特征
    features = extract_uni_features_from_patches(patches, model, transform, device, batch_size)
    
    return features


def extract_marker_features(
    marker_dir: Path,
    model,
    transform,
    device: str = "cpu",
    batch_size: int = 32,
    patch_size: int = 256,
    stride: int = 128,
    white_threshold: float = 0.8,
    min_content_ratio: float = 0.1,
) -> Dict[str, np.ndarray]:
    """
    为单个标记目录提取特征
    
    Args:
        marker_dir: 标记目录路径
        model: UNI模型
        transform: 图像变换
        device: 设备
        batch_size: 批大小
        patch_size: patch尺寸
        stride: 步长
        
    Returns:
        特征字典，键为文件名（不含扩展名），值为1024维特征数组
    """
    pngs = list_pngs(marker_dir)
    if not pngs:
        logger.warning(f"在 {marker_dir} 中没有找到PNG文件")
        return {}
    
    logger.info(f"处理标记目录: {marker_dir.name} | PNG数量: {len(pngs)}")
    
    # 构建每PNG字典
    out_dict: Dict[str, np.ndarray] = {}
    
    # 使用tqdm显示每个图像的进度
    for png_path in tqdm(pngs, desc=f"处理 {marker_dir.name}", unit="image"):
        try:
            # 从单个图像提取特征
            features = extract_uni_features_from_image(
                png_path, model, transform, device, batch_size, 
                patch_size, stride, white_threshold, min_content_ratio
            )
            
            key = png_path.stem  # 不含扩展名的文件名
            out_dict[key] = features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"处理图像 {png_path} 时出错: {e}")
            continue
    
    return out_dict


def main() -> None:
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="使用UNI (ViT-L/16) 提取TMA PNG编码特征")
    parser.add_argument("png_root", type=str, help="包含TMA PNG子文件夹的根目录")
    parser.add_argument("dest_dir", type=str, help="NPZ特征的输出目录")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="推理设备")
    parser.add_argument("--batch_size", type=int, default=32, help="特征提取的批大小")
    parser.add_argument("--patch_size", type=int, default=256, help="patch尺寸")
    parser.add_argument("--stride", type=int, default=128, help="patch步长")
    # UNI模型输出维度固定为1024，不需要参数
    parser.add_argument("--markers", type=str, nargs="+", 
                       default=["CD3", "CD8", "CD56", "CD68", "CD163", "HE", "MHC1", "PDL1"],
                       help="要处理的标记列表")
    parser.add_argument("--gpu_id", type=int, default=0, help="使用的GPU ID")
    parser.add_argument("--white_threshold", type=float, default=None, 
                       help="白色像素阈值 (0-1)，超过此值认为是白色。不设置则不进行patch过滤")
    parser.add_argument("--min_content_ratio", type=float, default=None,
                       help="最小内容比例，低于此比例的patch会被过滤。不设置则不进行patch过滤")
    
    args = parser.parse_args()

    png_root = Path(args.png_root)
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 设置设备
    if args.device == "cuda":
        if torch is None or not torch.cuda.is_available():
            logger.warning("CUDA不可用，改用CPU")
            device = "cpu"
        else:
            device = f"cuda:{args.gpu_id}"
            # 设置可见的GPU
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    else:
        device = "cpu"

    logger.info(f"🚀 开始使用UNI提取TMA PNG特征...")
    logger.info(f"📁 PNG根目录: {png_root}")
    logger.info(f"📁 输出目录: {dest_dir}")
    logger.info(f"🖥 设备: {device}")
    logger.info(f"🔢 批大小: {args.batch_size}")
    logger.info(f"📏 Patch尺寸: {args.patch_size}")
    logger.info(f"👣 步长: {args.stride}")
    logger.info(f"📊 输出维度: 1024 (UNI固定)")
    logger.info(f"🏷 标记列表: {args.markers}")
    
    # 显示patch过滤设置
    if args.white_threshold is not None and args.min_content_ratio is not None:
        logger.info(f"🔍 Patch过滤: 启用 (白色阈值: {args.white_threshold}, 最小内容比例: {args.min_content_ratio})")
    else:
        logger.info(f"🔍 Patch过滤: 禁用 (提取所有patches)")
    
    logger.info("")

    # 构建模型
    logger.info("🔧 构建UNI模型...")
    model, transform = build_uni_model(device=device)
    logger.info("✅ 模型构建完成")

    # 处理每个标记目录
    processed_count = 0
    total_markers = len(args.markers)
    
    # 使用tqdm显示总体进度
    for marker in tqdm(args.markers, desc="处理标记", unit="marker"):
        marker_dir = png_root / f"tma_tumorcenter_{marker}"
        
        if not marker_dir.exists():
            logger.warning(f"标记目录不存在: {marker_dir}")
            continue
        
        if not marker_dir.is_dir():
            logger.warning(f"不是目录: {marker_dir}")
            continue
        
        try:
            # 提取特征
            features_dict = extract_marker_features(
                marker_dir, model, transform, device, args.batch_size, 
                args.patch_size, args.stride, args.white_threshold, args.min_content_ratio
            )
            
            if not features_dict:
                logger.warning(f"没有提取到特征: {marker}")
                continue
            
            # 保存特征
            output_path = dest_dir / f"tma_uni_patch_{args.patch_size}_stride_{args.stride}_dim_1024_{marker}.npz"
            np.savez_compressed(output_path, **features_dict)
            
            # 统计特征数量
            total_features = sum(features.shape[0] for features in features_dict.values())
            logger.info(f"✅ 保存: {output_path} (图像数: {len(features_dict)}, 总特征数: {total_features})")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"❌ 处理标记 {marker} 时出错: {e}")
            continue

    logger.info("")
    logger.info(f"🎉 特征提取完成！")
    logger.info(f"📊 成功处理: {processed_count}/{total_markers} 个标记")
    logger.info(f"📁 结果保存在: {dest_dir}")
    logger.info(f"📋 生成的文件:")
    
    # 列出生成的文件
    npz_files = list(dest_dir.glob("tma_uni_*.npz"))
    for npz_file in sorted(npz_files):
        logger.info(f"   {npz_file.name}")


if __name__ == "__main__":
    main()
