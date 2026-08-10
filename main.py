import argparse
import gc
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from OtherFile.Tools import ConfigTools
from OtherFile.Tool2 import DataTools1
from OtherFile.PlaintTools import plot_segment_comparison

from OtherFile.ModelTools import AccuracyTools, LossFuncs
from models.OCMTCN import get_ocmtcn_net

SELECTED_FEATURE_COLUMNS = ConfigTools.load_config_data("input_feature")
FOLDER_PATH = ConfigTools.load_config_data("input_path")
WINDOW_SIZE = ConfigTools.load_config_data("window_size")
STRIDE = ConfigTools.load_config_data("stride")
BATCH_SIZE = ConfigTools.load_config_data("batch_size")
EPOCHS = ConfigTools.load_config_data("epochs")
PATIENCE = ConfigTools.load_config_data("patience")  # 早停耐心值
N_SPLITS = ConfigTools.load_config_data("n_splits")
IS_PLIANT = ConfigTools.load_config_data("is_plaint")
MODEL_NAME = ConfigTools.load_config_data("model")
SEED = ConfigTools.load_config_data("seed")
PIC_OUT_PATH = ConfigTools.load_config_data("pic_out_path")
CONTENT = ConfigTools.load_config_data("content")
IS_ADD_LOSS = ConfigTools.load_config_data("is_add_loss")
IS_PLUS = ConfigTools.load_config_data("is_plus")
IS_MUTI = ConfigTools.load_config_data("is_muti")

lambda_smooth = 0.1
alpha = 0.2
TARGET_ENHANCE_CLASSES = list(range(14))
# TARGET_ENHANCE_CLASSES = [0,1,2,3,5,10,11,12,13]
AUGMENT_TARGET_THRESHOLD = 8000
MAX_MIXUP_PER_SAMPLE = 4

# MixUp 配置
MIXUP_PROB = 0.5             # 与静态生成无关，可保留
MIXUP_BETA_ALPHA = 0.2       # Beta 分布参数

# ---------------------------------------------------
# region ToolMethod
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("已清空显存并关闭 cuDNN benchmark")


def set_seed(seed=SEED):
    """固定所有随机种子，保证结果可复现"""
    # Python 原生随机种子
    import random
    random.seed(seed)

    # NumPy 随机种子
    np.random.seed(seed)

    # PyTorch 随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU时需设置

    # 强制 CUDA 确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（会引入随机性）

    # 设置 Python 哈希种子（可选，防止字典等哈希随机）
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"已设置随机种子为 {seed}")


class WellDataset(Dataset):
    def __init__(self, features, labels, indices=None, train=False, num_classes=14, well_ids=None):
        """
        features: list of numpy arrays (seq_len, n_features)
        labels: list of numpy arrays (seq_len,)
        indices: list of original indices (用于追溯)
        train: 是否为训练集
        num_classes: 总类别数（必须提供）
        """
        self.train = train
        self.num_classes = num_classes
        self.samples = []  # 每个元素为 (feat, label, length, idx, depth_mapping)
        # depth_mapping[i] = (well_idx, pos_in_slice) 或 None（混合样本）
        
        # 1. 原始样本（基础增强）
        for i, (feat, lab, idx) in enumerate(zip(features, labels,
                indices if indices else range(len(features)))):
            if train:
                feat = augment_well_features(feat.copy())
            # 构建深度点位置映射（使用井ID + 切片内位置作为唯一标识）
            well_idx = well_ids[i] if well_ids is not None else i
            depth_mapping = [(well_idx, pos) for pos in range(len(lab))]
            
            self.samples.append((feat, lab, len(feat), idx, depth_mapping))
        
        # ==============================================
        # 统计增强前的各类别深度点数量
        # ==============================================
        if train:
            print("\n=== 增强前各类别深度点数量（去重后） ===")
            before_stats, before_total = self._count_unique_depth_points(self.samples)
            for label in sorted(before_stats.keys()):
                print(f"  类别 {label}: {before_stats[label]} 个深度点")
            print(f"  总计: {before_total} 个深度点（{len(self.samples)} 个切片）")

        # 2. 静态生成 MixUp 样本（仅训练阶段且启用）
        if train and IS_PLUS and MIXUP_PROB > 0:
            print("\n正在生成带位置和比例约束的 MixUp 样本...")
            mixup_samples = self._generate_mixup_with_constraints(features, labels, indices)
            self.samples.extend(mixup_samples)
            print(f"生成 {len(mixup_samples)} 个混合样本，总样本数 {len(self.samples)}")

            # ==============================================
            # 统计增强后的各类别深度点数量（去重后）
            # ==============================================
            print("\n=== 增强后各类别深度点数量（去重后） ===")
            after_stats, after_total = self._count_unique_depth_points(self.samples)
            for label in sorted(after_stats.keys()):
                before_count = before_stats.get(label, 0)
                after_count = after_stats[label]
                increase = after_count - before_count
                print(f"  类别 {label}: {after_count} 个深度点 (增加 {increase:+d})")
            print(f"  总计: {after_total} 个深度点（{len(self.samples)} 个切片，增加 {len(self.samples) - len(features):+d} 个切片）")
    
    def _count_unique_depth_points(self, samples):
        """统计去重后的各类别深度点数量（基于井ID + 深度位置去重）
        
        参数:
            samples: 样本列表，每个元素为 (feat, label, length, idx, depth_mapping)
            
        返回:
            stats: 各类别深度点数量字典
            total: 总深度点数量
        """
        stats = {}
        seen = set()  # 记录已统计的深度点位置 (well_idx, depth_pos)
        
        for _, lab_seq, _, _, depth_mapping in samples:
            for i, lab in enumerate(lab_seq):
                # 获取深度点位置（混合样本的位置为None）
                pos = depth_mapping[i] if depth_mapping is not None else None
                
                if pos is None:
                    # 混合样本：直接计数（不参与去重）
                    stats[lab] = stats.get(lab, 0) + 1
                else:
                    # 原始样本：基于位置去重
                    if pos not in seen:
                        stats[lab] = stats.get(lab, 0) + 1
                        seen.add(pos)
        
        return stats, sum(stats.values())


    def _generate_mixup_with_constraints(self, features, labels, indices):
        """生成满足位置约束和比例约束的 MixUp 样本，返回列表 [(feat, hard_label, length, -1)]"""
        num_classes = self.num_classes
        if num_classes is None:
            # fallback: 自动推断
            all_labels = np.concatenate([l for l in labels])
            num_classes = int(all_labels.max()) + 1
        
        mixup_list = []
        
        # 统计每个类别的深度点数量
        class_depth_counts = {}
        for lab_seq in labels:
            for lab in lab_seq:
                class_depth_counts[lab] = class_depth_counts.get(lab, 0) + 1
        
        # 需要增强的目标类别（深度点数量少于阈值）
        target_classes = [c for c in TARGET_ENHANCE_CLASSES 
                          if class_depth_counts.get(c, 0) < AUGMENT_TARGET_THRESHOLD]
        if not target_classes:
            print("所有目标类别均已达到阈值，不生成 MixUp 样本")
            return mixup_list
        
        # 预计算每个样本中每个目标类别的比例和平均位置
        sample_info = {}  # {sample_idx: {target_label: (ratio, avg_position)}}
        for i, lab_seq in enumerate(labels):
            length = len(lab_seq)
            for c in set(lab_seq):
                if c in target_classes:
                    positions = np.where(lab_seq == c)[0]
                    count = len(positions)
                    ratio = count / length
                    avg_pos = positions.mean() / length
                    sample_info.setdefault(i, {})[c] = (ratio, avg_pos)
        
        # 为每个目标类别独立处理
        for target_label in target_classes:
            # 收集包含该类别的样本及其信息
            candidates = []  # 每个元素 (idx, region, ratio)
            for idx, info in sample_info.items():
                if target_label in info:
                    ratio, avg_pos = info[target_label]
                    # 确定位置区域
                    if avg_pos < 0.33:
                        region = 'upper'
                    elif avg_pos < 0.66:
                        region = 'middle'
                    else:
                        region = 'lower'
                    candidates.append((idx, region, ratio))
            
            if len(candidates) < 2:
                continue
            
            # 按位置区域分组
            region_groups = {'upper': [], 'middle': [], 'lower': []}
            for idx, reg, ratio in candidates:
                region_groups[reg].append((idx, ratio))
            
            # 计算还需要生成的样本数量（基于深度点缺口）
            needed_depth = max(0, AUGMENT_TARGET_THRESHOLD - class_depth_counts[target_label])
            samples_needed = int(np.ceil(needed_depth / WINDOW_SIZE))
            # 限制总生成数量
            total_max = len(candidates) * MAX_MIXUP_PER_SAMPLE
            samples_needed = min(samples_needed, total_max)
            if samples_needed <= 0:
                continue
            
            # 用于记录每个样本已参与混合的次数
            mixup_count = {idx: 0 for idx, _, _ in candidates}
            
            # 对每个位置区域独立生成混合样本
            for region, items in region_groups.items():
                if len(items) < 2:
                    continue
                # 按比例排序后分成3个等频桶（低、中、高）
                items_sorted = sorted(items, key=lambda x: x[1])  # 按比例升序
                bucket_size = len(items_sorted) // 3
                if bucket_size == 0:
                    bucket_size = 1
                buckets = []
                for i in range(0, len(items_sorted), bucket_size):
                    buckets.append(items_sorted[i:i+bucket_size])
                # 确保正好3个桶（如果余数，最后一个桶会大一些，没关系）
                
                # 计算该区域应该分到的生成数量（按区域样本数比例分配）
                region_ratio = len(items) / len(candidates)
                region_target = int(samples_needed * region_ratio)
                # 至少每个区域尝试生成一些样本
                if region_target == 0 and len(items) >= 2 and samples_needed > 0:
                    region_target = 1
                # 限制该区域最大生成数量
                region_max = len(items) * MAX_MIXUP_PER_SAMPLE
                region_target = min(region_target, region_max)
                
                generated = 0
                # 在每个桶内随机配对
                for bucket in buckets:
                    if len(bucket) < 2:
                        continue
                    # 计算该桶的生成配额（按桶内样本数比例）
                    bucket_ratio = len(bucket) / len(items)
                    bucket_target = int(region_target * bucket_ratio)
                    if bucket_target == 0 and generated < region_target and len(bucket) >= 2:
                        bucket_target = 1
                    # 生成样本
                    for _ in range(bucket_target):
                        if generated >= region_target:
                            break
                        # 随机选择两个不同索引
                        if len(bucket) == 1:
                            # 如果桶内只有一个样本，尝试其他桶？这里简单跳过
                            continue
                        pair_idx = np.random.choice(len(bucket), 2, replace=False)
                        idx1, ratio1 = bucket[pair_idx[0]]
                        idx2, ratio2 = bucket[pair_idx[1]]
                        # 检查参与次数限制
                        if mixup_count[idx1] >= MAX_MIXUP_PER_SAMPLE or mixup_count[idx2] >= MAX_MIXUP_PER_SAMPLE:
                            continue
                        # 采样 lambda
                        lam = np.random.beta(MIXUP_BETA_ALPHA, MIXUP_BETA_ALPHA)
                        # 特征混合
                        mixed_feat = lam * features[idx1] + (1 - lam) * features[idx2]
                        # 标签混合并取硬标签（argmax）
                        lab1_onehot = np.eye(num_classes)[labels[idx1]]
                        lab2_onehot = np.eye(num_classes)[labels[idx2]]
                        mixed_soft = lam * lab1_onehot + (1 - lam) * lab2_onehot
                        mixed_hard = np.argmax(mixed_soft, axis=-1)  # (L,)
                        # 添加到结果列表，depth_mapping为None表示混合样本
                        mixup_list.append((mixed_feat, mixed_hard, len(mixed_feat), -1, None))
                        mixup_count[idx1] += 1
                        mixup_count[idx2] += 1
                        generated += 1
                        if generated >= region_target:
                            break
        return mixup_list
    
    def __getitem__(self, idx):
        feat, label, length, orig_idx, _ = self.samples[idx]
        return (torch.tensor(feat, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                length,
                orig_idx)
    
    def __len__(self):
        return len(self.samples)

def collate_fn(batch):
    features, labels, lengths, indices = zip(*batch)
    features_padded = pad_sequence(features, batch_first=True)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
    # return features_padded, labels_padded, torch.tensor(lengths)
    return features_padded, labels_padded, torch.tensor(lengths), torch.tensor(indices)


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_acc = 0
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_acc, model):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f'验证集 accuracy improved to {val_acc:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f"验证集 accuracy not improved ({val_acc:.4f}). Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='/disk6/xie_rh/Rwork/model/OtherFile/hrnet_Config.yaml')

    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam',action='store_true')
    parser.add_argument('--image',type=str)
    parser.add_argument('--write',action='store_true')
    parser.add_argument('--showFps',action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    # args = parser.parse_args()

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def visualize_single_run_results(test_results, well_names, class_names, output_dir='visualization_results'):
    """
    可视化单次运行的所有测试集切片结果（无 Fold 层级）

    参数:
        test_results: 列表，每个元素包含 {'well_idx': int, 'true': array, 'pred': array, 'accuracy': float}
        well_names: 井名列表
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    # 按井分组
    well_segments = {}
    for result in test_results:
        well_idx = result['well_idx']
        if well_idx not in well_segments:
            well_segments[well_idx] = []
        well_segments[well_idx].append(result)

    # 创建输出目录（去掉 Fold_X 层级）
    os.makedirs(output_dir, exist_ok=True)

    # 为每口井创建单独的文件夹并生成图片
    for well_idx, segments in well_segments.items():
        well_name = well_names[well_idx]
        well_dir = os.path.join(output_dir, well_name)
        os.makedirs(well_dir, exist_ok=True)

        print(f"\n正在生成 {well_name} 的可视化结果 ({len(segments)} 个片段)...")

        for seg_idx, result in enumerate(segments):
            segment_num = seg_idx + 1
            filename = f'{well_name}_Segment_{segment_num}.png'
            save_path = os.path.join(well_dir, filename)

            plot_segment_comparison(
                true_labels=result['true'],
                pred_labels=result['pred'],
                class_names=class_names,
                well_name=well_name,
                segment_idx=segment_num,
                save_path=save_path,
                segment_accuracy=result['accuracy']
            )

        print(f"  ✓ {well_name} 的 {len(segments)} 个片段已保存至: {well_dir}")

# 在训练循环外定义噪声强度衰减函数
def get_noise_strength(epoch, max_epochs, initial_strength=0.5, final_strength=0.1):
    """噪声强度随训练轮数线性衰减"""
    return initial_strength - (initial_strength - final_strength) * (epoch / max_epochs)

# region mixup
# ====================== 增强配置（可直接调参） ======================
# 基础增强开关
AUG_PROB = 0.5                  # 每个增强的执行概率
DROP_CHANNEL_MAX = 2            # 最多随机丢弃几条测井曲线
GAUSSIAN_NOISE_STD = 0.005      # 原始空间高斯噪声强度（极小值）
FEATURE_SCALE_RANGE = (0.98, 1.02)  # 特征缩放范围（极小范围）

# MixUp增强配置
MIXUP_PROB = 0.4                # MixUp执行概率（不要超过0.5）
MIXUP_BETA_ALPHA = 0.1          # Beta分布参数（越小混合越温和）
SAME_CLASS_RATIO = 0.7          # 同类别配对占比
ADJACENT_CLASS_RATIO = 0.3      # 相邻类别配对占比
CROP_RATIO = 1/6                # 裁剪掉首尾的比例（中心区域混合）
# ===================================================================

def augment_well_features(features, aug_prob=AUG_PROB):
    """
    基础原始空间增强（仅训练阶段使用）
    只保留最安全、最有效的增强方式
    """
    aug_feat = features.copy()
    seq_len, feat_dim = aug_feat.shape

    # 1. 特征通道随机丢弃（效果最好，必加）
    if np.random.rand() < aug_prob:
        num_drop = np.random.randint(1, DROP_CHANNEL_MAX + 1)
        drop_channels = np.random.choice(feat_dim, num_drop, replace=False)
        aug_feat[:, drop_channels] = 0.0

    # 2. 切片内随机裁剪（解决切片边界效应）
    if np.random.rand() < aug_prob and seq_len > 200:
        crop_len = np.random.randint(seq_len - 20, seq_len + 1)
        start = np.random.randint(0, seq_len - crop_len + 1)
        cropped = aug_feat[start:start+crop_len]
        pad_left = start
        pad_right = seq_len - start - crop_len
        aug_feat = np.pad(cropped, ((pad_left, pad_right), (0, 0)), mode='edge')

    # 3. 轻微高斯噪声（可选）
    if np.random.rand() < aug_prob:
        noise = np.random.normal(0, GAUSSIAN_NOISE_STD, aug_feat.shape)
        aug_feat += noise

    # 4. 特征幅值缩放（可选）
    if np.random.rand() < aug_prob:
        scale = np.random.uniform(*FEATURE_SCALE_RANGE, (1, feat_dim))
        aug_feat *= scale

    return aug_feat


# endregion 
# endregion


def get_model(input_size, num_classes, device, modelType = MODEL_NAME):
    model = None
    print(f"使用的是: {modelType}")
    match modelType:
        case "OCMTCN":
            model = get_ocmtcn_net(input_size=input_size, num_classes=num_classes, seq_len=WINDOW_SIZE).to(device)
    return model, modelType


def main():
    dataTool = DataTools1(WINDOW_SIZE, STRIDE)

    (train_wells_features, train_wells_labels,
     test_wells_features, test_wells_labels,
     val_wells_features, val_wells_labels,
     num_classes, scaler, la_encoder, li_encoder,
     train_well_ids, val_well_ids, test_well_ids, well_names,
     train_start_indices, val_start_indices, test_start_indices) = (
        dataTool.load_data_seed(FOLDER_PATH, SELECTED_FEATURE_COLUMNS, 0, SEED))

    print(f"训练集样本数: {len(train_wells_features)} | 验证集样本数: {len(val_wells_features)} | 测试集样本数: {len(test_wells_features)}")
    print(f"总类别: {num_classes}")
    print(f"井名列表: {well_names}")
    print(f"井的数量: {len(well_names)}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    input_size = train_wells_features[0].shape[1]

    # 获取原始类别名称
    class_names = la_encoder.classes_.tolist()
    print(f"类别名称: {class_names}")

    # 提取验证集和训练集的井名（去重）
    val_well_ids_unique = list(set(val_well_ids))
    val_well_names = [well_names[wid] for wid in sorted(val_well_ids_unique)]
    train_well_ids_unique = list(set(train_well_ids))
    train_well_names = [well_names[wid] for wid in sorted(train_well_ids_unique)]
    test_well_ids_unique = list(set(test_well_ids))
    test_well_names = [well_names[wid] for wid in sorted(test_well_ids_unique)]

    print(f"训练集井数: {len(train_well_names)}, 验证集井数: {len(val_well_names)}")
    print(f"验证集井名: {val_well_names}")

    # 先创建固定 seed 的生成器
    g = torch.Generator()
    g.manual_seed(SEED)

    # region 创建新的Dataset 和 Loader
    # 创建 Dataset 和 Loader
    train_ds = WellDataset(train_wells_features, train_wells_labels, train=IS_PLUS, well_ids=train_well_ids)
    val_ds = WellDataset(val_wells_features, val_wells_labels, train=False, well_ids=val_well_ids)
    test_ds = WellDataset(test_wells_features, test_wells_labels, train=False, well_ids=test_well_ids)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, 
        generator=g
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn, 
        generator=g
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )

    # endregion

    # 创建模型
    model, modelType = get_model(input_size, num_classes, device)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.Adam([
        {'params': model.parameters()}
    ], lr=0.001)
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0, verbose=True)

    

    # 训练
    for epoch in range(EPOCHS):
        # 初始化CUDA事件（记录GPU时间）
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()  # 记录开始
        model.train()
        total_loss = 0

           # 获取当前轮数的噪声强度
        # current_noise_strength = get_noise_strength(epoch, EPOCHS)

        for feat, lab, lens, i in train_loader:
            feat, lab = feat.to(device), lab.to(device)

            # 关键检查！
            max_label = lab.max().item()
            if max_label >= num_classes or max_label < 0:
                print(f"非法标签！ max={max_label}, num_classes={num_classes}")
                print("当前 batch 标签唯一值：", torch.unique(lab))
                raise ValueError("标签超出范围！")

            optimizer.zero_grad()
            out, projected_features, features = model(feat, lens)
            # 交叉熵损失
            loss_main = criterion(out.contiguous().view(-1, num_classes), lab.view(-1))
            seq_len = WINDOW_SIZE

            # endregion
            # 计算自适应损失
            if IS_MUTI:
                loss_main, loss_dict = model.compute_adaptive_loss(
                    out, projected_features, lab, loss_main, LossFuncs.multiscale_temporal_smoothness_loss
                )

            # 逆序损失
            order_tensor = torch.arange(num_classes, dtype=torch.float).to(device)
            order_loss = LossFuncs.inverse_order_loss(out, order_tensor)
            # CRF损失（主要损失）
            # crf_loss = model.crf_loss(out, lab)
            # 时间平滑损失
            # smooth_loss = LossFuncs.multiscale_temporal_smoothness_loss(out)
            # 总损失
            total_loss_batch = None
            if IS_ADD_LOSS:
                total_loss_batch = loss_main + lambda_smooth * order_loss   
            else:
                total_loss_batch = loss_main 
            total_loss_batch.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += total_loss_batch.item()

        # 验证集评估
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for feat, lab, lens, i in val_loader:
                feat = feat.to(device)
                out, _ , _= model(feat, lens)
                # 使用CRF解码得到最优路径（而不是argmax）
                # pred_sequences = model.crf_decode(out)
                # # 将pred_sequences转换为numpy数组
                # pred = np.array(pred_sequences)
                pred = torch.argmax(out, dim=2).cpu().numpy()
                for i in range(len(pred)):
                    valid_len = lens[i].item()
                    val_preds.extend(pred[i][:valid_len])
                    val_true.extend(lab[i][:valid_len].cpu().numpy())

        val_acc = accuracy_score(val_true, val_preds)

        # 计算并显示每个类别的准确率（原始预测）
        print(f"\n--- 原始预测结果 ---")
        # per_class_metrics, macro_acc = AccuracyTools.calculate_per_class_accuracy(
        #     val_true, val_preds, num_classes, class_names=class_names
        # )

        end_event.record()    # 记录结束
        torch.cuda.synchronize()  # 等待GPU操作完成（必须加，否则时间不准）

        epoch_time = start_event.elapsed_time(end_event) / 1000  # 转成秒
        print(f"Epoch {epoch+1}/{EPOCHS} --- Loss: {total_loss/len(train_loader):.4f}, GPU耗时: {epoch_time:.2f}秒")

        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("触发早停")
            model.load_state_dict(early_stopping.best_model_state)
            break

    # 使用早停恢复的最佳模型进行最终评估
    model.load_state_dict(early_stopping.best_model_state)
    model.eval()



    # 在测试集上评估最佳模型
    final_test_preds, final_test_true = [], []
    test_segment_results = []  # 存储每个切片的详细信息

    with torch.no_grad():
        for feat, lab, lens, indices in test_loader:
            feat = feat.to(device)
            out, _, _ = model(feat, lens)
            pred = torch.argmax(out, dim=2).cpu().numpy()

            for i in range(len(pred)):
                valid_len = lens[i].item()
                sample_pred = pred[i][:valid_len]
                sample_true = lab[i][:valid_len].cpu().numpy()
                final_test_preds.extend(sample_pred)
                final_test_true.extend(sample_true)

                # 计算该切片的准确率
                seg_acc = accuracy_score(sample_true, sample_pred)
                # 获取该样本对应的井ID（使用 test_well_ids）
                sample_well_id = test_well_ids[indices[i].item()]
                # 收集切片级结果
                test_segment_results.append({
                    'well_idx': sample_well_id,
                    'true': sample_true,
                    'pred': sample_pred,
                    'accuracy': seg_acc
                })

    # 计算测试集的各类别指标
    print(f"\n--- 测试集评估结果 ---")
    test_metrics, test_acc, test_pre, test_recall, test_f1, kappa = AccuracyTools.calculate_per_class_accuracy(
        final_test_true, final_test_preds, num_classes, class_names=class_names
    )


    # 打印测试集各类别详细指标
    print("\n" + "="*60)
    print("测试集各类别详细指标:")
    print("="*60)
    for class_name, metrics in test_metrics.items():
        print(f"{class_name}:")
        print(f"  F1-Score:  {metrics['f1-score']:.4f}")
        print(f"  Support:   {metrics['support']}")
    print("="*60)

    # 生成可视化结果
    if IS_PLIANT:
        print(f"\n正在生成可视化结果...")
        visualize_single_run_results(
            test_results=test_segment_results,
            well_names=well_names,
            class_names=class_names,
            output_dir=PIC_OUT_PATH
        )
        print(f"✓ 可视化完成！")

    # 打印数据集信息
    print("\n" + "="*80)
    print("数据集详细信息:")
    print("="*80)
    print(f"训练集井数: {len(train_well_names)}")
    print(f"训练集井名: {train_well_names}")
    print(f"验证集井数: {len(val_well_names)}")
    print(f"验证集井名: {val_well_names}")
    test_well_ids_unique = list(set(test_well_ids))
    test_well_names = [well_names[wid] for wid in sorted(test_well_ids_unique)]
    print(f"测试集井数: {len(test_well_names)}")
    print(f"测试集井名: {test_well_names}")
    print("="*80)

    print(f"测试集准确率: {test_acc:.4f}--- 测试集精准率: {test_pre:.4f}--- 测试集召回率: {test_recall:.4f}--- 测试集F1: {test_f1:.4f}--- 测试集kappa系数: {kappa:.4f}---验证集准确率: {early_stopping.best_acc:.4f}")

    return test_acc, test_recall, test_f1, test_metrics



if __name__ == "__main__":
    set_seed()
    main()
    print(SEED)
    print(MODEL_NAME)
    print(CONTENT)
    # CUDA_VISIBLE_DEVICES=0 python main.py
