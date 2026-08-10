import os
import time

import joblib
import yaml

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class DataTools:
    @staticmethod
    def load_data_from_folder(folder_path, selected_feature_columns, window_size, stride):
        features_list = []
        labels_list = []
        scaler = StandardScaler()
        label_encoder = LabelEncoder()  # 用于主标签
        lithology_encoder = LabelEncoder()  # 用于岩性（如果非数值）

        all_feature_cols = None

        # 先收集所有标签用于全局编码
        all_raw_labels = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="valid_data")
                if '分层' in df.columns:
                    all_raw_labels.extend(df['分层'].astype(str).dropna().tolist())

        if all_raw_labels:
            label_encoder.fit(all_raw_labels)
            print("全局标签类别：", label_encoder.classes_)
            print("总类别数：", len(label_encoder.classes_))

        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_excel(file_path, sheet_name="valid_data")
                if '岩性' in df.columns:
                    all_lithology.extend(df['岩性'].dropna().astype(str).unique())
        if all_lithology:
            lithology_encoder.fit(all_lithology)

        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_excel(file_path, sheet_name="valid_data")

                print(f"Processing {file_name}, columns: {df.columns.tolist()}")

                if '分层' not in df.columns:
                    print(f"Warning: '分层' column not found in {file_name}, skipping.")
                    continue

                # 提取标签
                labels = df['分层'].astype(str).values
                # labels = label_encoder.fit_transform(labels)  # 字符串转整数
                labels = label_encoder.transform(labels)
                # 确定特征列（排除深度和分层）
                exclude_cols = ['分层']
                if selected_feature_columns is not None:
                    feature_cols = [col for col in selected_feature_columns if col in df.columns]
                else:
                    feature_cols = [col for col in df.columns if col not in exclude_cols]

                if not feature_cols:
                    print(f"No feature columns found in {file_name}, skipping.")
                    continue

                # 分离数值特征和岩性（如果存在）
                num_cols = [col for col in feature_cols if col != '岩性']
                lith_col = '岩性' if '岩性' in feature_cols else None

                num_features = df[num_cols].fillna(0).values.astype(np.float32) if num_cols else np.empty((len(df), 0), dtype=np.float32)
                # num_features = scaler.fit_transform(num_features)
                # num_features = scaler.transform(num_features)

                if lith_col:
                    lith_values = df[lith_col].fillna('未知').astype(str)
                    lith_encoded = lithology_encoder.transform(lith_values).reshape(-1, 1).astype(np.float32)
                    features = np.hstack([num_features, lith_encoded]) if num_features.size > 0 else lith_encoded
                else:
                    features = num_features

                # 提取特征和标签（保持您当前的逻辑，包括岩性）
                # 假设 features 已提取为 (seq_len, feat_dim) 的 np.array
                # labels 为 (seq_len,) 的 np.array

                seq_len = len(features)
                starts = list(range(0, seq_len - window_size + 1, stride))

                # 如果最后一段不足窗口大小，也要包含
                if seq_len > 0 and (not starts or starts[-1] + window_size < seq_len):
                    last_start = max(0, seq_len - window_size)  # 从末尾向前取 WINDOW_SIZE
                    if last_start not in starts:  # 避免重复
                        starts.append(last_start)

                for start in starts:
                    end = min(start + window_size, seq_len)  # 防止越界
                    chunk_feat = features[start:end]
                    chunk_lab = labels[start:end]

                    # padding 到 WINDOW_SIZE（如果需要模型输入固定长度）
                    pad_len = window_size - (end - start)
                    if pad_len > 0:
                        chunk_feat = np.pad(chunk_feat, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                        chunk_lab = np.pad(chunk_lab, (0, pad_len), mode='constant', constant_values=-1)  # -1 用于 ignore_index

                    # 标准化（每个窗口独立，或全局？推荐全局先 fit 所有再 transform）
                    chunk_feat = scaler.fit_transform(chunk_feat)

                    features_list.append(chunk_feat)
                    labels_list.append(chunk_lab)

                # features_list.append(features)
                # labels_list.append(labels)

                if all_feature_cols is None:
                    all_feature_cols = num_cols + (['岩性'] if lith_col else [])

        num_classes = len(label_encoder.classes_)
        print(f"Loaded {len(features_list)} wells, feature dim: {features.shape[1] if features_list else 0}, classes: {num_classes}")
        print(f"Used feature columns: {all_feature_cols}")

        return features_list, labels_list, num_classes, scaler, label_encoder, lithology_encoder

    @staticmethod
    def load_test_data_from_folder(df, selected_feature_columns, window_size, stride, scaler, layer_encode, lithology_encoder):
        features_list = []
        labels_list = []
        all_feature_cols = None


        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        if '岩性' in df.columns:
            all_lithology.extend(df['岩性'].dropna().astype(str).unique())
        if all_lithology:
            lithology_encoder.transform(all_lithology)

        # if '分层' not in df.columns:
        #     print(f"Warning: '分层' column not found in {file_name}, skipping.")


        # 确定特征列（排除深度和分层）
        exclude_cols = ['分层']
        if selected_feature_columns is not None:
            feature_cols = [col for col in selected_feature_columns if col in df.columns]
        else:
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        # 分离数值特征和岩性（如果存在）
        num_cols = [col for col in feature_cols if col != '岩性']
        lith_col = '岩性' if '岩性' in feature_cols else None

        num_features = df[num_cols].fillna(0).values.astype(np.float32) if num_cols else np.empty((len(df), 0), dtype=np.float32)


        if lith_col:
            lith_values = df[lith_col].fillna('未知').astype(str)
            lith_encoded = lithology_encoder.transform(lith_values).reshape(-1, 1).astype(np.float32)
            features = np.hstack([num_features, lith_encoded]) if num_features.size > 0 else lith_encoded
        else:
            features = num_features

        # 提取特征和标签（保持您当前的逻辑，包括岩性）
        # 假设 features 已提取为 (seq_len, feat_dim) 的 np.array
        # labels 为 (seq_len,) 的 np.array

        seq_len = len(features)
        starts = list(range(0, seq_len - window_size + 1, stride))

        # 如果最后一段不足窗口大小，也要包含
        if seq_len > 0 and (not starts or starts[-1] + window_size < seq_len):
            last_start = max(0, seq_len - window_size)  # 从末尾向前取 WINDOW_SIZE
            if last_start not in starts:  # 避免重复
                starts.append(last_start)

        for start in starts:
            end = min(start + window_size, seq_len)  # 防止越界
            chunk_feat = features[start:end]
            # chunk_lab = labels[start:end]

            # padding 到 WINDOW_SIZE（如果需要模型输入固定长度）
            pad_len = window_size - (end - start)
            if pad_len > 0:
                chunk_feat = np.pad(chunk_feat, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                # chunk_lab = np.pad(chunk_lab, (0, pad_len), mode='constant', constant_values=-1)  # -1 用于 ignore_index

            # 标准化（每个窗口独立，或全局？推荐全局先 fit 所有再 transform）
            chunk_feat = scaler.fit_transform(chunk_feat)

            features_list.append(chunk_feat)
            # labels_list.append(chunk_lab)

        return features_list

    @staticmethod
    def collate_fn(batch):
        features, labels, lengths = zip(*batch)
        features_padded = pad_sequence(features, batch_first=True)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-1)
        return features_padded, labels_padded, torch.tensor(lengths)

class WellDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), \
            torch.tensor(self.labels[idx], dtype=torch.long), \
            len(self.features[idx])



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
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(f'Validation accuracy improved to {val_acc:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation accuracy not improved ({val_acc:.4f}). Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")


# config_path = r"D:\RT\Code\model\OtherFile\config.yaml"
config_path = r"/disk6/xie_rh/Rwork/model/OtherFile/config.yaml"
class ConfigTools:
    @staticmethod
    def load_config_data(field_name: str):
        # 1. 打开并读取YAML文件
        with open(config_path, 'r', encoding='utf-8') as f:
            # Loader=yaml.FullLoader 避免安全警告
            config = yaml.load(f, Loader=yaml.FullLoader)

        # 2. 读取配置项
        result = config[field_name]
        return result


class ModelTools:
    @staticmethod
    def save_model(scaler, layer_encoder, lithology_encoder, model, model_name):
        SAVE_DIR = time.strftime("%Y%m%d%H%M%S", time.localtime())
        SAVE_DIR = SAVE_DIR + "_" + model_name
        os.makedirs(SAVE_DIR, exist_ok=True)
        scaler_path = os.path.join(SAVE_DIR, "feature_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        le_path = os.path.join(SAVE_DIR, "layer_encoder.joblib")
        joblib.dump(layer_encoder, le_path)
        li_path = os.path.join(SAVE_DIR, "lithology_encoder.joblib")
        joblib.dump(lithology_encoder, li_path)
        model_save_path = os.path.join(SAVE_DIR, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"模型和工具已保存到 {SAVE_DIR}")
        


class DataProcess:
    @staticmethod
    def smooth_and_fill_segments(preds, window_size=5):
        """
        # 新增：后处理 - 平滑 + 边界填充
        后处理：多数投票平滑 + 强制段内连续填充
        preds: 原始逐点预测的层号列表
        """
        preds = np.array(preds)

        # Step 1: 多数投票平滑（去除孤立噪声点）
        smoothed = preds.copy()
        for i in range(window_size//2, len(preds) - window_size//2):
            window = preds[i - window_size//2 : i + window_size//2 + 1]
            smoothed[i] = np.bincount(window).argmax()  # 取窗口内出现最多的层号

        # Step 2: 强制段内连续（从左到右填充）
        final_segments = smoothed.copy()
        current_layer = final_segments[0]
        for i in range(1, len(final_segments)):
            if final_segments[i] != current_layer:
                # 检测到边界 → 从这里开始新层
                current_layer = final_segments[i]
            else:
                # 段内强制一致（可选：可注释掉，如果想保留原始预测）
                final_segments[i] = current_layer

        return final_segments

