import os
from datetime import time

import joblib
import yaml

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin

class GeoLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, order_list):
        self.order_list = order_list
        self.mapping_ = None
        self.inverse_mapping_ = None

    def fit(self, y, **fit_params):
        unique = np.unique(y)
        print(f"  {unique}")
        missing = set(unique) - set(self.order_list)
        if missing:
            raise ValueError(f"类别 {missing} 不在 order_list 中")
        self.mapping_ = {label: idx for idx, label in enumerate(self.order_list)}
        self.inverse_mapping_ = {idx: label for label, idx in self.mapping_.items()}
        self.classes_ = np.array(self.order_list)   # 添加 classes_ 属性以兼容
        return self

    def transform(self, y):
        return np.array([self.mapping_[label] for label in y])

    def fit_transform(self, y, **fit_params):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self.inverse_mapping_[idx] for idx in y])

class DataTools1:
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

    def slicer(self, features, labels):
        features_list = []
        labels_list = []

        seq_len = len(features)
        starts = list(range(0, seq_len - self.window_size + 1, self.stride))

        # 如果最后一段不足窗口大小，也要包含
        if seq_len > 0 and (not starts or starts[-1] + self.window_size < seq_len):
            last_start = max(0, seq_len - self.window_size)  # 从末尾向前取 WINDOW_SIZE
            if last_start not in starts:  # 避免重复
                starts.append(last_start)
        start_list = []
        for start in starts:
            if(start + self.window_size > seq_len):
                continue
            end = start + self.window_size
            chunk_feat = features[start:end]
            chunk_lab = labels[start:end]

            features_list.append(chunk_feat)
            labels_list.append(chunk_lab)

            start_list.append(start)

        return features_list, labels_list, start_list

    def load_data(self, folder_path, selected_feature_columns, scaler_model):
        """

        :param folder_path:
        :param selected_feature_columns:
        :param scaler_model: 0:训练集，验证集，测试集按井归一化；
        1：训练，验证集切片后合在一起归一化，测试集合在一起归一化；
        2：训练、验证合在一起归一化，测试集使用训练的归一化进行归一化
        :param window_size:
        :param stride:
        :return:
        """

        features_list = []
        labels_list = []
        well_features_list = []
        well_label_list = []
        scaler = StandardScaler()
        label_encoder = GeoLabelEncoder()  # 用于主标签
        lithology_encoder = LabelEncoder()  # 用于岩性（如果非数值）
        label_name = "分层"
        # label_name = "分层信息"
        # label_name = "切片地层标签"

        all_feature_cols = None

        # 先收集所有标签用于全局编码
        all_raw_labels = []
        all_dfs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="diff_smooth")
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="valid_data")
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="Sheet1")
                all_dfs.append(df)



        for df in all_dfs:
            if label_name in df.columns:
                all_raw_labels.extend(df[label_name].astype(str).dropna().tolist())

        if all_raw_labels:
            label_encoder.fit(all_raw_labels)
            print("全局标签类别：", label_encoder.classes_)
            print("总类别数：", len(label_encoder.classes_))

        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        for df in all_dfs:
            if '岩性' in df.columns:
                all_lithology.extend(df['岩性'].dropna().astype(str).unique())

        if all_lithology:
            lithology_encoder.fit(all_lithology)

        for df in all_dfs:
            print(f"Processing {file_name}, columns: {df.columns.tolist()}")

            if label_name not in df.columns:
                print(f"Warning: {label_name} column not found in {file_name}, skipping.")
                continue

            # 提取标签
            labels = df[label_name].astype(str).values
            # labels = label_encoder.fit_transform(labels)  # 字符串转整数
            labels = label_encoder.transform(labels)

            #region 不重要
            # 确定特征列（排除深度和分层）
            exclude_cols = [label_name]
            if selected_feature_columns is not None:
                feature_cols = [col for col in selected_feature_columns if col in df.columns]
            else:
                feature_cols = [col for col in df.columns if col not in exclude_cols]

            if not feature_cols:
                print(f"No feature columns found in {file_name}, skipping.")
                continue
            #endregion

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

            well_features_list.append(features)
            well_label_list.append(labels)

        train_wells_f, test_well_f, train_wells_l, test_well_l = train_test_split(well_features_list, well_label_list, test_size=0.1, random_state=42)

        if scaler_model == 2:
            all_feature = [well_feature for well_feature in train_wells_f]
            feature_all_temp = np.vstack(all_feature)
            feature_all_temp = scaler.fit_transform(feature_all_temp)

        train_wells_f, val_well_f, train_wells_l, val_well_l = train_test_split(train_wells_f, train_wells_l, test_size=0.2, random_state=42)

        train_wells_features = []
        train_wells_labels = []
        test_wells_features = []
        test_wells_labels = []
        val_wells_features = []
        val_wells_labels = []

        for feature, label in zip(train_wells_f, train_wells_l):

            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels = self.slicer(feature, label)
                train_wells_features.extend(features)
                train_wells_labels.extend(labels)
                # for _feature, _label in zip(features, labels):
                #     train_wells_features.append(_feature)
                #     train_wells_labels.append(_label)
            elif scaler_model == 1:
                # case1：窗口自己归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    train_wells_features.append(_feature)
                    train_wells_labels.append(_label)
            else:
                # case1：统一归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    train_wells_features.append(_feature)
                    train_wells_labels.append(_label)


        for feature, label in zip(val_well_f, val_well_l):
            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels = self.slicer(feature, label)
                val_wells_features.extend(features)
                val_wells_labels.extend(labels)
            elif scaler_model == 1:
                # case1：先切片，组合起来进行归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    val_wells_features.append(_feature)
                    val_wells_labels.append(_label)
            else:
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    val_wells_features.append(_feature)
                    val_wells_labels.append(_label)

        for feature, label in zip(test_well_f, test_well_l):
            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels = self.slicer(feature, label)
                test_wells_features.extend(features)
                test_wells_labels.extend(labels)
            elif scaler_model == 1:
                # 窗口自己归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    test_wells_features.append(_feature)
                    test_wells_labels.append(_label)
            else:
                # 使用训练的归一化器
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    test_wells_features.append(_feature)
                    test_wells_labels.append(_label)



        num_classes = len(label_encoder.classes_)

        return (train_wells_features, train_wells_labels, test_wells_features, test_wells_labels,
                val_wells_features, val_wells_labels, num_classes, scaler, label_encoder, lithology_encoder)

    def load_data_seed(self, folder_path, selected_feature_columns, scaler_model, seed):
        """

        :param folder_path:
        :param selected_feature_columns:
        :param scaler_model: 0:训练集，验证集，测试集按井归一化；
        1：训练，验证集切片后合在一起归一化，测试集合在一起归一化；
        2：训练、验证合在一起归一化，测试集使用训练的归一化进行归一化
        :param window_size:
        :param stride:
        :return:
        """
        strata_order = ["IV-1", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6", "IV-7", "IV-8", "IV-9", "IV-10",
                        "IV-11", "IV-12","IV-13", "IV-14",]
        # strata_order = ["III", "IV", "V", "VI", "nan"]

        features_list = []
        labels_list = []
        well_features_list = []
        well_label_list = []
        scaler = StandardScaler()
        # label_encoder = GeoLabelEncoder(strata_order)  # 用于主标签
        label_encoder = LabelEncoder()  # 用于主标签
        lithology_encoder = LabelEncoder()  # 用于岩性（如果非数值）
        label_name = "分层"
        # label_name = "岩性_处理后"
        # label_name = "切片地层标签"

        lithology_col = "岩性_处理后"

        all_feature_cols = None

        # 先收集所有标签用于全局编码
        all_raw_labels = []
        all_dfs = []
        well_file_names = []  # 保存每个DataFrame对应的文件名
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="Sheet1")
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="valid_data")
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="Sheet1")
                all_dfs.append(df)
                well_file_names.append(file_name)  # 保存文件名，与df一一对应



        for df in all_dfs:
            if label_name in df.columns:
                all_raw_labels.extend(df[label_name].astype(str).dropna().tolist())

        if all_raw_labels:
            label_encoder.fit(all_raw_labels)
            print("全局标签类别：", label_encoder.classes_)
            print("总类别数：", len(label_encoder.classes_))

        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        for df in all_dfs:
            if lithology_col in df.columns:
                all_lithology.extend(df[lithology_col].dropna().astype(str).unique())

        if all_lithology:
            lithology_encoder.fit(all_lithology)

        for idx, df in enumerate(all_dfs):
            file_name = well_file_names[idx]  # 获取对应的文件名
            print(f"Processing {file_name}, columns: {df.columns.tolist()}")

            if label_name not in df.columns:
                print(f"Warning: {label_name} column not found in {file_name}, skipping.")
                continue

            # 提取标签
            labels = df[label_name].astype(str).values
            # labels = label_encoder.fit_transform(labels)  # 字符串转整数
            labels = label_encoder.transform(labels)

            #region 不重要
            # 确定特征列（排除深度和分层）
            exclude_cols = [label_name]
            if selected_feature_columns is not None:
                feature_cols = [col for col in selected_feature_columns if col in df.columns]
            else:
                feature_cols = [col for col in df.columns if col not in exclude_cols]

            if not feature_cols:
                print(f"No feature columns found in {file_name}, skipping.")
                continue
            #endregion

            # 分离数值特征和岩性（如果存在）
            num_cols = [col for col in feature_cols if col != lithology_col]
            lith_col = lithology_col if lithology_col in feature_cols else None

            num_features = df[num_cols].fillna(0).values.astype(np.float32) if num_cols else np.empty((len(df), 0), dtype=np.float32)

            if lith_col:
                lith_values = df[lith_col].fillna('定名').astype(str)
                lith_encoded = lithology_encoder.transform(lith_values).reshape(-1, 1).astype(np.float32)
                features = np.hstack([num_features, lith_encoded]) if num_features.size > 0 else lith_encoded
            else:
                features = num_features

            well_features_list.append(features)
            well_label_list.append(labels)

        # 在分割之前先提取井名（从文件名中去掉扩展名），与 well_features_list 的索引对应
        well_names = [os.path.splitext(fname)[0] for fname in well_file_names]
        print(f"井名列表: {well_names}")
        print(f"井的数量: {len(well_names)}")

        train_wells_f, test_well_f, train_wells_l, test_well_l = train_test_split(well_features_list, well_label_list, test_size=0.1, random_state=seed)

        if scaler_model == 2:
            all_feature = [well_feature for well_feature in train_wells_f]
            feature_all_temp = np.vstack(all_feature)
            feature_all_temp = scaler.fit_transform(feature_all_temp)

        train_wells_f, val_well_f, train_wells_l, val_well_l = train_test_split(train_wells_f, train_wells_l, test_size=0.2, random_state=seed)

        train_wells_features = []
        train_wells_labels = []
        test_wells_features = []
        test_wells_labels = []
        val_wells_features = []
        val_wells_labels = []

        train_well_ids = []
        val_well_ids = []
        test_well_ids = []

        train_start_indices = []
        val_start_indices = []
        test_start_indices = []

        # 需要找到每口井在原始 well_names 中的索引
        # 由于 train_test_split 打乱了顺序，需要通过对象引用来追踪

        for feature, label in zip(train_wells_f, train_wells_l):
            # 找到这口井在原始 well_features_list 中的索引
            # well_idx = well_features_list.index(feature)
            well_idx = -1
            for i, arr in enumerate(well_features_list):
                if id(arr) == id(feature):
                    well_idx = i
                    break

            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels, starts = self.slicer(feature, label)
                train_wells_features.extend(features)
                train_wells_labels.extend(labels)
                train_well_ids.extend([well_idx] * len(labels))

                train_start_indices.extend(starts)
            elif scaler_model == 1:
                # case1：窗口自己归一化
                features, labels, starts = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    train_wells_features.append(_feature)
                    train_wells_labels.append(_label)
                train_well_ids.extend([well_idx] * len(labels))
                train_start_indices.extend(starts)
            else:
                # case1：统一归一化
                features, labels, starts = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    train_wells_features.append(_feature)
                    train_wells_labels.append(_label)
                train_well_ids.extend([well_idx] * len(labels))
                train_start_indices.extend(starts)


        for feature, label in zip(val_well_f, val_well_l):
            well_idx = -1
            for i, arr in enumerate(well_features_list):
                if id(arr) == id(feature):
                    well_idx = i
                    break

            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels, starts = self.slicer(feature, label)
                val_wells_features.extend(features)
                val_wells_labels.extend(labels)
                val_well_ids.extend([well_idx] * len(labels))
                val_start_indices.extend(starts)
            elif scaler_model == 1:
                # case1：先切片，组合起来进行归一化
                features, labels, starts = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    val_wells_features.append(_feature)
                    val_wells_labels.append(_label)
                val_well_ids.extend([well_idx] * len(labels))
                val_start_indices.extend(starts)
            else:
                features, labels, starts = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    val_wells_features.append(_feature)
                    val_wells_labels.append(_label)
                val_well_ids.extend([well_idx] * len(labels))
                val_start_indices.extend(starts)

        for feature, label in zip(test_well_f, test_well_l):
            well_idx = -1
            for i, arr in enumerate(well_features_list):
                if id(arr) == id(feature):
                    well_idx = i
                    break

            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels, starts = self.slicer(feature, label)
                test_wells_features.extend(features)
                test_wells_labels.extend(labels)
                test_well_ids.extend([well_idx] * len(labels))
                test_start_indices.extend(starts)
            elif scaler_model == 1:
                # 窗口自己归一化
                features, labels, starts = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    test_wells_features.append(_feature)
                    test_wells_labels.append(_label)
                test_well_ids.extend([well_idx] * len(labels))
                test_start_indices.extend(starts)
            else:
                # 使用训练的归一化器
                features, labels, starts = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    test_wells_features.append(_feature)
                    test_wells_labels.append(_label)
                test_well_ids.extend([well_idx] * len(labels))
                test_start_indices.extend(starts)



        num_classes = len(label_encoder.classes_)

        return (train_wells_features, train_wells_labels, test_wells_features, test_wells_labels,
                val_wells_features, val_wells_labels, num_classes, scaler, label_encoder, lithology_encoder,
                train_well_ids, val_well_ids, test_well_ids, well_names,
                train_start_indices, val_start_indices,test_start_indices)


    def load_data_k(self, folder_path, selected_feature_columns, scaler_model):
        """

        :param folder_path:
        :param selected_feature_columns:
        :param scaler_model:
        0:训练集，验证集，测试集按井归一化；
        1：训练，验证集切片后合在一起归一化，测试集合在一起归一化；
        2：训练、验证合在一起归一化，测试集使用训练的归一化进行归一化
        :param window_size:
        :param stride:
        :return:
        """
        strata_order = ["IV-1", "IV-2", "IV-3", "IV-4", "IV-5", "IV-6", "IV-7", "IV-8", "IV-9", "IV-10",
                        "IV-11", "IV-12","IV-13", "IV-14",]

        features_list = []
        labels_list = []
        well_features_list = []
        well_label_list = []
        scaler = StandardScaler()
        label_encoder = GeoLabelEncoder(strata_order)  # 用于主标签
        lithology_encoder = LabelEncoder()  # 用于岩性（如果非数值）
        label_name = "分层"
        # label_name = "分层信息"
        # label_name = "切片地层标签"

        all_feature_cols = None

        # 先收集所有标签用于全局编码
        all_raw_labels = []
        all_dfs = []
        well_file_names = []  # 保存每个DataFrame对应的文件名
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="processed_data_layer")
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="extract_effective_value")
                df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="Sheet1")
                all_dfs.append(df)
                well_file_names.append(file_name)  # 保存文件名，与df一一对应



        for df in all_dfs:
            if label_name in df.columns:
                all_raw_labels.extend(df[label_name].astype(str).dropna().tolist())

        if all_raw_labels:
            label_encoder.fit(all_raw_labels)
            print("全局标签类别：", label_encoder.classes_)
            print("总类别数：", len(label_encoder.classes_))

        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        for df in all_dfs:
            if '岩性' in df.columns:
                all_lithology.extend(df['岩性'].dropna().astype(str).unique())

        if all_lithology:
            lithology_encoder.fit(all_lithology)

        for idx, df in enumerate(all_dfs):
            file_name = well_file_names[idx]  # 获取对应的文件名
            print(f"Processing {file_name}, columns: {df.columns.tolist()}")

            if label_name not in df.columns:
                print(f"Warning: {label_name} column not found in {file_name}, skipping.")
                continue

            # 提取标签
            labels = df[label_name].astype(str).values
            # labels = label_encoder.fit_transform(labels)  # 字符串转整数
            labels = label_encoder.transform(labels)

            #region 不重要
            # 确定特征列（排除深度和分层）
            exclude_cols = [label_name]
            if selected_feature_columns is not None:
                feature_cols = [col for col in selected_feature_columns if col in df.columns]
            else:
                feature_cols = [col for col in df.columns if col not in exclude_cols]

            if not feature_cols:
                print(f"No feature columns found in {file_name}, skipping.")
                continue
            #endregion

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

            well_features_list.append(features)
            well_label_list.append(labels)

        train_wells_f, test_well_f, train_wells_l, test_well_l = train_test_split(well_features_list, well_label_list, test_size=0.2, random_state=42)

        if scaler_model == 2:
            all_feature = [well_feature for well_feature in well_features_list]
            feature_all_temp = np.vstack(all_feature)
            feature_all_temp = scaler.fit_transform(feature_all_temp)

        all_feature = []
        all_label = []
        well_ids = []

        # 提取井名（从文件名中去掉扩展名），与 well_features_list 的索引对应
        well_names = [os.path.splitext(fname)[0] for fname in well_file_names]
        print(f"井名列表: {well_names}")
        print(f"井的数量: {len(well_names)}")

        for i, (feature, label) in enumerate(zip(well_features_list, well_label_list)):

            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels = self.slicer(feature, label)
                all_feature.extend(features)
                all_label.extend(labels)
                well_ids.extend([i] * len(labels))
            elif scaler_model == 1:
                # case1：窗口自己归一化
                features, labels = self.slicer(feature, label)
                well_ids.extend([i] * len(labels))
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    all_feature.append(_feature)
                    all_label.append(_label)
            else:
                # case1：统一归一化
                features, labels = self.slicer(feature, label)
                well_ids.extend([i] * len(labels))
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    all_feature.append(_feature)
                    all_label.append(_label)

        num_classes = len(label_encoder.classes_)
        return (all_feature, all_label,
                num_classes, scaler, label_encoder, lithology_encoder,
                well_ids, well_names)   # ← 新增返回值


    def load_test_data(self, df, selected_feature_columns, window_size, stride, scaler, lithology_encoder, scaler_model):
        """

        :param df:
        :param selected_feature_columns:
        :param window_size:
        :param stride:
        :param scaler:
        :param lithology_encoder:
        :param scaler_model: 0:先以井进行归一化，在切片
        1：先切片，再进行归一化
        2：先切片，再用训练的scaler进行归一化
        :return:
        """
        features_list = []
        label_name = "分层"

        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        if '岩性' in df.columns:
            all_lithology.extend(df['岩性'].dropna().astype(str).unique())
        if all_lithology:
            lithology_encoder.transform(all_lithology)

        # 确定特征列（排除深度和分层）
        exclude_cols = [label_name]
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

        if scaler_model == 0:
            features = scaler.fit_transform(features)

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
            if scaler_model == 1:
                chunk_feat = scaler.fit_transform(chunk_feat)

            features_list.append(chunk_feat)
            # labels_list.append(chunk_lab)

        return features_list

    def load_data_for_feature(self, folder_path, selected_feature_columns, scaler_model):
        """

        :param folder_path:
        :param selected_feature_columns:
        :param scaler_model: 0:训练集，验证集，测试集按井归一化；
        1：训练，验证集切片后合在一起归一化，测试集合在一起归一化；
        2：训练、验证合在一起归一化，测试集使用训练的归一化进行归一化
        :param window_size:
        :param stride:
        :return:
        """

        features_list = []
        labels_list = []
        well_features_list = []
        well_label_list = []
        scaler = StandardScaler()
        label_encoder = LabelEncoder()  # 用于主标签
        lithology_encoder = LabelEncoder()  # 用于岩性（如果非数值）
        label_name = "分层"
        # label_name = "分层信息"
        # label_name = "切片地层标签_1"

        all_feature_cols = None

        # 先收集所有标签用于全局编码
        all_raw_labels = []
        all_dfs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="valid_data")
                # df = pd.read_excel(os.path.join(folder_path, file_name), sheet_name="Sheet1")
                all_dfs.append(df)



        for df in all_dfs:
            if label_name in df.columns:
                all_raw_labels.extend(df[label_name].astype(str).dropna().tolist())

        if all_raw_labels:
            label_encoder.fit(all_raw_labels)
            print("全局标签类别：", label_encoder.classes_)
            print("总类别数：", len(label_encoder.classes_))

        # 先收集所有岩性以全局编码（确保一致）
        all_lithology = []
        for df in all_dfs:
            if '岩性' in df.columns:
                all_lithology.extend(df['岩性'].dropna().astype(str).unique())

        if all_lithology:
            lithology_encoder.fit(all_lithology)

        for df in all_dfs:
            print(f"Processing {file_name}, columns: {df.columns.tolist()}")

            if label_name not in df.columns:
                print(f"Warning: {label_name} column not found in {file_name}, skipping.")
                continue

            # 提取标签
            labels = df[label_name].astype(str).values
            # labels = label_encoder.fit_transform(labels)  # 字符串转整数
            labels = label_encoder.transform(labels)

            #region 不重要
            # 确定特征列（排除深度和分层）
            exclude_cols = [label_name]
            if selected_feature_columns is not None:
                feature_cols = [col for col in selected_feature_columns if col in df.columns]
            else:
                feature_cols = [col for col in df.columns if col not in exclude_cols]

            if not feature_cols:
                print(f"No feature columns found in {file_name}, skipping.")
                continue
            #endregion

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

            well_features_list.append(features)
            well_label_list.append(labels)

        train_wells_f, test_well_f, train_wells_l, test_well_l = train_test_split(well_features_list, well_label_list, test_size=0.2, random_state=42)

        if scaler_model == 2:
            all_feature = [well_feature for well_feature in train_wells_f]
            feature_all_temp = np.vstack(all_feature)
            feature_all_temp = scaler.fit_transform(feature_all_temp)


        train_wells_features = []
        train_wells_labels = []
        test_wells_features = []
        test_wells_labels = []
        group = []
        count = 0

        for feature, label in zip(train_wells_f, train_wells_l):

            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels = self.slicer(feature, label)
                train_wells_features.extend(features)
                train_wells_labels.extend(labels)
                for n in range(0, len(features)):
                    group.append(count)

            elif scaler_model == 1:
                # case1：窗口自己归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    train_wells_features.append(_feature)
                    train_wells_labels.append(_label)

                for n in range(0, len(features)):
                    group.append(count)
            else:
                # case1：统一归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    train_wells_features.append(_feature)
                    train_wells_labels.append(_label)

                for n in range(0, len(features)):
                    group.append(count)

        for feature, label in zip(test_well_f, test_well_l):
            if scaler_model == 0:
                # case1：先对井进行归一化，再切片
                feature = scaler.fit_transform(feature)
                features, labels = self.slicer(feature, label)
                test_wells_features.extend(features)
                test_wells_labels.extend(labels)
            elif scaler_model == 1:
                # 窗口自己归一化
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.fit_transform(_feature)
                    test_wells_features.append(_feature)
                    test_wells_labels.append(_label)
            else:
                # 使用训练的归一化器
                features, labels = self.slicer(feature, label)
                for _feature, _label in zip(features, labels):
                    _feature = scaler.transform(_feature)
                    test_wells_features.append(_feature)
                    test_wells_labels.append(_label)



        num_classes = len(label_encoder.classes_)

        return train_wells_features, train_wells_labels, test_wells_features, test_wells_labels, num_classes, scaler, label_encoder, lithology_encoder, group

