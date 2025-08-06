from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import logging


def create_datasets(X, y, random_state=42):
    """
    非十折数据集划分策略
    """
    # 80% 训练，10% 验证，10% 测试
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )  # Split into train and test+val
    X_val, X_test, y_val, y_test = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=random_state, stratify=y_test_val
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def read_mci(participants, data_folder, patient_group="all"):
    """
    读取 MCI 数据集
    """
    x = []
    y_emotion = []
    y_mci = []
    loaded_subjects = []

    for participant in participants:
        temp = pd.read_csv(f'{data_folder}/{participant}.csv', index_col=0)
        subject_mci = temp.iloc[0, 1]

        if patient_group != "all":
            if patient_group == "mci" and subject_mci != 1:
                continue
            elif patient_group == "hc" and subject_mci != 0:
                continue

        loaded_subjects.append(participant)

        mci_label = temp.iloc[:, 1].to_numpy()
        y_mci.append(mci_label)

        emotion_label = temp.iloc[:, 2].to_numpy()
        y_emotion.append(emotion_label)

        temp_array = temp.iloc[:, 3:].to_numpy()
        x.append(temp_array)

    logging.info(f"共加载 {len(loaded_subjects)} 名受试者的数据")

    x = np.concatenate(x, axis=0)
    y_emotion = np.concatenate(y_emotion, axis=0)
    y_mci = np.concatenate(y_mci, axis=0)

    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    x_normalized = scaler.fit_transform(x)

    return x_normalized, y_emotion, y_mci, loaded_subjects


def load_mci_data(return_numpy=False, patient_group="all"):
    """
    加载 MCI 数据集
    """
    participants = [f'S{i}' for i in range(1, 40)]
    data_folder = "data/mci/ecg_processed_256hz/"

    X, y_emotion, y_mci, loaded_subjects = read_mci(participants, data_folder, patient_group=patient_group)

    if return_numpy:
        return X, y_emotion
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = create_datasets(X, y_emotion)

        # 转换为 PyTorch 张量并增加通道维度
        X_train_tensor = torch.Tensor(X_train).unsqueeze(1)
        X_val_tensor = torch.Tensor(X_val).unsqueeze(1)
        X_test_tensor = torch.Tensor(X_test).unsqueeze(1)
        y_train_tensor = torch.Tensor(y_train).long()
        y_val_tensor = torch.Tensor(y_val).long()
        y_test_tensor = torch.Tensor(y_test).long()

        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor


def read_dreamer(participants, folder_channel1, folder_channel2, num_classes, classification_type):
    """
    根据参与者列表、分类类别数和分类任务类型（valence 或 arousal）读取 Dreamer 数据集
    """
    x_ch1 = []
    x_ch2 = []
    y = []

    for participant in participants:
        temp_ch1 = pd.read_csv(f'{folder_channel1}/{participant}.csv', index_col=0)
        temp_ch2 = pd.read_csv(f'{folder_channel2}/{participant}.csv', index_col=0)

        valence = temp_ch1['Valence'].to_numpy()
        arousal = temp_ch1['Arousal'].to_numpy()

        # 根据分类任务类型选择目标变量
        target = valence if classification_type == "valence" else arousal

        # 根据 num_classes 动态生成标签
        if num_classes == 2:
            # 二分类：低、高
            y_participant = np.where(target < 3, 0, 1)
        elif num_classes == 3:
            # 三分类：低、中、高
            y_participant = np.zeros_like(target)
            y_participant = np.where(target < 2, 0, y_participant)  # 低
            y_participant = np.where((target >= 2) & (target < 4), 1, y_participant)  # 中
            y_participant = np.where(target >= 4, 2, y_participant)  # 高
        elif num_classes == 5:
            # 五分类：假设目标变量范围为 1-5 分
            y_participant = target - np.min(target)
        elif num_classes == 4:
            # 四分类：高效价高唤醒、低效价高唤醒、高效价低唤醒、低效价低唤醒
            y_participant = np.zeros_like(valence)
            y_participant = np.where((valence >= 3) & (arousal >= 3), 3, y_participant)  # 高效价高唤醒
            y_participant = np.where((valence >= 3) & (arousal < 3), 2, y_participant)  # 高效价低唤醒
            y_participant = np.where((valence < 3) & (arousal >= 3), 1, y_participant)  # 低效价高唤醒
            y_participant = np.where((valence < 3) & (arousal < 3), 0, y_participant)  # 低效价低唤醒
        else:
            raise ValueError(f"Unsupported number of classes: {num_classes}")

        y.append(y_participant)

        # 提取特征数据
        temp_array_ch1 = temp_ch1.iloc[:, 4:].to_numpy()
        temp_array_ch2 = temp_ch2.iloc[:, 4:].to_numpy()

        x_ch1.append(temp_array_ch1)
        x_ch2.append(temp_array_ch2)

    # 合并所有参与者的数据
    x_ch1 = np.concatenate(x_ch1, axis=0)
    x_ch2 = np.concatenate(x_ch2, axis=0)
    y = np.concatenate(y, axis=0)

    # 标准化特征数据
    scaler_ch1 = StandardScaler()
    x_ch1_normalized = scaler_ch1.fit_transform(x_ch1.reshape(-1, x_ch1.shape[-1])).reshape(x_ch1.shape)

    scaler_ch2 = StandardScaler()
    x_ch2_normalized = scaler_ch2.fit_transform(x_ch2.reshape(-1, x_ch2.shape[-1])).reshape(x_ch2.shape)

    # 合并通道
    x_normalized = np.stack((x_ch1_normalized, x_ch2_normalized), axis=1)

    return x_normalized, y


def load_dreamer_data(num_classes, classification_type=None, return_numpy=False):
    """
    加载并处理 Dreamer 数据集
    """
    # 配置 Dreamer 数据集
    participants = [f'S{i}' for i in range(1, 24)]
    data_folder = {
        "channel1": "data/dreamer/channel1/",
        "channel2": "data/dreamer/channel2/"
    }

    # 读取数据
    X, y = read_dreamer(participants, data_folder["channel1"], data_folder["channel2"], num_classes,
                        classification_type)

    if return_numpy:
        return X, y
    else:
        # 使用 create_datasets 函数进行数据划分
        X_train, X_val, X_test, y_train, y_val, y_test = create_datasets(X, y)

        # 转换为 PyTorch 张量（Dreamer 数据集已经包含通道维度）
        X_train_tensor = torch.Tensor(X_train)
        X_val_tensor = torch.Tensor(X_val)
        X_test_tensor = torch.Tensor(X_test)
        y_train_tensor = torch.Tensor(y_train).long()
        y_val_tensor = torch.Tensor(y_val).long()
        y_test_tensor = torch.Tensor(y_test).long()

        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor


def read_wesad(participants, data_folder, num_classes):
    """
    读取 WESAD 数据集
    """
    x = []
    y = []
    # 定义有效的标签列表
    if num_classes == 3:
        valid_labels = [1, 2, 3]
    elif num_classes == 4:
        valid_labels = [1, 2, 3, 4]
    else:
        raise ValueError("num_classes 必须为 3 或 4")

    for participant in participants:
        temp = pd.read_csv(f'{data_folder}/{participant}.csv', index_col=0)

        y_participant = temp['Emotion Label'].to_numpy()

        # 过滤出有效标签的数据
        mask = np.isin(y_participant, valid_labels)
        y_filtered = y_participant[mask]
        x_filtered = temp.iloc[mask, 1:].to_numpy()

        # 将标签调整为从0开始
        y_adjusted = y_filtered - np.min(valid_labels)

        y.append(y_adjusted)
        x.append(x_filtered)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)

    # 标准化
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)

    return x_normalized, y


def load_wesad_data(num_classes, return_numpy=False):
    """
    加载并处理 WESAD 数据集
    """
    # 配置 WESAD 数据集
    participants = [f"S{x}" for x in range(2, 18) if x != 12]
    data_folder = "data/wesad/"

    # 读取数据
    X, y = read_wesad(participants, data_folder, num_classes)

    if return_numpy:
        return X, y
    else:
        # 使用 create_datasets 函数进行数据划分
        X_train, X_val, X_test, y_train, y_val, y_test = create_datasets(X, y)

        # 转换为 PyTorch 张量并增加通道维度
        X_train_tensor = torch.Tensor(X_train).unsqueeze(1)
        X_val_tensor = torch.Tensor(X_val).unsqueeze(1)
        X_test_tensor = torch.Tensor(X_test).unsqueeze(1)
        y_train_tensor = torch.Tensor(y_train).long()
        y_val_tensor = torch.Tensor(y_val).long()
        y_test_tensor = torch.Tensor(y_test).long()

        return X_train_tensor, X_val_tensor, X_test_tensor, y_train_tensor, y_val_tensor, y_test_tensor