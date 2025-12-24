from sklearn.metrics import accuracy_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import torch
import os
import numpy as np
#
#
# def evaluate_model(model, test_loader, output_dir, dataset, num_classes, device=None):
#     """
#     根据模型和数据集动态生成混淆矩阵并评估模型性能。支持微调阶段评估。
#
#     Args:
#         model: 训练好的模型。
#         test_loader: 测试集数据加载器。
#         output_dir: 结果保存的输出目录。
#         dataset: 数据集名称（如 "dreamer" 或 "mci"）。
#         num_classes: 分类类别数。
#         device: 计算设备。默认为 None，自动选择 CPU 或 CUDA。
#     """
#     if device is None:  # 如果没有传入 device，则自动选择
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)  # 将模型移动到指定的设备
#     model.eval()  # 设置模型为评估模式
#     all_preds = []
#     all_labels = []
#
#     with torch.no_grad():  # 在评估过程中禁用梯度计算
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             all_preds.extend(predicted.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#     # 计算评估指标
#     accuracy = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds, average='macro')
#     uar = recall_score(all_labels, all_preds, average='macro')
#
#     # 保存评估结果到文本文件
#     with open(os.path.join(output_dir, f'evaluation_results.txt'), 'w') as f:
#         f.write(f'Accuracy: {accuracy:.3f}\n')
#         f.write(f'F1 Score: {f1:.3f}\n')
#         f.write(f'UAR: {uar:.3f}\n')
#
#     # 绘制并保存混淆矩阵
#     cm = confusion_matrix(all_labels, all_preds)
#
#     # 根据数据集和分类类别设置标签
#     if dataset == "dreamer":
#         if num_classes == 2:
#             labels = ['low', 'high']
#         elif num_classes == 3:
#             labels = ['low', 'medium', 'high']
#         elif num_classes == 4:
#             labels = ['LVLA', 'LVHA', 'HVLA', 'HVHA']
#         elif num_classes == 5:
#             labels = ['1', '2', '3', '4', '5']
#         else:
#             labels = [str(i) for i in range(1, num_classes + 1)]
#     elif dataset == "mci":
#         labels = ['neutral', 'sad', 'angry', 'happy', 'boredom', 'tension']
#     elif dataset == "wesad":
#         if num_classes == 3:
#             labels = ['baseline', 'stress', 'amusement']
#         elif num_classes == 4:
#             labels = ['baseline', 'stress', 'amusement', 'meditation']
#     else:
#         labels = [str(i) for i in range(1, num_classes + 1)]  # 默认使用编号标签
#
#     # 绘制混淆矩阵
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.title(f"Confusion Matrix - {dataset} ({num_classes}-class)")
#     cm_path = os.path.join(output_dir, f'confusion_matrix.png')
#     plt.savefig(cm_path)  # 保存混淆矩阵为图片
#     plt.close()  # 关闭图形
#
#     return accuracy, f1, uar


def evaluate_model(model, test_loader, output_dir, dataset, num_classes, device=None):
    """
    评估模型性能并为当前折生成混淆矩阵。

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        output_dir: 保存结果的输出目录
        dataset: 数据集名称
        num_classes: 类别数
        device: 计算设备

    Returns:
        accuracy, f1, uar, confusion matrix
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    uar = recall_score(all_labels, all_preds, average='macro')

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 保存评估结果
    with open(os.path.join(output_dir, f'evaluation_results.txt'), 'w') as f:
        f.write(f'Accuracy: {accuracy:.3f}\n')
        f.write(f'F1 Score: {f1:.3f}\n')
        f.write(f'UAR: {uar:.3f}\n')

    # 根据数据集和类别数生成标签
    labels = get_class_labels(dataset, num_classes)

    # 绘制并保存此折的混淆矩阵 (调整字体大小)
    plt.figure(figsize=(10, 8))  # 增大图表尺寸

    # 使用annot_kws调整混淆矩阵中数字的字体大小
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 30})  # 设置注释字体大小

    # 调整标签和标题的字体大小
    plt.xlabel('Predicted labels', fontsize=16)
    plt.ylabel('True labels', fontsize=16)
    plt.title(f"Confusion Matrix - {dataset} ({num_classes}-class)", fontsize=18)

    # 调整刻度标签字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()  # 确保所有元素正确显示
    cm_path = os.path.join(output_dir, f'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)  # 增加DPI以提高图像质量
    plt.close()

    return accuracy, f1, uar, cm


def get_class_labels(dataset, num_classes):
    """Returns appropriate class labels based on dataset and number of classes"""
    if dataset == "dreamer":
        if num_classes == 2:
            labels = ['low', 'high']
        elif num_classes == 3:
            labels = ['low', 'medium', 'high']
        elif num_classes == 4:
            labels = ['LVLA', 'LVHA', 'HVLA', 'HVHA']
        elif num_classes == 5:
            labels = ['1', '2', '3', '4', '5']
        else:
            labels = [str(i) for i in range(1, num_classes + 1)]
    elif dataset == "mci":
        labels = ['neutral', 'sad', 'angry', 'happy', 'boredom', 'tension']
    elif dataset == "wesad":
        if num_classes == 3:
            labels = ['baseline', 'stress', 'amusement']
        elif num_classes == 4:
            labels = ['baseline', 'stress', 'amusement', 'meditation']
    else:
        labels = [str(i) for i in range(1, num_classes + 1)]

    return labels


def save_average_confusion_matrix(all_cms, dataset, num_classes, output_dir):
    """
    计算并保存所有折的平均混淆矩阵

    Args:
        all_cms: 所有折的混淆矩阵列表
        dataset: 数据集名称
        num_classes: 类别数
        output_dir: 输出目录
    """
    # 计算平均混淆矩阵
    avg_cm = np.mean(all_cms, axis=0)

    # 获取类别标签
    labels = get_class_labels(dataset, num_classes)

    # 1. 保存原始计数平均混淆矩阵 (四舍五入为整数)
    plt.figure(figsize=(12, 10))  # 进一步增大图表尺寸

    # 调整混淆矩阵中数字的字体大小
    sns.heatmap(np.round(avg_cm).astype(int), annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 24})  # 更大的注释字体

    # 调整标签和标题的字体大小
    plt.xlabel('Predicted labels', fontsize=18)
    plt.ylabel('True labels', fontsize=18)
    plt.title(f"Average Confusion Matrix (Counts) - {dataset} ({num_classes}-class)", fontsize=20)

    # 调整刻度标签字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    avg_cm_path = os.path.join(output_dir, f'average_confusion_matrix_counts.png')
    plt.savefig(avg_cm_path, dpi=300)  # 高分辨率输出
    plt.close()

    # 2. 保存百分比版本的混淆矩阵
    # 计算行和 (每个真实标签的总样本数)
    row_sums = avg_cm.sum(axis=1)
    # 避免除以零
    row_sums[row_sums == 0] = 1
    # 计算百分比
    cm_percent = (avg_cm / row_sums[:, np.newaxis]) * 100

    plt.figure(figsize=(12, 10))
    # 调整百分比混淆矩阵的字体大小
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 24})  # 更大的注释字体

    # 调整标签和标题的字体大小
    plt.xlabel('Predicted labels', fontsize=18)
    plt.ylabel('True labels', fontsize=18)
    plt.title(f"Average Confusion Matrix (Percentages) - {dataset} ({num_classes}-class)", fontsize=20)

    # 调整刻度标签字体大小
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.tight_layout()
    percent_cm_path = os.path.join(output_dir, f'average_confusion_matrix_percent.png')
    plt.savefig(percent_cm_path, dpi=300)  # 高分辨率输出
    plt.close()

