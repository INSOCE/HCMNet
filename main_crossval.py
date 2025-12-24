
import argparse
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from train import train_model
from evaluate import *
from utils.utils import *
from dataset import *
import time
import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from models.cnn_base import cnn_base
from models.HCMnet import HCMnet


def normalize_data(X_train, X_val, X_test, method='standard'):

    original_shape_train = X_train.shape
    original_shape_val = X_val.shape
    original_shape_test = X_test.shape

    if method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    if len(original_shape_train) == 2:
        X_train_normalized = scaler.fit_transform(X_train)
        X_val_normalized = scaler.transform(X_val)
        X_test_normalized = scaler.transform(X_test)

    elif len(original_shape_train) == 3:
        batch_train, channels, length = original_shape_train
        batch_val = original_shape_val[0]
        batch_test = original_shape_test[0]

        X_train_reshaped = X_train.reshape(-1, length)
        X_val_reshaped = X_val.reshape(-1, length)
        X_test_reshaped = X_test.reshape(-1, length)

        X_train_normalized = scaler.fit_transform(X_train_reshaped)
        X_val_normalized = scaler.transform(X_val_reshaped)
        X_test_normalized = scaler.transform(X_test_reshaped)

        X_train_normalized = X_train_normalized.reshape(batch_train, channels, length)
        X_val_normalized = X_val_normalized.reshape(batch_val, channels, length)
        X_test_normalized = X_test_normalized.reshape(batch_test, channels, length)

    else:
        raise ValueError(f"Unsupported data shape: {original_shape_train}")

    return X_train_normalized, X_val_normalized, X_test_normalized


def main():
    now = time.strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description="在不同数据集上训练和评估模型")
    parser.add_argument("--dataset", type=str, default="mci", choices=["wesad", "mci", "dreamer"])
    parser.add_argument("--model", type=str, default="HCMnet",
                        choices=["cnn_base", "ECG_LSTM", "ecgTransForm", "CSA_Net", "HCMnet", "Informer", "Transformer", "PatchTST",
                                 "ResNet1D", "SCINetClassifier", "InceptionTime", "TimesNet", "SegRNN", "MICN",  "ViT", "ECGMamba"])
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--classification_type", type=str, default="valence",
                        choices=["valence", "arousal"],
                        help="分类任务类型，仅对 dreamer 数据集有效。可选值：'valence', 'arousal'")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None, help="日志文件路径")
    parser.add_argument("--cuda_device", type=str, default="0")
    parser.add_argument("--k_folds", type=int, default=10, help="交叉验证折数")
    parser.add_argument("--normalization", type=str, default="robust", choices=["standard", "robust"],
                        help="归一化方法：standard (StandardScaler) 或 robust (RobustScaler)")

    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau",
                        choices=["ReduceLROnPlateau", "CosineAnnealing"], help="选择学习率调度器类型")
    parser.add_argument("--patience_lr", type=int, default=15, help="ReduceLROnPlateau的patience")
    parser.add_argument("--factor", type=float, default=0.1, help="ReduceLROnPlateau的factor")
    parser.add_argument("--T_max", type=int, default=100, help="CosineAnnealing的T_max")
    parser.add_argument("--eta_min", type=float, default=1e-6, help="CosineAnnealing的eta_min")

    args = parser.parse_args()

    seed_everything(args.seed)

    # 输出目录
    output_dir_name = f"{now}-{args.dataset}-{args.model}-{args.num_classes}-{args.classification_type}-{args.k_folds}fold"
    output_dir = os.path.join('logs_cv', output_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    # 设置日志记录
    if args.log_file is None:
        args.log_file = os.path.join(output_dir, f"{now}.log")
    setup_logging(args.log_file)
    save_arguments(args, args.log_file)

    # 设备选择
    if args.cuda_device and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
        logging.info(f"使用CUDA设备: {device}")
    else:
        device = torch.device("cpu")
        logging.info("使用CPU")

    if args.dataset == "dreamer":
        if args.num_classes == 4:
            classification_type = None
        else:
            classification_type = args.classification_type
        X, y = load_dreamer_data(num_classes=args.num_classes, classification_type=classification_type)
    elif args.dataset == "mci":
        if args.num_classes != 6:
            raise ValueError("MCI 数据集固定为6分类任务，num_classes 必须设置为 6")
        X, y = load_mci_data()
    elif args.dataset == "wesad":
        if args.num_classes not in [3, 4]:
            raise ValueError("WESAD 数据集固定为3/4分类任务，num_classes 必须设置为 3或4")
        X, y = load_wesad_data(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # 打印数据集信息
    logging.info(f"数据集: {args.dataset}")
    logging.info(f"模型: {args.model}")
    logging.info(f"分类类别数: {args.num_classes}")
    if args.dataset == "dreamer" and args.num_classes != 4:
        logging.info(f"分类任务类型: {args.classification_type}")
    logging.info(f"原始数据形状: X={X.shape}, y={y.shape}")

    # 设置K折交叉验证
    k = args.k_folds
    kfold = KFold(n_splits=k, shuffle=True, random_state=args.seed)

    # 结果记录
    all_accuracies = []
    all_f1s = []
    all_uars = []
    all_confusion_matrices = []

    # 记录整体开始时间
    overall_start_time = time.time()

    # 设置模型类字典
    MODEL_CLASSES = {
        "cnn_base": cnn_base,
        "HCMnet": HCMnet,
    }

    # 执行K折交叉验证
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        # 为每个折重置随机种子，确保可重复性
        seed_everything(args.seed)

        # 创建当前折的输出目录
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        # 保存当前折的索引以便复现
        np.save(os.path.join(fold_output_dir, "train_val_idx.npy"), train_val_idx)
        np.save(os.path.join(fold_output_dir, "test_idx.npy"), test_idx)

        logging.info(f"\n{'=' * 20} 开始第 {fold}/{k} 折 {'=' * 20}\n")

        try:
            # 获取当前折的训练+验证数据和测试数据（原始数据）
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

            # 将训练+验证数据再次划分为训练集和验证集
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.1, random_state=args.seed, stratify=y_train_val
            )

            # 打印划分后的数据分布信息
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_val, counts_val = np.unique(y_val, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            logging.info(f"Fold {fold} - 划分后数据形状:")
            logging.info(f"  训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
            logging.info(f"Fold {fold} - Train distribution: {dict(zip(unique_train, counts_train))}")
            logging.info(f"Fold {fold} - Validation distribution: {dict(zip(unique_val, counts_val))}")
            logging.info(f"Fold {fold} - Test distribution: {dict(zip(unique_test, counts_test))}")

            logging.info(f"Fold {fold} - 开始使用 {args.normalization} 方法进行归一化...")
            X_train, X_val, X_test = normalize_data(X_train, X_val, X_test, method=args.normalization)
            logging.info(f"Fold {fold} - 归一化完成")

            # 转换为PyTorch张量
            X_train_tensor = torch.Tensor(X_train)
            X_val_tensor = torch.Tensor(X_val)
            X_test_tensor = torch.Tensor(X_test)

            # 对于单通道数据集，添加通道维度
            if args.dataset in ["mci", "wesad"]:
                X_train_tensor = X_train_tensor.unsqueeze(1)
                X_val_tensor = X_val_tensor.unsqueeze(1)
                X_test_tensor = X_test_tensor.unsqueeze(1)

            y_train_tensor = torch.Tensor(y_train).long()
            y_val_tensor = torch.Tensor(y_val).long()
            y_test_tensor = torch.Tensor(y_test).long()

            logging.info(f"Fold {fold} - 转换为张量后:")
            logging.info(
                f"  训练集: {X_train_tensor.shape}, 验证集: {X_val_tensor.shape}, 测试集: {X_test_tensor.shape}")

            # 创建数据加载器
            train_loader = get_dataloader(
                TensorDataset(X_train_tensor, y_train_tensor),
                batch_size=args.batch_size,
                shuffle=True,
                seed=args.seed
            )
            val_loader = get_dataloader(
                TensorDataset(X_val_tensor, y_val_tensor),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed
            )
            test_loader = get_dataloader(
                TensorDataset(X_test_tensor, y_test_tensor),
                batch_size=args.batch_size,
                shuffle=False,
                seed=args.seed
            )

            # 创建模型实例
            model_class = MODEL_CLASSES.get(args.model)
            if not model_class:
                raise ValueError(f"不支持的模型: {args.model}")

            model = model_class(
                num_classes=args.num_classes,
                input_channels=2 if args.dataset == "dreamer" else 1
            )
            model.to(device)

            if fold == 1:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logging.info(f"模型总参数量: {total_params:,}")
                logging.info(f"可训练参数量: {trainable_params:,}")

            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

            # 定义学习率调度器
            if args.scheduler == "ReduceLROnPlateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=args.factor, patience=args.patience_lr
                )
            elif args.scheduler == "CosineAnnealing":
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.T_max, eta_min=args.eta_min
                )
            else:
                raise ValueError(f"不支持的调度器类型: {args.scheduler}")

            # 训练模型
            start_time = time.time()
            model, train_losses, val_losses = train_model(
                model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs=args.num_epochs, patience=args.patience,
                output_dir=fold_output_dir, device=device
            )
            end_time = time.time()
            logging.info(f"Fold {fold} 训练总耗时: {end_time - start_time:.2f} 秒")

            # 评估模型
            accuracy, f1, uar, cm = evaluate_model(
                model, test_loader,
                output_dir=fold_output_dir,
                dataset=args.dataset,
                num_classes=args.num_classes,
                device=device
            )

            # 记录结果
            all_accuracies.append(accuracy)
            all_f1s.append(f1)
            all_uars.append(uar)
            all_confusion_matrices.append(cm)

            logging.info(f"Fold {fold} 评估结果 - 准确率: {accuracy:.3f}, F1分数: {f1:.3f}, UAR: {uar:.3f}")

        except Exception as e:
            logging.error(f"处理Fold {fold}时发生错误: {str(e)}", exc_info=True)
            continue

    # 计算平均结果
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time

    avg_accuracy = np.mean(all_accuracies)
    avg_f1 = np.mean(all_f1s)
    avg_uar = np.mean(all_uars)

    std_accuracy = np.std(all_accuracies)
    std_f1 = np.std(all_f1s)
    std_uar = np.std(all_uars)

    logging.info(f"\n{'=' * 20} 十折交叉验证结果摘要 {'=' * 20}")
    logging.info(f"总耗时: {overall_time / 60:.2f} 分钟")
    logging.info(f"平均准确率: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    logging.info(f"平均F1分数: {avg_f1:.3f} ± {std_f1:.3f}")
    logging.info(f"平均UAR: {avg_uar:.3f} ± {std_uar:.3f}")

    if all_confusion_matrices:
        save_average_confusion_matrix(
            all_confusion_matrices,
            args.dataset,
            args.num_classes,
            output_dir
        )
        logging.info(f"已生成并保存十折平均混淆矩阵 (原始计数和百分比版本)")

    result_file = os.path.join(output_dir, "all_results.txt")
    with open(result_file, "w") as f:
        f.write("折叠号 | Accuracy | F1 Score | UAR\n")
        f.write("=" * 50 + "\n")
        for i in range(len(all_accuracies)):
            f.write(f"Fold {i + 1} | {all_accuracies[i]:.4f} | {all_f1s[i]:.4f} | {all_uars[i]:.4f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"平均值 | {avg_accuracy:.4f} ± {std_accuracy:.4f} | "
                f"{avg_f1:.4f} ± {std_f1:.4f} | {avg_uar:.4f} ± {std_uar:.4f}\n")

    logging.info(f"详细结果已保存到 {result_file}")


if __name__ == "__main__":
    main()

