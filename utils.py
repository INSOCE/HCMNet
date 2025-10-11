# 开发时间：2024/11/20 15:52
import torch
import numpy as np
import random
import logging
import sys
import json
from torch.utils.data import DataLoader
import os

def seed_everything(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file):
    """
    修复版日志配置
    """
    import logging
    import sys

    # 清除现有handlers
    logging.getLogger().handlers.clear()

    # 设置日志级别
    logging.getLogger().setLevel(logging.INFO)

    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    simple_formatter = logging.Formatter('%(message)s')

    # 文件handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(file_handler)
    except Exception as e:
        print(f"创建文件日志失败: {e}")

    # 控制台handler - 强制输出到stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(simple_formatter)
    console_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(console_handler)

    # 测试日志输出
    print("📝 测试print输出")
    logging.info("📝 测试logging输出")
    sys.stdout.flush()

    # 检查handlers
    print(f"日志handlers数量: {len(logging.getLogger().handlers)}")
    for i, handler in enumerate(logging.getLogger().handlers):
        print(f"Handler {i}: {type(handler)}")


def save_arguments(args, log_file="output.log"):
    """将命令行参数保存到日志文件中"""
    with open(log_file, "a", encoding="utf-8") as f:  # 确保使用 UTF-8 编码
        f.write("\n--- Arguments ---\n")
        json.dump(vars(args), f, indent=4, ensure_ascii=False)  # 禁止 ASCII 转义
        f.write("\n-----------------\n")


def get_dataloader(dataset, batch_size, shuffle, seed, num_workers=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id),
        generator=generator
    )
