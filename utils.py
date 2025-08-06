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


def setup_logging(log_file="output.log"):
    """设置日志记录，保存到文件并同时输出到控制台"""
    # 确保目录存在
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )


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
