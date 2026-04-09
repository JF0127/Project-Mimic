# # Copyright (c) 2026, Master Jia
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

import csv
import os
from datetime import datetime
from typing import Any

import numpy as np
import torch


class DataLogger:
    """
    通用数据记录器。
    特性：
    1. 内存缓冲：运行期间不写磁盘，避免卡顿，结束后一次性保存。
    2. 自动展开：自动将 Array/Tensor 类型的字段展开为多个 CSV 列 (e.g. obs -> obs_0, obs_1...)。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.buffer = []
        self.headers = []

        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    def log(self, data: dict[str, Any]):
        """
        记录一帧数据。
        Args:
            data: 字典，值可以是标量(float/int)、numpy数组、pytorch tensor 或 list。
        """
        flat_data = {}

        for key, value in data.items():
            # 1. 处理 PyTorch Tensor -> Numpy
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()

            # 2. 处理数组/列表 (自动展开)
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                # 确保是扁平的 (flatten)
                value = np.array(value).flatten()
                for i, v in enumerate(value):
                    flat_data[f"{key}_{i}"] = v.item()  # .item() 确保转为 Python float

            # 3. 处理标量
            else:
                flat_data[key] = value

        # 第一次记录时，确定表头 (Headers)
        if not self.headers:
            self.headers = list(flat_data.keys())

        self.buffer.append(flat_data)

    def save(self):
        """将缓冲区数据写入 CSV"""
        if not self.buffer:
            print("[Logger] Buffer is empty, nothing to save.")
            return

        # 加上时间戳，防止覆盖旧文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 如果文件名是以 .csv 结尾，插入时间戳
        if self.file_path.endswith(".csv"):
            real_filename = self.file_path.replace(".csv", f"_{timestamp}.csv")
        else:
            real_filename = f"{self.file_path}_{timestamp}.csv"

        print(f"\n💾 Saving log to: {real_filename} ... ", end="")

        try:
            with open(real_filename, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.buffer)
            print("Done! ✅")
        except Exception as e:
            print(f"Failed! ❌\nError: {e}")
