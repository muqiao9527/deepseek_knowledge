#!/usr/bin/env python
# scripts/check_gpus.py
import os
import sys
import json
import torch


def print_gpu_info():
    """打印GPU信息并生成设备映射建议"""
    if not torch.cuda.is_available():
        print("未检测到可用的CUDA设备")
        return

    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个CUDA设备:")

    # 打印每个GPU的详细信息
    total_memory = 0
    gpu_info = []

    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        name = props.name
        total_mem = props.total_memory / (1024 ** 3)  # 转换为GB
        free_mem = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)  # 可用显存 (GB)

        print(f"  GPU {i}: {name}, 总显存: {total_mem:.2f} GB, 可用显存: {free_mem:.2f} GB")

        gpu_info.append({
            "index": i,
            "name": name,
            "total_memory_gb": total_mem,
            "free_memory_gb": free_mem
        })

        total_memory += total_mem

    print(f"\n总显存: {total_memory:.2f} GB")

    # 为大模型生成设备映射建议
    print("\n== 针对大模型的设备映射建议 ==")
    print("对于7B大模型 (约需14-16GB显存):")
    if total_memory < 15:
        print("  - 显存不足，建议使用4-bit量化 (设置 DEEPSEEK_LOAD_4BIT=True)")
    elif total_memory < 30:
        print("  - 使用单GPU + float16:")
        print("    DEEPSEEK_DEVICE_MAP=0")
    else:
        print("  - 使用自动设备映射:")
        print("    DEEPSEEK_DEVICE_MAP=auto")

    print("\n对于13B大模型 (约需28-32GB显存):")
    if total_memory < 30:
        print("  - 显存不足，建议使用4-bit量化 (设置 DEEPSEEK_LOAD_4BIT=True)")
    elif total_memory < 60:
        if gpu_count >= 2:
            print("  - 使用2个GPU:")
            print("    DEEPSEEK_DEVICE_MAP=auto")
        else:
            print("  - 显存可能不足，建议使用4-bit量化 (设置 DEEPSEEK_LOAD_4BIT=True)")
    else:
        print("  - 使用自动设备映射:")
        print("    DEEPSEEK_DEVICE_MAP=auto")

    # 生成样本ENV配置
    print("\n== 建议的.env配置 ==")
    env_config = [
        "# Deepseek本地模型配置",
        "DEEPSEEK_MODEL_PATH=/path/to/your/deepseek-model",
        "DEEPSEEK_DEVICE_MAP=auto",
        "DEEPSEEK_MODEL_TYPE=deepseek-llm",
        "DEEPSEEK_LOAD_8BIT=False",
        "DEEPSEEK_LOAD_4BIT=False"
    ]
    print("\n".join(env_config))

    print("\n如果您的模型为70B级别，建议分布在多个GPU上，并设置:")
    print("DEEPSEEK_DEVICE_MAP=auto")
    print("或使用4-bit量化: DEEPSEEK_LOAD_4BIT=True")


if __name__ == "__main__":
    print_gpu_info()