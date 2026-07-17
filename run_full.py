#!/usr/bin/env python3
import os
import time
import GPUtil
import subprocess
import argparse
parser = argparse.ArgumentParser(description="GPU Task Scheduler")
parser.add_argument('--gpus', type=str, required=True, help='Comma-separated list of GPU IDs to monitor')
args = parser.parse_args()

# 认为 GPU 空闲的显存使用阈值（单位：MB）
THRESHOLD = 100

# 要分配的任务命令（根据实际情况修改）
TASK_CMD = ["python", "run.py"]

# 用于记录每个 GPU 上正在运行的任务（键：GPU 编号，值：subprocess 对象）
running_tasks = {}

gpu_eye_on_list = list(map(int, args.gpus.split(',')))

print(f"启动 GPU 调度器，正在监控 GPU: {gpu_eye_on_list} 是否空闲...")

while True:
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        if gpu.id not in gpu_eye_on_list:
            continue
        # 如果该 GPU 的显存使用量低于阈值
        if gpu.memoryUsed < THRESHOLD:
            # 检查该 GPU 是否已有任务正在运行
            proc = running_tasks.get(gpu.id)
            if proc is not None and proc.poll() is None:
                # 任务还在运行，跳过该 GPU
                continue

            # 分配新任务给该 GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu.id)
            proc = subprocess.Popen(TASK_CMD, env=env)
            running_tasks[gpu.id] = proc
            print(f"为 GPU {gpu.id} 分配了新任务。")

    # 每次检查之间休眠几秒（可以根据需要调整）
    time.sleep(5)