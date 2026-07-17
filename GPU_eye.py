import GPUtil

# Get the list of available GPUs
gpus = GPUtil.getGPUs()

for gpu in gpus:
    print(f"GPU {gpu.id}: Free Memory: {gpu.memoryFree} MB, "
          f"Used Memory: {gpu.memoryUsed} MB, Total Memory: {gpu.memoryTotal} MB")