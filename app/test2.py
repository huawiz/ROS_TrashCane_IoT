import sys
import platform
import psutil
import torch
import GPUtil

def check_system():
    # 系統信息
    print("\n=== 系統信息 ===")
    print(f"操作系統: {platform.system()} {platform.version()}")
    print(f"Python版本: {sys.version.split()[0]}")
    
    # CPU信息
    print("\n=== CPU信息 ===")
    print(f"CPU核心數: {psutil.cpu_count()} 核心")
    print(f"CPU型號: {platform.processor()}")
    
    # 記憶體信息
    memory = psutil.virtual_memory()
    print("\n=== 記憶體信息 ===")
    print(f"總記憶體: {memory.total / (1024 ** 3):.1f} GB")
    print(f"可用記憶體: {memory.available / (1024 ** 3):.1f} GB")
    
    # GPU信息
    print("\n=== GPU信息 ===")
    try:
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                gpu = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {gpu.name}")
                print(f"顯存大小: {gpu.total_memory / (1024**2):.1f} MB")
                print(f"CUDA能力: {gpu.major}.{gpu.minor}")
        else:
            print("未檢測到支持CUDA的GPU")
            
        # 使用GPUtil獲取更詳細的GPU信息
        gpus = GPUtil.getGPUs()
        if gpus:
            for gpu in gpus:
                print(f"\nGPU使用率: {gpu.load*100}%")
                print(f"顯存使用率: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
    except Exception as e:
        print(f"獲取GPU信息時出錯: {e}")
    
    # 建議
    print("\n=== 訓練建議 ===")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        if gpu.total_memory >= 6 * (1024**3):  # 6GB
            print("✅ GPU顯存充足，可以正常訓練")
            print(f"建議batch_size: {min(16, max(4, int(gpu.total_memory / (1024**3) / 0.75)))}")
        else:
            print("⚠️ GPU顯存較小，需要調整訓練參數:")
            print("1. 降低batch_size（建議4-8）")
            print("2. 降低圖片尺寸（如416或384）")
            print("3. 使用較小的模型（如yolov5n或yolov5s）")
    else:
        print("⚠️ 未檢測到GPU，將使用CPU訓練（速度會很慢）")
    
    if memory.total < 8 * (1024**3):  # 8GB
        print("⚠️ 系統記憶體較小，可能會影響訓練效率")
    
if __name__ == "__main__":
    check_system()