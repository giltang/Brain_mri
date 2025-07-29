import torch

if torch.cuda.is_available():
    gpu_id = torch.cuda.current_device()
    print(f"GPU name: {torch.cuda.get_device_name(gpu_id)}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated(gpu_id) / 1024**2:.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(gpu_id) / 1024**2:.2f} MB")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**2:.2f} MB")
else:
    print("CUDA is not available")

