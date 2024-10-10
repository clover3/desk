import torch


def compute_per_device_batch_size(train_batch_size):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # If no GPUs are available, set num_gpus to 1 (for CPU)
    if num_gpus == 0:
        num_gpus = 1

    # Compute per_device_train_batch_size
    per_device_train_batch_size = train_batch_size // num_gpus

    # Ensure per_device_train_batch_size is at least 1
    per_device_train_batch_size = max(1, per_device_train_batch_size)

    return per_device_train_batch_size
