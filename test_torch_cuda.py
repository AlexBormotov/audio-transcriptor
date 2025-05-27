import torch
print(torch.version.cuda)  # Должна быть не None
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))  # Если есть хотя бы 1 GPU