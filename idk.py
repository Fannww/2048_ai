import torch
print(torch.cuda.is_available())        # Should be True
print(torch.cuda.device_count())        # Should be 1 for RTX 3050
print(torch.cuda.get_device_name(0))    # RTX 3050
