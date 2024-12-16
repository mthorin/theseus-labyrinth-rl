import torch
print(torch.backends.mps.is_available())  # Should return True if supported
