import torch
import torch.nn.functional as F
import statistics
log_offset = 1e-7



# print(torch.nn.functional.softmax(pred3, dim= -1))
a = torch.tensor([1.])
print(torch.clamp(a - 2, min= 0.05))