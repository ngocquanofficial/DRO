import torch
perturb_output = torch.arange(9).reshape(3, 3).float()
original_output = torch.tensor([[0., 4., 6.],
        [3., 4., 5.],
        [7., 7., 8.]])
print(perturb_output)
print(original_output)
print(torch.norm( perturb_output - original_output.detach().clone() ,p= 2, dim= 1))
dist = torch.norm( perturb_output - original_output.detach().clone() ,p= 2, dim= 1).mean()
print(dist)