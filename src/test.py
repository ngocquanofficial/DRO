import torch
import torch.nn.functional as F
import statistics
log_offset = 1e-7



def fisher_distance(pred1, pred2) :

    # Make sure tensors are normalized to probability distributions (sum to 1)
    pred1 = pred1 / pred1.sum(dim=1, keepdim=True)
    pred2 = pred2 / pred2.sum(dim=1, keepdim=True)

    # Calculate KL divergence between corresponding vectors in x and y
    # Note: kl_div expects the input to be in log form
    kl_divergence1 = F.kl_div(pred1.log(), pred2, reduction='batchmean')
    kl_divergence2 = F.kl_div(pred2.log(), pred1, reduction='batchmean')

    return 1/2 * (kl_divergence1 + kl_divergence2)

for i in range(10) :
    x = torch.rand(24, 100)
    y = torch.rand(24, 100)
    # x = torch.tensor(torch.arange(9).reshape(3, 3), dtype= torch.float32)
    # y =  torch.tensor(torch.arange(9).reshape(3, 3), dtype= torch.float32)


    distances = torch.norm(x - y, p=2, dim=1).mean()
    print(distances.shape)

    # Calculate the average distance
    average_distance = distances.mean().item()
    print(average_distance)
    print(fisher_distance(x, y))