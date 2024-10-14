import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn.functional
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
import torch.linalg as linalg

def euclid_distance(pred1, pred2, bound, tau= 0.01) :
    prob1 = F.softmax(pred1, dim=1)
    prob2 = F.softmax(pred2, dim=1)

    raw_dist = torch.norm( prob1 - prob2 , p= 2, dim= 1).mean()

    return torch.exp(max( torch.tensor([0.0]).to(raw_dist.device) , raw_dist - bound) / tau) * raw_dist, raw_dist

def fisher_distance(pred1, pred2, bound, tau= 0.01) :
    # Make sure tensors are normalized to probability distributions (sum to 1)
    prob1 = F.softmax(pred1, dim=1)
    prob2 = F.softmax(pred2, dim=1)

    # Calculate KL divergence between corresponding vectors in x and y
    # Note: kl_div expects the input to be in log form
    kl_divergence1 = F.kl_div(torch.log(prob1 + 1e-8), prob2 + 1e-8, reduction='batchmean')
    kl_divergence2 = F.kl_div(torch.log(prob2+ 1e-8), prob1 + 1e-8, reduction='batchmean')
    fisher = 1/2 * (kl_divergence1 + kl_divergence2)
    raw_dist = fisher

    return torch.exp(max( torch.tensor([0.0]).to(raw_dist.device) , raw_dist - bound) / tau) * raw_dist, raw_dist
    
def log_det(y_true, pred, num_models):
    mask_non_y_true = ~y_true.bool()  
    log_dets = []
    
    for batch in range(pred[0].shape[0]):  # Iterate over each batch
        masked_preds = []
        for i in range(num_models):
            mask_pred = pred[i][batch][mask_non_y_true[batch]]
            masked_preds.append(mask_pred)

        masked_preds = torch.stack(masked_preds)
        norm_preds = masked_preds / torch.norm(masked_preds, dim=1, keepdim=True)

        # # Check var
        # for idx in range(num_models) :
        #     vector = norm_preds[idx]
        #     l2_norm = torch.norm(vector, p=2).item()

        matrix = torch.matmul(norm_preds, norm_preds.t())

        log_det_val = torch.logdet(matrix + 1e-7 * torch.eye(num_models).to(pred[0].device))
        log_dets.append(log_det_val)

    return torch.stack(log_dets).mean()


def cal_cosine_similarity(y_true, pred, num_models):
    min_cosines = []
    max_cosines = []
    avg_cosines = []

    mask_non_y_true = ~y_true.bool()  # Mask out the true class

    
    for batch in range( pred[0].shape[0] ):  # Iterate over each batch
        masked_preds = []
        for i in range(num_models):
            mask_pred = pred[i][batch][mask_non_y_true[batch]]
            masked_preds.append(mask_pred)

        masked_preds = torch.stack(masked_preds)

        norm_preds = masked_preds / torch.norm(masked_preds, dim=1, keepdim=True)

        nonmaximal = norm_preds

        n = nonmaximal.shape[0]
        total_similarity = 0
        count = 0
        max_cosine = 0
        min_cosine = 1

        

        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = torch.sum(nonmaximal[i] * nonmaximal[j])
                current = cos_sim.item()  
                total_similarity += current

                if current > max_cosine :
                    max_cosine = current
                if current < min_cosine :
                    min_cosine = current

                count += 1
        

        average_cosine_similarity = total_similarity / count if count > 0 else 0
        avg_cosines.append(average_cosine_similarity)
        min_cosines.append(min_cosine)
        max_cosines.append(max_cosine)

    
    return sum(avg_cosines)/len(avg_cosines), statistics.median(max_cosines), statistics.median(min_cosines)




class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def step1(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step2(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups





class DRO(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False,distance= "euclid", **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(DRO, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.distance = distance
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()

    def step1(self, lamda, zero_grad=False):

        for group in self.param_groups:

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                if self.distance == 'euclid' :
                    e_w = p.grad / ( 2 * lamda)
                elif self.distance == 'fisher' :
                    e_w = 1 / ( 2 * lamda * p.grad)
                else :
                    print("Distance error")


                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step2(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()


    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups