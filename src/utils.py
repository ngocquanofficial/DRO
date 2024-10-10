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
    


def wasserstein_distance(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()

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



# Hàm này e dùng tính cosine_similarity để debug
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


class DRO_old(torch.optim.Adam):
    def __init__(self, param, base_optimizer, lr=0, betas=(0.9, 0.999), weight_decay=0, num_particles=0, train_module=0, net=None, rho=0.1, adaptive=False,lamda= 1, **kwargs):

        # Base optimizer arguments
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, num_particles=num_particles, train_module=train_module, net=net, rho=rho, adaptive=adaptive, lamda=1, **kwargs)
        
        # Initialize the base optimizer (Adam)
        super(DRO, self).__init__(param, lr=lr, betas=betas, weight_decay=weight_decay)  # Pass individual arguments
        
        self.net = net
        self.num_particles = num_particles
        self.lr = lr
        self.train_module = train_module

        self.rho = rho
        self.adaptive = adaptive

        # Initialize base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        
        self.momentum = betas[0]
        self.grad_loop = 2
        # Init velocity and lamda

        for n, p in self.net.lora_vit.named_parameters():

            if p.requires_grad :
                self.state[p]['velocity'] = torch.zeros_like(p.data)


    
    def get_grad1(self): #for LoRA
        
        q_A = torch.empty(0).cuda()
        q_B = torch.empty(0).cuda()
        v_A = torch.empty(0).cuda()
        v_B = torch.empty(0).cuda()
        cls_w = torch.empty(0).cuda()
        cls_b = torch.empty(0).cuda()
        tq_A = torch.empty(0).cuda()
        tq_B = torch.empty(0).cuda()
        tv_A = torch.empty(0).cuda()
        tv_B = torch.empty(0).cuda()
        tcls_w = torch.empty(0).cuda()
        tcls_b = torch.empty(0).cuda()
        i_qA = 0
        i_qB = 0
        i_vA = 0
        i_vB = 0
        i_cls_w = 0
        i_cls_b = 0
        for n, p in self.net.named_parameters():
            if p.requires_grad:
                if "proj_q" in n:
                    if "w_a" in n:
                        if i_qA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_A = torch.cat((tq_A, p_), dim=1)
                            i_qA += 1
                            if i_qA == self.num_particles:   
                                q_A = torch.cat((q_A, tq_A), dim=0)
                                tq_A = torch.empty(0).cuda()
                                i_qA = 0
                    elif "w_b" in n:
                        if i_qB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tq_B = torch.cat((tq_B, p_), dim=1)
                            i_qB += 1
                            if i_qB == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                q_B = torch.cat((q_B, tq_B), dim=0)
                                tq_B = torch.empty(0).cuda()
                                i_qB = 0
                elif "proj_v" in n:
                    if "w_a" in n:
                        if i_vA < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_A = torch.cat((tv_A, p_), dim=1)
                            i_vA += 1
                            if i_vA == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                v_A = torch.cat((v_A, tv_A), dim=0)
                                tv_A = torch.empty(0).cuda()
                                i_vA = 0
                    elif "w_b" in n:
                        if i_vB < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tv_B = torch.cat((tv_B, p_), dim=1)
                            i_vB += 1
                            if i_vB == self.num_particles:   
                                # print(q_A.shape, tq_A.shape)
                                v_B = torch.cat((v_B, tv_B), dim=0)
                                tv_B = torch.empty(0).cuda()
                                i_vB = 0
                                
                elif 'fc' in n:
                    if 'weight' in n:
                        if i_cls_w < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_w = torch.cat((tcls_w, p_), dim=1)
                            i_cls_w += 1
                            if i_cls_w == self.num_particles:   
                                cls_w = torch.cat((cls_w, tcls_w), dim=0)
                                tcls_w = torch.empty(0).cuda()
                                i_cls_w = 0
                    elif 'bias' in n:
                        if i_cls_b < self.num_particles:
                            p_ = p.grad.data.view(1, 1, -1)
                            tcls_b = torch.cat((tcls_b, p_), dim=1)
                            i_cls_b += 1
                            if i_cls_b == self.num_particles:   
                                cls_b = torch.cat((cls_b, tcls_b), dim=0)
                                tcls_b = torch.empty(0).cuda()
                                i_cls_b = 0
                    
                    
        return q_A, q_B, v_A, v_B, cls_w, cls_b






##############################################################


    @torch.no_grad()
    def step1(self, lamda, zero_grad=False):
        """First step: Perturb particle-specific parameters using SAM logic and save the original parameters."""
        
        # Get the particle-specific gradients for all LoRA layers
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_grad, clsB_grad = self.get_grad1()

        # Create a set to keep track of the updated parameters
        updated_n = set()
        lr = self.param_groups[0]['lr']



        for net_id in range(self.num_particles):
            for layer_id in range(12):  # Assuming 12 layers
                for n, p in self.net.lora_vit.named_parameters():

                    if p.requires_grad and n not in updated_n:


                        # Perturb the specific gradients for each particle and layer
                        if f'blocks.{str(layer_id)}' in n:
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:

                                    self.state[p]['old_p'] = p.data.clone()
                                    e_w = p.grad / (2 * lamda)
                                    p.add_(e_w.view(p.data.shape))
                                
                                elif f"w_b.layer.{net_id}" in n:

                                    self.state[p]['old_p'] = p.data.clone()
                                    e_w = p.grad / (2 * lamda)
                                    p.add_(e_w.view(p.data.shape))

                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:

                                    self.state[p]['old_p'] = p.data.clone()
                                    e_w = p.grad / (2 * lamda)
                                    p.add_(e_w.view(p.data.shape))

                                elif f"w_b.layer.{net_id}" in n:

                                    self.state[p]['old_p'] = p.data.clone()
                                    e_w = p.grad / (2 * lamda)
                                    p.add_(e_w.view(p.data.shape))


                        elif 'fc' in n:
                            if 'weight' in n:

                                self.state[p]['old_p'] = p.data.clone()
                                e_w = p.grad / (2 * lamda)
                                p.add_(e_w.view(p.data.shape))
                                

                            elif 'bias' in n:
                            
                                self.state[p]['old_p'] = p.data.clone()
                                e_w = p.grad / (2 * lamda)
                                p.add_(e_w.view(p.data.shape))
                        # Mark this parameter as updated
                        updated_n.add(n)

                    

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step2(self, zero_grad=False):
        """Second step: Restore original parameters and apply the gradient update."""
        lr = self.param_groups[0]['lr']
        # Restore the original parameters
        updated_n = set()  # Track which parameters have been restored
        curr_dist = False

        for net_id in range(self.num_particles):
            for layer_id in range(12):  # Assuming 12 layers
                for n, p in self.net.lora_vit.named_parameters():
                    if p.requires_grad and n not in updated_n:
                        # Restore the original parameters for each particle and layer

                        if f'blocks.{str(layer_id)}' in n:
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    p.data = self.state[p]['old_p']



                                elif f"w_b.layer.{net_id}" in n:
                                    p.data = self.state[p]['old_p']


                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    p.data = self.state[p]['old_p']



                                elif f"w_b.layer.{net_id}" in n:
                                    p.data = self.state[p]['old_p']


                        elif 'fc' in n:
                            if 'weight' in n:
                                p.data = self.state[p]['old_p']




                            elif 'bias' in n:
                                p.data = self.state[p]['old_p']



                        # Mark this parameter as updated
                        updated_n.add(n)
                    
                    
                    # ### DEBUG
                    # a = random.randint(0, 10000)
                    # if a == 2306 :


                    # ### DEBUG

        self.base_optimizer.step()


        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.step1(zero_grad=True)
        closure()
        self.step2()


###################################################################

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


import torch


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