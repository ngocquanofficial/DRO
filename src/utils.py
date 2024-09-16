import math
import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.optim as optim
from scipy.spatial.distance import pdist, squareform
import random
def block_expansion(ckpt, split, original_layers):

    layer_cnt = 0
    selected_layers = []
    output = {}

    for i in range(original_layers):
        for k in ckpt:
            if ('layer.' + str(i) + '.') in k:
                output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
        layer_cnt += 1
        if (i+1) % split == 0:
            for k in ckpt:
                if ('layer.' + str(i) + '.') in k:
                    if 'attention.output' in k or str(i)+'.output' in k:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = torch.zeros_like(ckpt[k])
                        selected_layers.append(layer_cnt)
                    else:
                        output[k.replace(('layer.' + str(i) + '.'), ('layer.' + str(layer_cnt) + '.'))] = ckpt[k]
            layer_cnt += 1

    for k in ckpt:
        if not 'layer' in k:
            output[k] = ckpt[k]
        elif k == "vit.layernorm.weight" or k == "vit.layernorm.bias" or k == "dinov2.layernorm.bias" or k == "dinov2.layernorm.weight":
            output[k] = ckpt[k]
    
    selected_layers = list(set(selected_layers))

    return output, selected_layers


class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X): # X.shape = [n_particle, n_dim_of_theta]

    if X.shape[0] == 0:
        return 0

    distances = torch.cdist(X, X, p=2)
    dnorm2 = distances ** 2

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()
        
    return K_XY
  
class SVGD(torch.optim.Adam):
    def __init__(self, param, base_optimizer, lr=0, betas=(0.9, 0.999), weight_decay=0, num_particles=0, train_module=0, net=None, rho=0.1, adaptive=False,lamda= 1, **kwargs):

        # Base optimizer arguments
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, num_particles=num_particles, train_module=train_module, net=net, rho=rho, adaptive=adaptive, lamda=1, **kwargs)
        
        # Initialize the base optimizer (Adam)
        super(SVGD, self).__init__(param, lr=lr, betas=betas, weight_decay=weight_decay)  # Pass individual arguments
        
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
        self.grad_loop = 1
        # Init velocity and lamda

        for n, p in self.net.lora_vit.named_parameters():

            if p.requires_grad :
                self.state[p]['velocity'] = torch.zeros_like(p.data)
                self.state[p]['lamda'] = lamda



    @torch.no_grad()
    def step1(self, zero_grad=False):
        """First step: Perturb particle-specific parameters using SAM logic and save the original parameters."""
        
        lr = self.param_groups[0]['lr']

        # Create a set to keep track of the updated parameters

        for n, p in self.net.lora_vit.named_parameters():


            if p.requires_grad :

                # Save the original parameters for each particle and layer
                self.state[p]['old_p'] = p.data.clone()
                perturb = self.state[p]['velocity'] * lr
                p.add_(perturb.view(p.data.shape))  # Apply perturbation


                for _ in range(self.grad_loop) : 
                    e_w = ( p.grad - 2 * self.state[p]['lamda'] * (p - self.state[p]["old_p"]) ) * lr
                    p.add_(e_w.view(p.data.shape))



                # Update velocity, notice that p now is theta prime, NOT theta
                self.state[p]['velocity'].mul_(self.momentum).add_( (1 - self.momentum) * p.grad.data )


                



        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step2(self, zero_grad=False):
        """Second step: Restore original parameters and apply the gradient update."""
        lr = self.param_groups[0]['lr']


        for n, p in self.net.lora_vit.named_parameters():
            if p.requires_grad: # and n not in updated_n:

                curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                lamda_ew =  (self.rho - curr_dist )
                self.state[p]['lamda'] = self.state[p]['lamda'] - lamda_ew * lr

                p.data = self.state[p]['old_p']

                print()
                print("Current lamda:", self.state[p]['lamda'])
                print("Current distance:", curr_dist)
                print("Current lamda_ew: ", lamda_ew)
                print()

                

        self.base_optimizer.step()



        if zero_grad:
            self.zero_grad()


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
