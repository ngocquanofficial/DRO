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
        self.lamda = lamda

        # Initialize base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.momentum = betas[0]
        
    
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


    @torch.no_grad()
    def step1(self, zero_grad=False):
        """First step: Perturb particle-specific parameters using SAM logic and save the original parameters."""
        
        # Get the particle-specific gradients for all LoRA layers
        q_A_grad, q_B_grad, v_A_grad, v_B_grad, clsW_grad, clsB_grad = self.get_grad1()

        # Create a set to keep track of the updated parameters
        updated_n = set()


        for n, p in self.net.lora_vit.named_parameters():

            if p.requires_grad :

                if 'velocity' not in self.state[p] :
                    self.state[p]['velocity'] = torch.zeros_like(p.data)





        for net_id in range(self.num_particles):
            for layer_id in range(12):  # Assuming 12 layers
                for n, p in self.net.lora_vit.named_parameters():


                    if p.requires_grad and n not in updated_n:


                        # Perturb the specific gradients for each particle and layer
                        if f'blocks.{str(layer_id)}' in n:
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    # Save the original parameters for each particle and layer
                                    self.state[p]['old_p'] = p.data.clone()
                                    velocity = self.state[p]['velocity']
                                    velocity.mul_(self.momentum).add_
                                    perturb = self.rho * q_A_grad[layer_id][net_id] / (q_A_grad[layer_id][net_id].norm() + 1e-12)
                                    p.add_(perturb.view(p.data.shape))  # Apply perturbation

                                    e_w = ( p.grad - 2 * self.lamda * (p - self.state[p]["old_p"]) ) # e_w =  grad(theta') - 2 * lambda * (theta' - theta) 
                                    p.add_(e_w.view(p.data.shape))
                                
                                elif f"w_b.layer.{net_id}" in n:
                                    self.state[p]['old_p'] = p.data.clone()
                                    perturb = self.rho * q_B_grad[layer_id][net_id] / (q_B_grad[layer_id][net_id].norm() + 1e-12)
                                    p.add_(perturb.view(p.data.shape))  # Apply perturbation

                                    e_w = ( p.grad - 2 * self.lamda * (p - self.state[p]["old_p"]) ) # e_w =  grad(theta') - 2 * lambda * (theta' - theta) 
                                    p.add_(e_w.view(p.data.shape))

                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    self.state[p]['old_p'] = p.data.clone()
                                    perturb = self.rho * v_A_grad[layer_id][net_id] / (v_A_grad[layer_id][net_id].norm() + 1e-12)
                                    p.add_(perturb.view(p.data.shape))  # Apply perturbation

                                    e_w = ( p.grad - 2 * self.lamda * (p - self.state[p]["old_p"]) ) # e_w =  grad(theta') - 2 * lambda * (theta' - theta) 
                                    p.add_(e_w.view(p.data.shape))

                                elif f"w_b.layer.{net_id}" in n:
                                    self.state[p]['old_p'] = p.data.clone()
                                    perturb = self.rho * v_B_grad[layer_id][net_id] / (v_B_grad[layer_id][net_id].norm() + 1e-12)
                                    p.add_(perturb.view(p.data.shape))  # Apply perturbation

                                    e_w = ( p.grad - 2 * self.lamda * (p - self.state[p]["old_p"]) ) # e_w =  grad(theta') - 2 * lambda * (theta' - theta) 
                                    p.add_(e_w.view(p.data.shape))

                        elif 'fc' in n:
                            if 'weight' in n:
                                self.state[p]['old_p'] = p.data.clone()
                                perturb = self.rho * clsW_grad[layer_id][net_id] / (clsW_grad[layer_id][net_id].norm() + 1e-12)
                                p.add_(perturb.view(p.data.shape))  # Apply perturbation

                                e_w = ( p.grad - 2 * self.lamda * (p - self.state[p]["old_p"]) ) # e_w =  grad(theta') - 2 * lambda * (theta' - theta) 
                                p.add_(e_w.view(p.data.shape))

                            elif 'bias' in n:
                                self.state[p]['old_p'] = p.data.clone()
                                perturb = self.rho * clsB_grad[layer_id][net_id] / (clsB_grad[layer_id][net_id].norm() + 1e-12)
                                p.add_(perturb.view(p.data.shape))  # Apply perturbation
                        
                                e_w = ( p.grad - 2 * self.lamda * (p - self.state[p]["old_p"]) ) # e_w =  grad(theta') - 2 * lambda * (theta' - theta) 
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
        step_lamda_ew = 0

        for net_id in range(self.num_particles):
            for layer_id in range(12):  # Assuming 12 layers
                for n, p in self.net.lora_vit.named_parameters():
                    if p.requires_grad and n not in updated_n:
                        # Restore the original parameters for each particle and layer

                        if f'blocks.{str(layer_id)}' in n:
                            if "proj_q" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    
                                        
                                    curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                                    lamda_ew =  (self.rho - curr_dist )
                                    lamda_ew = lamda_ew.detach()
                                    step_lamda_ew += lamda_ew


                                    p.data = self.state[p]['old_p']

                                    print("Current distance:", curr_dist)



                                elif f"w_b.layer.{net_id}" in n:
                                    

                                    curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                                    lamda_ew =  (self.rho - curr_dist )
                                    lamda_ew = lamda_ew.detach()
                                    step_lamda_ew += lamda_ew

                                    p.data = self.state[p]['old_p']

                                    print("Current distance:", curr_dist)



                            elif "proj_v" in n:
                                if f"w_a.layer.{net_id}" in n:
                                    

                                    curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                                    lamda_ew =  (self.rho - curr_dist )
                                    lamda_ew = lamda_ew.detach()
                                    step_lamda_ew += lamda_ew

                                    p.data = self.state[p]['old_p']


                                    print("Current distance:", curr_dist)



                                elif f"w_b.layer.{net_id}" in n:

                                    curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                                    lamda_ew =  (self.rho - curr_dist )
                                    lamda_ew = lamda_ew.detach()
                                    step_lamda_ew += lamda_ew

                                    p.data = self.state[p]['old_p']



                                    print("Current distance:", curr_dist)



                        elif 'fc' in n:
                            if 'weight' in n:

                                curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                                lamda_ew =  (self.rho - curr_dist )
                                lamda_ew = lamda_ew.detach()
                                step_lamda_ew += lamda_ew

                                p.data = self.state[p]['old_p']



                                print("Current distance:", curr_dist)


                            elif 'bias' in n:
                                
                                curr_dist = torch.dist( p, self.state[p]["old_p"] ,p= 2)
                                lamda_ew =  (self.rho - curr_dist )
                                lamda_ew = lamda_ew.detach()
                                step_lamda_ew += lamda_ew

                                p.data = self.state[p]['old_p']

                                print("Current distance:", curr_dist)


                        # Mark this parameter as updated
                        updated_n.add(n)
                    
                    
                    # ### DEBUG
                    # a = random.randint(0, 10000)
                    # if a == 2306 :


                    # ### DEBUG

        self.lamda = max(self.lamda - lr * step_lamda_ew, 0)
        self.base_optimizer.step()

        print()
        print("Current lamda:", self.lamda)
        print("Current distance:", curr_dist)
        print("Current lamda_ew: ", lamda_ew)
        print()

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
