from typing import List, Optional, Tuple

import copy
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import SGD, Adam, AdamW
from .utils import DRO, SAM
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassCalibrationError as CalibrationError
from torchmetrics.classification.stat_scores import StatScores
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.optimization import get_cosine_schedule_with_warmup
import timm
import time
from src.loss import SoftTargetCrossEntropy
from src.mixup import Mixup

from .lora import LoRA_ViT
from .base_vit2 import ViT, CustomLinear, CustomLinear2
from .swag import SWAG, bn_update
from src.utils import log_det, fisher_distance, cal_cosine_similarity, euclid_distance
# from .base_vit import ViT, CustomLinear

torch.autograd.set_detect_anomaly(True)


MODEL_DICT = {
    "vit-b16-224-in21k": "google/vit-base-patch16-224-in21k",
    "vit-b32-224-in21k": "google/vit-base-patch32-224-in21k",
    "vit-l32-224-in21k": "google/vit-large-patch32-224-in21k",
    "vit-l15-224-in21k": "google/vit-large-patch16-224-in21k",
    "vit-h14-224-in21k": "google/vit-huge-patch14-224-in21k",
    "vit-b16-224": "google/vit-base-patch16-224",
    "vit-l16-224": "google/vit-large-patch16-224",
    "vit-b16-384": "google/vit-base-patch16-384",
    "vit-b32-384": "google/vit-base-patch32-384",
    "vit-l16-384": "google/vit-large-patch16-384",
    "vit-l32-384": "google/vit-large-patch32-384",
    "vit-b16-224-dino": "facebook/dino-vitb16",
    "vit-b8-224-dino": "facebook/dino-vitb8",
    "vit-s16-224-dino": "facebook/dino-vits16",
    "vit-s8-224-dino": "facebook/dino-vits8",
    "beit-b16-224-in21k": "microsoft/beit-base-patch16-224-pt22k-ft22k",
    "beit-l16-224-in21k": "microsoft/beit-large-patch16-224-pt22k-ft22k",
}


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vit-b16-224-in21k",
        optimizer: str = "sgd",
        rho: float= 0.05,
        lr: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        scheduler: str = "cosine",
        warmup_steps: int = 0,
        num_cycles: float = 0.5,
        n_classes: int = 10,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        mix_prob: float = 1.0,
        label_smoothing: float = 0.0,
        image_size: int = 224,
        weights: Optional[str] = None,
        training_mode: str = "full",
        lora_r: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: List[str] = ["query", "value"],
        lora_dropout: float = 0.0,
        lora_bias: str = "none",
        from_scratch: bool = False,
        num_particles: int = 10,
        use_sam: bool = False,
        weights_path: str = 'checkpoint/B_16.pth',
        epsilon: float = 0.01,
        cov_mat: bool = True,
        max_num_models: int = 20,
        start_swa_step: int = 10000,
        swa_freq: int = 10,
        use_swa_svgd: bool = False,
        use_sym_kl: bool = False,
        sigma = 1,

        # DRO
        grad_loop: int = 3,
        lamda = 3,
        distance= "fisher",
        bound= None,
        clip= 0.2,
        alpha_div=0.02,
        save_ckpt= False

    ):
        """Classification Model

        Args:
            model_name: Name of model checkpoint. List found in src/model.py
            optimizer: Name of optimizer. One of [adam, adamw, sgd]
            lr: Learning rate
            betas: Adam betas parameters
            momentum: SGD momentum parameter
            weight_decay: Optimizer weight decay
            scheduler: Name of learning rate scheduler. One of [cosine, none]
            warmup_steps: Number of warmup steps
            n_classes: Number of target class
            mixup_alpha: Mixup alpha value
            cutmix_alpha: Cutmix alpha value
            mix_prob: Probability of applying mixup or cutmix (applies when mixup_alpha and/or
                cutmix_alpha are >0)
            label_smoothing: Amount of label smoothing
            image_size: Size of input images
            weights: Path of checkpoint to load weights from (e.g when resuming after linear probing)
            training_mode: Fine-tuning mode. One of ["full", "linear", "lora"]
            lora_r: Dimension of LoRA update matrices
            lora_alpha: LoRA scaling factor
            lora_target_modules: Names of the modules to apply LoRA to
            lora_dropout: Dropout probability for LoRA layers
            lora_bias: Whether to train biases during LoRA. One of ['none', 'all' or 'lora_only']
            from_scratch: Initialize network with random weights instead of a pretrained checkpoint
        """
        super().__init__()
        # self.automatic_optimization = False
        self.save_hyperparameters()
        self.model_name = model_name
        self.optimizer = optimizer
        self.rho = rho
        self.lr = lr
        self.betas = betas
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.num_cycles = num_cycles
        self.n_classes = n_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.label_smoothing = label_smoothing
        self.image_size = image_size
        self.weights = weights
        self.training_mode = training_mode
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.from_scratch = from_scratch
        self.num_particles =  num_particles
        self.use_sam = use_sam
        self.epsilon = epsilon
        self.cov_mat = cov_mat
        self.max_num_models = max_num_models
        self.start_swag_step = start_swa_step
        self.swa_freq = swa_freq
        self.use_swa_svgd =  use_swa_svgd
        self.use_sym_kl = use_sym_kl
        self.sigma = sigma


        #DRO
        self.grad_loop = grad_loop
        self.lamda = torch.tensor(float(lamda), requires_grad=False).to("cuda")
        self.distance = distance
        self.bound = bound
        self.clip = clip
        self.alpha_div = alpha_div

        self.best_val_acc = 0.0
        self.epoch_start_time= 0
        self.save_ckpt = save_ckpt


        # Initialize network
        try:
            model_path = MODEL_DICT[self.model_name]
        except:
            raise ValueError(
                f"{model_name} is not an available model. Should be one of {[k for k in MODEL_DICT.keys()]}"
            )

        print('Model name', self.model_name)
        # self.net = ViT(name='B_16_imagenet1k', pretrained=True, num_classes=self.n_classes, image_size=self.image_size, num_particles=self.num_particles)
        self.net = ViT(name='vit-b16-224-in21k', pretrained=True, num_classes=self.n_classes, image_size=self.image_size, num_particles=self.num_particles, weight_path=weights_path)
        
        self.net = self.net.cuda()
                

        # Load checkpoint weights
        if self.weights:
            print(f"Loaded weights from {self.weights}")
            ckpt = torch.load(self.weights)["state_dict"]

            # Remove prefix from key names
            new_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith("net"):
                    k = k.replace("net" + ".", "")
                    new_state_dict[k] = v

            self.net.load_state_dict(new_state_dict, strict=True)
            


        if self.training_mode == "lora":
            
            # Wrap in LoRA model
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                modules_to_save=["classifier"],
            )

            self.net = LoRA_ViT(num_particles=self.num_particles, vit_model=self.net, r=self.lora_r, alpha=self.lora_alpha, num_classes=self.n_classes)
                

        else:
            raise ValueError(
                f"{self.training_mode} is not an available fine-tuning mode. Should be one of ['full', 'linear', 'lora']"
            )


        # Define metrics
        self.train_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                # "acc_top5": Accuracy(
                #     num_classes=self.n_classes,
                #     task="multiclass",
                #     top_k=min(5, self.n_classes),
                # ),
                # "ece": CalibrationError(num_classes=self.n_classes, norm='l1').to("cpu")
            }
        )
        self.val_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                # "acc_top5": Accuracy(
                #     num_classes=self.n_classes,
                #     task="multiclass",
                #     top_k=min(5, self.n_classes),
                # ),
                "ece": CalibrationError(num_classes=self.n_classes, norm='l1').to("cpu")
            }
        )
        self.test_metrics = MetricCollection(
            {
                "acc": Accuracy(num_classes=self.n_classes, task="multiclass", top_k=1),
                # "acc_top5": Accuracy(
                #     num_classes=self.n_classes,
                #     task="multiclass",
                #     top_k=min(5, self.n_classes),
                # ),
                "ece": CalibrationError(num_classes=self.n_classes, norm='l1').to("cpu"),
                "stats": StatScores(
                    task="multiclass", average=None, num_classes=self.n_classes
                ),
                
            }
        )

        # Define loss
        self.loss_fn = SoftTargetCrossEntropy()

        # Define regularizers
        self.mixup = Mixup(
            mixup_alpha=self.mixup_alpha,
            cutmix_alpha=self.cutmix_alpha,
            prob=self.mix_prob,
            label_smoothing=self.label_smoothing,
            num_classes=self.n_classes,
        )

        self.test_metric_outputs = []
        
        self.automatic_optimization = False

    def forward(self, x):
        
        res = self.net(x)
        return res

    def shared_step(self, batch, mode="train", logging= True):
        x, y = batch
        x, y = x.cuda(), y.cuda()

        if mode == "train":
            # Only converts targets to one-hot if no label smoothing, mixup or cutmix is set
            x, y = self.mixup(x, y)
        else:
            y = F.one_hot(y, num_classes=self.n_classes).float()


        pred = self(x) # List containing model predictions

        pred_ = 0 #final_prediction
        for j in range(self.num_particles):
            pred_ = pred_ + pred[j]
        pred_ = pred_/max(1, self.num_particles)


        entropy_loss = self.loss_fn(pred_, y)

        prob_pred = [torch.nn.functional.softmax(i, dim= -1) for i in pred]
        div_loss = - log_det(y, prob_pred, self.num_particles).to(pred[0].device)
        # ensemble_loss = ensemble_entropy(y, prob_pred, self.num_particles)

        if self.optimizer == 'sam' :
            loss = entropy_loss
        elif self.optimizer == 'dro' :
            loss = entropy_loss  + self.alpha_div * div_loss
        
        # Get accuracy
        metrics = getattr(self, f"{mode}_metrics")(pred_, y.argmax(1))


        # Log
        if logging :
            self.log(f"{mode}_DIV_LOSS", div_loss.item(), on_epoch=True)
            self.log(f"{mode}_loss", loss.item(), on_epoch=True)
            for k, v in metrics.items():
                if len(v.size()) == 0:
                    self.log(f"{mode}_{k.lower()}", v, on_epoch=True)


        if mode == "test":
            self.test_metric_outputs.append(metrics["stats"])
        
        return loss, pred_



    def training_step(self, batch, _):

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, prog_bar=True)

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        torch.autograd.set_detect_anomaly(True)

        if self.optimizer == 'sam' :

            # STEP 1 
            loss, original_output = self.shared_step(batch, "train")
            opt.zero_grad()
            self.manual_backward(loss)
            opt.step1(zero_grad= True)


            # STEP 2
            final_loss, final_output = self.shared_step(batch, "train", logging = False)

            self.manual_backward(final_loss)
            opt.step2(zero_grad= True) 

            opt.zero_grad()
            scheduler.step()




        elif self.optimizer == 'dro' :
                
            # STEP 1 (same as SAM)
            loss, original_output = self.shared_step(batch, "train")

            opt.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
            opt.step1(lamda= self.lamda, zero_grad= True)

            # STEP 2
            final_loss, final_output = self.shared_step(batch, "train", logging = False)
        
            if self.distance == "euclid" :
                final_distance, raw_distance = euclid_distance(final_output.detach(), original_output.detach(), bound= self.bound)
            elif self.distance == 'fisher':
                final_distance, raw_distance = fisher_distance(final_output.detach(), original_output.detach(), bound= self.bound)
            else :
                print("ERROR distance")

            self.manual_backward(final_loss)
            opt.step2(zero_grad= True) 
            

            # Update lamda by hand 

            lamda_ew = self.bound - final_distance.detach().clone()
            self.lamda = torch.clamp(self.lamda - current_lr * lamda_ew, min= 0.5, max= 5)


            opt.zero_grad()
            scheduler.step()


            # Log lamda
            self.log(f"LAMDA", self.lamda.item(), on_epoch=True)
            self.log(f"perturb_distance", raw_distance.item(), on_epoch=True)


    def validation_step(self, batch, _):
        val = self.shared_step(batch, "val")
        # self.test_step(batch, _)
        return val
    
    def on_validation_epoch_end(self):
        test_dataloader = self.trainer.datamodule.test_dataloader()
        for batch in test_dataloader:
            self.test_step(batch, 0)
            
    def test_step(self, batch, _):
        return self.shared_step(batch, "test")

    def on_train_epoch_start(self):
        # Capture the start time of the epoch
        self.epoch_start_time = time.time()
    
    def on_validation_epoch_end(self):
        if self.save_ckpt :

                
            # Retrieve the current validation accuracy
            current_val_acc = self.trainer.callback_metrics.get("val_acc", None)
            print(current_val_acc)
            if current_val_acc is not None:
                current_val_acc = current_val_acc.item()

                # Save checkpoint if current val_acc is higher than the best recorded val_acc
                if current_val_acc > self.best_val_acc:
                    print(f"New best val_acc: {current_val_acc}, saving checkpoint.")
                    self.best_val_acc = current_val_acc
                    self.trainer.save_checkpoint(f"{self.optimizer}_best_val_acc_{current_val_acc}.ckpt")
                else:
                    print(f"Current val_acc: {current_val_acc} did not exceed best val_acc: {self.best_val_acc}.")

    def on_train_epoch_end(self):
        # Calculate elapsed time
        print("TRAIN EPOCH END")
        epoch_duration = time.time() - self.epoch_start_time
        self.log("epoch_time", epoch_duration, prog_bar=True)
    

        current_val_acc = self.trainer.callback_metrics.get("val_acc", None)

        if self.save_ckpt : 

                
            # Check if the current validation accuracy is the best
            if current_val_acc is None:
                return
            
            if current_val_acc > self.best_val_acc:
                print(f"New best val_acc: {current_val_acc}, saving checkpoint.")
                self.best_val_acc = current_val_acc

                # Save the checkpoint
                checkpoint_path = f"{self.optimizer}_best_val_acc.ckpt"
                self.trainer.save_checkpoint(checkpoint_path)
            else:
                print(f"Current val_acc: {current_val_acc} did not exceed best val_acc: {self.best_val_acc}.")

    def on_test_epoch_end(self):
        """Save per-class accuracies to csv"""
        # Aggregate all batch stats
        print("HERE WE GO!!!!!!!!!!!!!!!")
        combined_stats = torch.sum(
            torch.stack(self.test_metric_outputs, dim=-1), dim=-1
        )

        # Calculate accuracy per class
        per_class_acc = []
        for tp, _, _, _, sup in combined_stats:
            acc = tp / sup
            per_class_acc.append((acc.item(), sup.item()))

        # Save to csv
        df = pd.DataFrame(per_class_acc, columns=["acc", "n"])
        df.to_csv("per-class-acc-test.csv")
        print("Saved per-class results in per-class-acc-test.csv")

        # Retrieve the current validation accuracy
        current_val_acc = self.trainer.callback_metrics.get("val_acc", None)
        print(current_val_acc)
        if current_val_acc is not None:
            current_val_acc = current_val_acc.item()

            # Save checkpoint if current val_acc is higher than the best recorded val_acc
            if current_val_acc > self.best_val_acc:
                print(f"New best val_acc: {current_val_acc}, saving checkpoint.")
                self.best_val_acc = current_val_acc
                self.trainer.save_checkpoint(f"best_val_acc_{current_val_acc}.ckpt")
            else:
                print(f"Current val_acc: {current_val_acc} did not exceed best val_acc: {self.best_val_acc}.")


    def configure_optimizers(self):
        # Initialize optimizer

        if self.optimizer == "dro":  #use Adam as the base optimizer by default @@        
            base_optimizer = torch.optim.SGD

            # optimizer =  DRO(param = self.net.parameters(),base_optimizer= base_optimizer, lr=self.lr, betas=self.betas,
            #     weight_decay=self.weight_decay, num_particles=self.num_particles, train_module=self, net=self.net, rho= self.rho)
            optimizer = DRO(self.net.parameters(), base_optimizer, lr= self.lr, momentum=0.9)


        elif self.optimizer == 'sam' :
            base_optimizer = torch.optim.SGD
            optimizer = SAM(self.net.parameters(), base_optimizer, lr= self.lr, momentum=0.9)

        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd', 'deepEns']"
            )




        # Initialize learning rate scheduler

        if self.scheduler == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=int(self.trainer.estimated_stepping_batches),
                num_warmup_steps=self.warmup_steps,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
            
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
