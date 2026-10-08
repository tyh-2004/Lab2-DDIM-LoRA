from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scheduler import extract

class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
        self.predictor = kwargs.get("predictor", "noise")


    def get_loss_noise(self, x0, class_label=None, noise=None):
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)  # (B,)
        x_t, eps = self.var_scheduler.add_noise(x0, t, eps=noise)
        eps_pred = self.network(x_t, t, class_label) if class_label is not None else self.network(x_t, t)
        return F.mse_loss(eps_pred, eps)
    
    def get_loss_x0(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # Here we implement the "predict x0" version. # 直接預測乾淨(無噪)影像
        # 1. Sample a timestep and add noise to get (x_t, noise).
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)
        x_t, noise = self.var_scheduler.add_noise(x0, t, eps=noise)
        # 2. Pass (x_t, timestep) into self.network, where the output should represent the clean sample x0_pred.
        if class_label is not None:
            x0_pred = self.network(x_t, t, class_label) # class_label : 條件控制標籤，可控模型定向生成
        else:
            x0_pred = self.network(x_t, t)
        # 3. Compute the loss as MSE(predicted x0_pred, ground-truth x0).
        loss = F.mse_loss(x0_pred, x0)
        ######################
        #loss = None
        return loss
    
    def get_loss_mean(self, x0, class_label=None, noise=None):
        ######## TODO ########
        # Here we implement the "predict mean" version.
        # 1. Sample a timestep and add noise to get (x_t, noise).
        B = x0.shape[0]
        t = self.var_scheduler.uniform_sample_t(B, x0.device)
        x_t, noise = self.var_scheduler.add_noise(x0, t, eps=noise)
        # 2. Pass (x_t, timestep) into self.network, where the output should represent the posterior mean μθ(x_t, t).
        # 直接預測反向去噪時所需要的平均值
        if class_label is not None:
            mean_pred = self.network(x_t, t, class_label)
        else:
            mean_pred = self.network(x_t, t)
        # 3. Compute the *true* posterior mean from the closed-form DDPM formula (using x0, x_t, and scheduler terms).
        alpha_bar_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        alpha_t = extract(self.var_scheduler.alphas, t, x0)
        beta_t = extract(self.var_scheduler.betas, t, x0)
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, x0)
        alpha_bar_prev = torch.where((t == 0).reshape(-1, 1, 1, 1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)
        
        true_mean = ((torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)) * x0 + (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)) * x_t)
        # 4. Compute the loss as MSE(predicted mean, true mean).
        loss = F.mse_loss(mean_pred, true_mean)
        ######################
        #loss = None
        return loss
    
    def get_loss(self, x0, class_label=None, noise=None):
        if self.predictor == "noise":
            return self.get_loss_noise(x0, class_label, noise)
        elif self.predictor == "x0":
            return self.get_loss_x0(x0, class_label, noise)
        elif self.predictor == "mean":
            return self.get_loss_mean(x0, class_label, noise)
        else:
            raise ValueError(f"Unknown predictor: {self.predictor}")
    
    
    
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
        guidance_scale: Optional[float] = 1.0,
    ):
        x_T = torch.randn([batch_size, 3, self.image_resolution, self.image_resolution]).to(self.device)

        do_classifier_free_guidance = guidance_scale > 1.0

        if do_classifier_free_guidance:

            ######## CFG ########
            # Implement the classifier-free guidance.
            # Specifically, given a tensor of shape (batch_size,) containing class labels,
            # create a tensor of shape (2*batch_size,) where the first half is filled with zeros (i.e., null condition).
            assert class_label is not None
            assert len(class_label) == batch_size, f"len(class_label) != batch_size. {len(class_label)} != {batch_size}"
            #raise NotImplementedError("TODO")
            class_label = class_label.to(self.device)
            class_label = torch.cat([torch.zeros_like(class_label), class_label], dim=0)
            #######################

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            if do_classifier_free_guidance:
                ######## CFG ########
                # Implement the classifier-free guidance.
                # Assignment 2. Implement the classifier-free guidance.
                #raise NotImplementedError("TODO")
                x_t_input = torch.cat([x_t, x_t], dim=0)
                # 將合併後的影像、時間步長與類別標籤輸入神經網路
                # 另此處傳入的 class_label 前半段通常已設為 null token
                timestep = torch.full((2 * batch_size,), t.item(), device=self.device, dtype=torch.long)
                net_out = self.network(x_t_input, timestep=timestep, class_label=class_label)
                # 將網路輸出沿 batch 維度對半拆開，分別取得無條件預測與條件預測結果
                net_out_uncond, net_out_cond = net_out.chunk(2, dim=0)
                # 利用 guidance_scale (s) 強化符合 class_label 的特徵方向，同時壓制無條件的通用特徵
                net_out = net_out_uncond + guidance_scale * (net_out_cond - net_out_uncond)
                #######################
            else:
                # 如果是 conditional 就傳 class_label，否則就兩個參數
                if class_label is not None:
                    net_out = self.network(x_t, timestep=t.to(self.device), class_label=class_label)
                else:
                    net_out = self.network(x_t, timestep=t.to(self.device))

            x_t_prev = self.var_scheduler.step(x_t, t, net_out, predictor=self.predictor)


            if return_traj:
                traj[-1] = traj[-1].cpu()
                traj.append(x_t_prev.detach())
            else:
                # Final-only sampling needs no saved history or CPU transfers.
                traj = [x_t_prev.detach()]

        if return_traj:
            return traj
        else:
            return traj[-1]


    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            "predictor": self.predictor,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu", weights_only=False)
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]
        # Older checkpoints need an explicit --predictor at sampling time.
        self.predictor = hparams.get("predictor")

        self.load_state_dict(state_dict)
