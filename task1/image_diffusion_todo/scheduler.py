from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear",
        betas: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if betas is not None:
            # A pretrained checkpoint already contains its exact noise schedule.
            betas = betas.detach().clone()
            if betas.ndim != 1 or betas.numel() != num_train_timesteps:
                raise ValueError("Saved betas must have one entry per training timestep.")
            if not torch.isfinite(betas).all() or not ((betas > 0) & (betas < 1)).all():
                raise ValueError("Saved betas must be finite and strictly between 0 and 1.")
        elif mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # steps=num_train_timesteps ( 擴散模型在訓練時使用的總時間步數 T )
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            s = 0.008
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            steps = torch.arange(num_train_timesteps + 1, dtype=torch.float32)
            alpha_bar = torch.cos(((steps / num_train_timesteps + s) / (1 + s)) * (torch.pi / 2)) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            T = self.num_train_timesteps
            betas = 1 - (alpha_bar[1 : T + 1] / alpha_bar[0 : T])
            # betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1] 
            # [1:] 代表「從索引 1 一路切到最尾端」，[:-1] 代表「從頭開始切，切到倒數第 1 個元素之前（不含最後一個）」。
            # 3. Clip beta_t to at most 0.999 (singularity at t = T).
            betas = torch.clamp(betas, max = 0.999) # 將輸入的所有元素數值限制在指定的最小值固定為正)與最大值之間
            # 4. Return betas as a tensor of shape [num_train_timesteps].
            #raise NotImplementedError("TODO: Implement cosine beta schedule here!")
               
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        
        self.schedule_mode = mode      

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    
    
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    
    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, alpha_bar_t, and alpha_bar_{t-1} from the
        #    scheduler (ᾱ_{t-1} = 1 at t = 0).
        # 確保 t 是一維 LongTensor，形狀為 [B]
        t = t.reshape(-1).long().to(x_t.device)
        # 取得每個 batch 對應 timestep 的參數，
        # extract 會自動 reshape 成 [B, 1, 1, 1]
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        # 取得 alpha_bar_{t-1}
        # t = 0 時先使用 index 0，之後再替換為 tensor 形式的 1.0
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t)
        # 數學上 alpha_bar_{-1} = 1
        alpha_bar_prev = torch.where((t == 0).reshape(-1, 1, 1, 1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)
        # 2. Convert the predicted noise into the predicted clean sample
        #       x̂₀ = (x_t - √(1-ᾱ_t) * ε̂_θ) / √ᾱ_t
        x0_pred = ( x_t - torch.sqrt( 1.0 - alpha_bar_t) * eps_theta) / torch.sqrt( alpha_bar_t)
        #    and clamp it to [-1, 1].
        x0_pred = torch.clamp( x0_pred, -1.0, 1.0)
        # 3. Compute the posterior mean
        #       \tilde{μ}_t = (√ᾱ_{t-1}·β_t/(1-ᾱ_t)) * x̂₀ + (√α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)) * x_t.
        # x0, xt的權重係數
        coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t))
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
        posterior_mean = coef_x0 * x0_pred + coef_xt * x_t
        # 4. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        posterior_var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t)
        # 加上 clamp(min=0.0)，防止 t=0 或數值浮點誤差導致根號內為負而產生 NaN
        posterior_std = torch.sqrt(torch.clamp(posterior_var, min=0.0))
        sigmas = torch.sqrt(posterior_var) # 不為負
        # 5. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0. # 反向採樣的隨機探索項
        noise = torch.randn_like(x_t) # 抽樣標準高斯隨機變數 z ~ N(0, I)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1) # 當 t == 0 時，已經到達最後一步，不應該再加入隨機噪聲
        #noise_term = (nonzero_mask) * torch.sqrt(posterior_var) * noise
        # 6. Return the final sample at t-1.
        # 簡化原先重複計算的根號變異數，直接複用已受 clamp 保護的 posterior_std
        sample_prev = posterior_mean + nonzero_mask * posterior_std * noise
        #sample_prev = posterior_mean + noise_term
        #######################
        return sample_prev

    
    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 統一將 t 轉換為 1D LongTensor 並搬移至正確裝置，避免測試中型別不相容引發報錯
        t = t.reshape(-1).long().to(x_t.device)
        # Remember to clamp x0_pred to [-1, 1], as in step_predict_noise.
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        # 取得posterior mean所需變數，每個 batch 的 t 可能不同，所以使用 extract
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        # 取得 alpha_bar_{t-1}
        t_prev = (t - 1).clamp(min=0)
        alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t,)
        # t = 0 時，定義 alpha_bar_{t-1} = 1
        #alpha_bar_prev = torch.where(t.reshape(-1, 1, 1, 1) == 0, torch.ones_like(alpha_bar_prev), alpha_bar_prev,)
        # 將 (t.reshape(...) == 0) 改寫為標準的 (t == 0).reshape(-1, 1, 1, 1)，與 step_predict_noise 保持一致的布林廣播行為
        alpha_bar_prev = torch.where((t == 0).reshape(-1, 1, 1, 1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)
        # 3. 計算 posterior mean
        coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t))
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
        posterior_mean = coef_x0 * x0_pred + coef_xt * x_t
        # 4. 計算 posterior variance
        posterior_var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t)
        # 加上 clamp(min=0.0) 安全防護，避免微小負浮點數被送入開根號
        posterior_std = torch.sqrt(torch.clamp(posterior_var, min=0.0))
        # 5. t != 0 時才加入隨機噪音
        noise = torch.randn_like(x_t) 
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1) # 當 t == 0 時，已經到達最後一步，不應該再加入隨機噪聲
        #noise_term = (nonzero_mask) * torch.sqrt(posterior_var) * noise
        # 6. 得到 x_{t-1}
        sample_prev = posterior_mean + nonzero_mask * posterior_std * noise
        #sample_prev = posterior_mean + noise_term
        #######################
        return sample_prev

    
    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 統一將 t 轉換為 1D LongTensor 並搬移至正確裝置，避免測試中型別不相容引發報錯
        t = t.reshape(-1).long().to(x_t.device)
        posterior_mean = mean_theta
        beta_t = extract(self.betas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        t_prev = (t - 1).clamp(min = 0)
        # 先取得一般情況下的 alpha_bar_{t-1}
        alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x_t)
        # 修正 t = 0 的特殊情況
        # 數學上 alpha_bar_{-1} 定義為 1
        #alpha_bar_prev = torch.where(t.reshape(-1, 1, 1, 1) == 0, torch.ones_like(alpha_bar_prev), alpha_bar_prev)
        # 統一布林條件的 reshape 寫法
        alpha_bar_prev = torch.where((t == 0).reshape(-1, 1, 1, 1), torch.ones_like(alpha_bar_prev), alpha_bar_prev)
        posterior_var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t)
        posterior_std = torch.sqrt(torch.clamp(posterior_var, min=0.0))
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1) # 當 t == 0 時，已經到達最後一步，不應該再加入隨機噪聲
        #noise_term = (nonzero_mask) * torch.sqrt(posterior_var) * noise
        sample_prev = posterior_mean + nonzero_mask * posterior_std * noise
        #sample_prev = posterior_mean + noise_term
        #######################
        return sample_prev

    
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn_like(x_0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        alphas_prod_t = extract(self.alphas_cumprod, t, x_0)
        x_t = torch.sqrt(alphas_prod_t) * x_0 + torch.sqrt(1.0 - alphas_prod_t) * eps
        #######################

        return x_t, eps

class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode: str = "linear",
        num_inference_timesteps: int = 50,
        eta: float = 0.0,
        trained_scheduler: Optional[BaseScheduler] = None,
    ):
        if trained_scheduler is not None and num_train_timesteps != trained_scheduler.num_train_timesteps:
            raise ValueError("DDIM must use the checkpoint's number of training timesteps.")
        super().__init__(
            num_train_timesteps, beta_1, beta_T, mode,
            betas=None if trained_scheduler is None else trained_scheduler.betas,
        )
        if trained_scheduler is not None:
            # Copy before the timestep TODO, which may cache inference coefficients.
            self.alphas = trained_scheduler.alphas.detach().clone()
            self.alphas_cumprod = trained_scheduler.alphas_cumprod.detach().clone()
            self.schedule_mode = getattr(trained_scheduler, "schedule_mode", None)
        else:
            self.schedule_mode = mode
        self.eta = float(eta)
        self.set_inference_timesteps(num_inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Define the inference schedule (a subset of training timesteps, descending order).
        Inputs:
            num_inference_timesteps (int): number of inference steps (e.g., 50).
        """
        ######## TODO ########
        # Hint:
        #   - Define the DDIM inference schedule based on the given num_inference_timesteps.
        #   - The schedule should be a subset of training timesteps, ordered in descending fashion.
        #   - Store the result in `self.timesteps` (as a torch tensor) 
        #   - Store the step ratio in `self._ddim_step_ratio` for later use when computing previous t.
        #   - Compute a `step_ratio` that maps inference steps to training steps.
        # DO NOT change the code outside this part.
        raise NotImplementedError("TODO")
        #######################

    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor, predictor: str):
        """
        One step DDIM update: x_t -> x_{t_prev} with deterministic/stochastic control via eta.

        Input:
            x_t: [B,C,H,W]
            t: current absolute timestep index
            eps_theta: predicted noise
            predictor: predictor type
        Output:
            sample_prev: x at previous inference timestep
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        assert predictor == "noise", "In assignment 2, we only implement DDIM with noise predictor."
        sample_prev = None
        #######################
        return sample_prev
