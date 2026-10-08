import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)


class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """

    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_T: float = 0.02,
        mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)


class DiffusionModule(nn.Module):
    """
    A high-level wrapper of DDPM and DDIM.
    If you want to sample data based on the DDIM's reverse process, use `ddim_p_sample()` and `ddim_p_sample_loop()`.
    """

    def __init__(self, network: nn.Module, var_scheduler: BaseScheduler):
        super().__init__()
        self.network = network ### predict noise
        self.var_scheduler = var_scheduler

    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        # For image diffusion model.
        return getattr(self.network, "image_resolution", None)

    def q_sample(self, x0, t, noise=None):
        """
        sample x_t from q(x_t | x_0) of DDPM.

        Input:
            x0 (`torch.Tensor`): clean data to be mapped to timestep t in the forward process of DDPM.
            t (`torch.Tensor`): timestep
            noise (`torch.Tensor`, optional): random Gaussian noise. if None, randomly sample Gaussian noise in the function.
        Output:
            xt (`torch.Tensor`): noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x0)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Compute xt.
        alphas_prod_t = extract(self.var_scheduler.alphas_cumprod, t, x0)
        xt = torch.sqrt(alphas_prod_t) * x0 + torch.sqrt(1.0 - alphas_prod_t) * noise

        #######################

        return xt

    @torch.no_grad()
    def p_sample(self, xt, t):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.
        Input:
            xt (`torch.Tensor`): samples at arbitrary timestep t.
            t (`torch.Tensor`): current timestep in a reverse process.
        Ouptut:
            x_t_prev (`torch.Tensor`): one step denoised sample. (= x_{t-1})
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute x_t_prev.
        if isinstance(t, int):
            t = torch.tensor([t]).to(self.device)
        eps_factor = (1 - extract(self.var_scheduler.alphas, t, xt)) / (
            1 - extract(self.var_scheduler.alphas_cumprod, t, xt)
        ).sqrt()

        beta_t      = extract(self.var_scheduler.betas,           t, xt)         # β_t
        alpha_t     = extract(self.var_scheduler.alphas,          t, xt)         # α_t = 1 - β_t
        alpha_bar_t = extract(self.var_scheduler.alphas_cumprod,  t, xt)         # \bar{α}_t
        t_prev      = (t - 1).clamp(min=0)
        alpha_bar_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt) # \bar{α}_{t-1}

       # 1. predict noise (注意使用 self.network)
        eps_pred = self.network(xt, t)
        # 2. Posterior mean
        # 使用你現成的 eps_factor 變數
         # 公式: μ_θ(x_t, t) = (1 / sqrt(α_t)) * (x_t - (β_t / sqrt(1 - \bar{α}_t)) * ε_θ)
        mean = (xt - eps_factor * eps_pred) * torch.rsqrt(alpha_t)
        # 3. Posterior variance: \tilde{β}_t = β_t * (1 - \bar{α}_{t-1}) / (1 - \bar{α}_t)
        var = beta_t * (1.0 - alpha_bar_t_prev) / (1.0 - alpha_bar_t)
        # 4. Reverse step: x_{t-1} = mean + sqrt(var) * z
        noise = torch.randn_like(xt)
        nonzero_mask = (t > 0).float().view(-1, *([1] * (xt.ndim - 1)))
        x_t_prev = mean + nonzero_mask * torch.sqrt(var) * noise
        
        #######################
        return x_t_prev

    @torch.no_grad()
    def p_sample_loop(self, shape):
        """
        The loop of the reverse process of DDPM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        xt = torch.randn(shape).to(self.device) # 原本的：抽樣初始純雜訊 x_T
        x0_pred = None                          # 原本的：預設佔位符
        # 補上倒數迴圈：從 T-1, T-2, ..., 一路去噪回到 0
        for t in reversed(range(self.var_scheduler.num_train_timesteps)):
            xt = self.p_sample(xt, t)

        # 把最後完全去噪完的圖片賦值給 x0_pred
        x0_pred = xt
        
        ######################
        return x0_pred

    @torch.no_grad()
    def ddim_p_sample(self, xt, t, t_prev, eta=0.0):
        """
        One step denoising function of DDIM: $x_t{\tau_i}$ -> $x_{\tau{i-1}}$.

        Input:
            xt (`torch.Tensor`): noisy data at timestep $\tau_i$.
            t (`torch.Tensor`): current timestep (=\tau_i)
            t_prev (`torch.Tensor`): next timestep in a reverse process (=\tau_{i-1})
            eta (float): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
           x_t_prev (`torch.Tensor`): one step denoised sample. (= $x_{\tau_{i-1}}$)
        """
        ######## TODO ########
        # NOTE: This code is used for assignment 2. You don't need to implement this part for assignment 1.
        # DO NOT change the code outside this part.
        # compute x_t_prev based on ddim reverse process.
        # 抽出 (extract)」當前時間步 $t$ 對應的那個 alpha_bar 數值，以便拿來進行後續的 DDIM 數學計算
        alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt) 
        if t_prev >= 0:
            alpha_prod_t_prev = extract(self.var_scheduler.alphas_cumprod, t_prev, xt)
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        x_t_prev = xt
        # Convert both timesteps to one-dimensional LongTensors.
        # During sampling these usually contain one timestep shared by
        # the whole batch.
        t = torch.as_tensor(t, device=xt.device).reshape(-1).long()
        t_prev = torch.as_tensor(t_prev, device=xt.device).reshape(-1).long()

        # alpha_prod_t     = alpha_bar_t
        # alpha_prod_t_prev = alpha_bar_{t_prev}
        alpha_prod_t = extract(self.var_scheduler.alphas_cumprod, t, xt)

        # There is no actual timestep -1 in the scheduler.
        # For the final step, define alpha_bar_{-1} = 1.
        if torch.all(t_prev >= 0):
            alpha_prod_t_prev = extract(
                self.var_scheduler.alphas_cumprod,
                t_prev,
                xt,
            )
        else:
            alpha_prod_t_prev = torch.ones_like(alpha_prod_t)

        # Predict the noise epsilon_theta(x_t, t).
        eps_pred = self.network(xt, t)

        # Estimate the clean sample x_0: p.65
        # x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * eps_pred) / sqrt(alpha_bar_t)
        pred_x0 = (xt - torch.sqrt(torch.clamp(1.0 - alpha_prod_t, min=0.0)) * eps_pred) / torch.sqrt(alpha_prod_t)

        # DDIM standard deviation: p.64
        # sigma_t^2 = eta^2 * (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
        # eta = 0 gives deterministic DDIM sampling.
        # eta = 1 same as DDPM
        # beta_t = (1.0 - alpha_prod_t / alpha_prod_t_prev)
        sigma = eta * torch.sqrt(torch.clamp(((1.0 - alpha_prod_t_prev) / (1.0 - alpha_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)), min=0.0))

        # This is the direction determined by the predicted noise. p.61
        direction = torch.sqrt(torch.clamp(1.0 - alpha_prod_t_prev - sigma.square(), min=0.0)) * eps_pred

        # Add random noise only when eta > 0.
        noise = torch.randn_like(xt)

        # Complete DDIM update: p.61
        # x_{t_prev} = sqrt(alpha_bar_prev) * x_0_hat + direction + sigma_t * z
        x_t_prev = (torch.sqrt(alpha_prod_t_prev) * pred_x0 + direction + sigma * noise)
        ######################
        return x_t_prev

    @torch.no_grad()
    def ddim_p_sample_loop(self, shape, num_inference_timesteps=50, eta=0.0):
        """
        The loop of the reverse process of DDIM.

        Input:
            shape (`Tuple`): The shape of output. e.g., (num particles, 2)
            num_inference_timesteps (`int`): the number of timesteps in the reverse process.
            eta (`float`): correspond to η in DDIM which controls the stochasticity of a reverse process.
        Output:
            x0_pred (`torch.Tensor`): The final denoised output through the DDPM reverse process.
        """
        ######## TODO ########
        # NOTE: This code is used for assignment 2. You don't need to implement this part for assignment 1.
        # DO NOT change the code outside this part.
        # sample x0 based on Algorithm 2 of DDPM paper.
        step_ratio = self.var_scheduler.num_train_timesteps // num_inference_timesteps # decide how many steps to skip per time
        timesteps = (
            (np.arange(0, num_inference_timesteps) * step_ratio)
            .round()[::-1] # 因去噪是從最後一步倒著走回第 0 步，所以需陣列反轉
            .copy() # rearrange
            .astype(np.int64)
        )
        timesteps = torch.from_numpy(timesteps)
        prev_timesteps = timesteps - step_ratio # get pre-time step matrix

        # Start from pure Gaussian noise x_T.
        xt = torch.randn(shape, device=self.device)

        # Follow the DDIM reverse process:
        # x_t -> x_{t_prev}
        for t, t_prev in zip(timesteps, prev_timesteps): # 1 v 1 pack together
            t = t.to(self.device)
            t_prev = t_prev.to(self.device)

            xt = self.ddim_p_sample(xt, t, t_prev, eta=eta) # 當前的圖像 xt、t 與 t_prev 計算並回傳降噪過後的 xt

        x0_pred = xt

        ######################

        return x0_pred

    def compute_loss(self, x0):
        """
        The simplified noise matching loss corresponding Equation 14 in DDPM paper.
        Input:
            x0 (`torch.Tensor`): clean data
        Output:
            loss: the computed loss to be backpropagated.
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        # compute noise matching loss.
        batch_size = x0.shape[0]
        
        # 1) random choose timestep
        t = (
            torch.randint(0, self.var_scheduler.num_train_timesteps, size=(batch_size,))
            .to(x0.device)
            .long()
        )
        # 2) get GT noise, and use q_sample to get x_t
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise=noise)       
        # 3) predict noise 
        eps_pred = self.network(xt, t)
        # 4) MSE loss (eps, eps_pred)
        loss = F.mse_loss(eps_pred, noise)

        ######################
        return loss

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
        }
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        # These checkpoints contain module objects; load only trusted files.
        dic = torch.load(file_path, map_location="cpu", weights_only=False)
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
