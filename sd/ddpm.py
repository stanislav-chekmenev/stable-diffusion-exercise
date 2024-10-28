import torch
import numpy as np


class DDPMSampler:

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ...] - Cumulative product of alphas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        self.num_inference_steps = num_inference_steps
        # 999, 998, 997, ..., 0 - 1000 steps
        # 999, 999 - 20, 999 - 40, ... - 50 steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_prev_timestep(self, timestep: int) -> int:
        """
        Get the previous timestep
        Args:
            timestep: Current timestep
        """
        return timestep - self.num_training_steps // self.num_inference_steps

    def _get_variance_timestep(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_prev_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Compute variance using the formulas 6, 7 of the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """
        Step through the diffusion process and remove the noise predicted by the UNet
        Args:
            timestep: Timestep to remoe the noise at
            latents: Latents to remove the noise from
            model_output: Noise predicted by the UNet
        """
        t = timestep
        prev_t = self._get_prev_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Compute the predicted original sample using formula 15 of DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        pred_original_sample = (latents - beta_prod_t**0.5 * model_output) / (alpha_prod_t**0.5)

        # Compute coefficients for pred_original_sample and current sample x_t based on formula 7 of DDPM paper
        # (https://arxiv.org/pdf/2006.11239.pdf)
        pred_original_sample_coeff = (alpha_prod_t_prev**0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = (current_alpha_t**0.5 * beta_prod_t_prev) / beta_prod_t

        # Compute the predicted previous sample using formula 7 (latents is the x_t) of DDPM paper
        # (https://arxiv.org/pdf/2006.11239.pdf)
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        noise = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator=self.generator, device=device, dtype=model_output.dtype)
            # Compute the standard deviation of the noise to be added
            stdev = self._get_variance_timestep(t) ** 0.5
            # N(0, I) -> N(mean, sigma^2)
            # X = mean + sigma * Z, where Z ~ N(0, I)
            noise *= stdev

        # Add noise to the predicted previous sample
        pred_prev_sample = pred_prev_sample + noise

        return pred_prev_sample

    def set_strength(self, strength: float = 1):
        """
        Set the strength of the noise to be added
        Args:
            strength: Strength of the noise to be added
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        Add noise to the original samples
        Args:
            original_samples: Original samples to add noise to, which is the latent space of the VAE
            timesteps: Timesteps to add noise
        """
        alpha_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(device=original_samples.device)

        sqrt_alpha_cumprod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_cumprod = sqrt_alpha_cumprod.flatten()

        # Add dimension to sqrt_alpha_cumprod to match the dimension than original_samples
        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)

        # Get the standard deviation of the noise to be added
        sqrt_one_minus_alpha_cumprod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.flatten()

        while len(sqrt_one_minus_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )

        # Add noise to the origianl samples (our latents)
        # According to the equation 4 of the DDPM paper (https://arxiv.org/pdf/2006.11239.pdf)
        noisy_samples = (sqrt_alpha_cumprod * original_samples) + sqrt_one_minus_alpha_cumprod * noise
        return noisy_samples
