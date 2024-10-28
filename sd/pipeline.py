import torch
import numpy as np

from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = 512 // 8
LATENTS_HEIGHT = 512 // 8


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    """
    Args:
        prompt: Text prompt to generate the image from for text-to-image generation
        uncond_prompt: Text prompt to guide the model to avoid some objects in the picture or to use in CFG
        input_image: Image to be used as input to the model for image-to-image generation
        strength: How much noise to add to the original image for image-to-image generation. The higher the strength the
        more noise we have.
        do_cfg: Whether to use the classifier free guidance
        cfg_scale: Scale of the classifier free guidance:
        (output = output_cond - output_uncond) * cfg_scale + output_uncond
        sampler_name: Name of the sampler to be used
        n_inference_steps: Number of inference steps (50 is good for DDPM sampler)
        models: Dictionary containing the models
        seed: Seed for the random number generator
        device: Device to be used
        idle_device: Device to be used for offloading unused models to free up the space on the main device
        tokenizer: Tokenizer to be used
    """
    with torch.no_grad():

        if not (0 < strength <= 1):
            raise ValueError("strength must be in (0, 1]")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        # output = cfg_scale(output_cond - output_uncond) + output_uncond
        # Uncoditional prompt = uncond_prompt is usually an empty string, but can be used to guide the model to avoid
        # some objects in the picture.

        if do_cfg:
            # output = cfg_scale(output_cond - output_uncond) + output_uncond
            # Uncoditional prompt = uncond_prompt is usually an empty string, but can be used to guide the model
            # to avoid some objects in the picture.
            # Convert the prompt to tokens, using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, Seq_length)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_length) -> (Batch_size, Seq_length, Dim=768)
            cond_context = clip(cond_tokens)

            # Convert the uncond_prompt to tokens, using the tokenizer
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, Seq_length)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_length) -> (Batch_size, Seq_length, Dim=768)
            uncond_context = clip(uncond_tokens)

            # (Batch_size, Seq_length, Dim=768) -> (Batch_size=2, Seq_length=77, Dim=768)
            context = torch.cat([cond_context, uncond_context], dim=0)
        else:
            # Here the classifier free guidance is not used, since we do not contrast it with the unconditional prompt
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, Seq_length)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_size=1, Seq_length=77, Dim=768)
            context = clip(tokens)

        # Move the model to the idle device. Usually offload the GPU back to CPU to free up the GPU for other tasks.
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler: {sampler_name}")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channels)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channels) -> (Batch_Size, Channels, Height, Width)
            input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            # Run the image through the encoder of the VAE
            latents = encoder(input_image_tensor, encoder_noise)

            # Add noise to the latents of the original image we start with
            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            # If we are doing text-to-image, start with random noise N(0, I)
            latents = torch.randn(latents_shape, generator=generator, device=device)

        # We start at 1000 timesteps and remove the noise at 1000 / n_inference_steps = 1000 / 50 = 20 timesteps, e.g.
        # 1000 -> 980 -> 960 -> 940 -> ... -> 20 -> 0 timesteps
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # Timestep -> (1, 320) timestep embedding, similar to positional encoding in transformers
            time_embedding = get_time_embedding(timestep).to(device)

            # (Batch_size, 4, Latents_Height, Latents_Width)
            model_input = latents

            if do_cfg:
                # (Batch_size, 4, Latents_Height, Latents_Width) -> (2 * Batch_size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is predicted noise by the UNet
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2, dim=0)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # Remove the noise predicted by the UNet
            latents = sampler.step(timestep, latents, model_output)

        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        # Decode the latents to produce the final image
        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_size, Channels, Height, Width) -> (Batch_size, Height, Width, Channels) to save on disk
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


def rescale(x, old_range, new_range, clamp=False):
    """
    Args:
        x: Tensor to be rescaled
        old_range: Old range of the tensor
        new_range: New range of the tensor
        clamp: Whether to clamp the values of the tensor
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min

    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep):
    # Positional embeddings taken from the transformer architecture
    # PE(pos, 2i) = sin(pos / 10000^(2i / d_model)) and PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
    # (1, 160)
    freqs = torch.pow(10_000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 160) -> (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
