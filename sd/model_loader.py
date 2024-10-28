import model_converter

from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion import Diffusion


def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(
        state_dict["encoder"], strict=True
    )  # If strict and at least 1 name doesn't match, it will throw an error

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict["decoder"], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict["diffusion"], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict["clip"], strict=True)

    return {"encoder": encoder, "decoder": decoder, "diffusion": diffusion, "clip": clip}