import argparse
import os

import jax
import jax.numpy as jnp
import flax
from PIL import Image
import torch
from diffusers.models import AutoencoderKL

from jax_diffusion.models import DiT_models
from jax_diffusion.diffusion import create_diffusion
from jax_diffusion.train import TrainState, create_logger, update_ema

def main(args):
    model = DiT_models[args.model](
        input_size=args.image_size // 8,
        num_classes=args.num_classes
    )

    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.ones((1, 4, args.image_size // 8, args.image_size // 8))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_y = jnp.ones((1,), dtype=jnp.int32)
    params = model.init(rng, dummy_x, dummy_t, dummy_y)['params']
    dummy_state = TrainState(step=0, apply_fn=model.apply, params=params, tx=None, opt_state=None, ema_params=params)

    # Load checkpoint
    with open(args.ckpt_path, 'rb') as f:
        state = flax.serialization.from_bytes(dummy_state, f.read())

    diffusion = create_diffusion(timestep_respacing="")

    sample_fn = jax.jit(lambda rng: diffusion.p_sample_loop(
        lambda x, t, y: model.apply({'params': state.ema_params}, x, t, y),
        (args.num_samples, 4, args.image_size // 8, args.image_size // 8),
        progress=True,
        rng_key=rng,
        model_kwargs=dict(y=jnp.array([args.class_label] * args.num_samples))
    ))

    # Generate samples
    rng = jax.random.PRNGKey(args.seed)
    samples = sample_fn(rng)

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")

    samples = 1 / 0.18215 * samples
    samples = torch.from_numpy(samples)
    decoded_samples = vae.decode(samples).sample

    os.makedirs(args.output_dir, exist_ok=True)
    for i, sample in enumerate(decoded_samples):
        sample = (sample / 2 + 0.5).clamp(0, 1)
        sample = (sample * 255).permute(1, 2, 0).to(torch.uint8).numpy()
        img = Image.fromarray(sample)
        img.save(os.path.join(args.output_dir, f'sample_{i}.png'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="samples")
    args = parser.parse_args()
    main(args)
