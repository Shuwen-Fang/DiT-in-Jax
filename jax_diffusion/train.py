import argparse
import logging
import os
from time import time
from copy import deepcopy
from PIL import Image
import flax

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.core import freeze, unfreeze
from flax.training import train_state
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from diffusers.models import AutoencoderKL
from jax_diffusion.models import DiT_models
from jax_diffusion.diffusion import create_diffusion


class TrainState(train_state.TrainState):
    ema_params: any

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger

def update_ema(ema_params, params, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    return jax.tree_util.tree_map(lambda ema, p: ema * decay + p * (1 - decay), ema_params, params)

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def main(args):
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(os.listdir(args.results_dir))
    model_string_name = args.model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    assert args.image_size % 8 == 0, "Image size has to be divisble by 8"
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    opt = optax.adamw(learning_rate=1e-4, weight_decay=0)

    rng = jax.random.PRNGKey(args.global_seed)
    rng, params_rng = jax.random.split(rng)
    dummy_x = jnp.ones((1, 4, latent_size, latent_size))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_y = jnp.ones((1,), dtype=jnp.int32)
    params = model.init(params_rng, dummy_x, dummy_t, dummy_y)['params']
    
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=opt,
        ema_params=deepcopy(params)
    )

    logger.info(f"DiT Parameters: {sum(p.size for p in jax.tree_util.tree_leaves(state.params)):,}")

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
    for param in vae.parameters():
        param.requires_grad = False
    
    diffusion = create_diffusion(timestep_respacing="")

    # Prepare models for training:
    # TODO: update ema

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    


    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x = jnp.asarray(x_latent.numpy())
            y = jnp.asarray(y)
            
            rng, t_rng, train_step_rng = jax.random.split(rng, 3)
            t = jax.random.randint(t_rng, (x.shape[0],), 0, diffusion.num_timesteps)
            model_kwargs = dict(y=y)
            
            state, loss = train_step(state, x, t, model_kwargs, train_step_rng, diffusion)
            
            running_loss += loss
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                avg_loss = running_loss / log_steps
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.msgpack"
                with open(checkpoint_path, "wb") as f:
                    f.write(flax.serialization.to_bytes(state))
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    logger.info("Done!")

def _train_step(state, x, t, model_kwargs, rng_key, diffusion):
    """
    A single training step.
    """
    def loss_fn(params):
        model_fn = lambda x, t, y: state.apply_fn({'params': params}, x, t, y)
        terms = diffusion.training_losses(model_fn, x, t, rng_key, model_kwargs)
        return terms["loss"].mean()

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Update EMA
    state = state.replace(ema_params=update_ema(state.ema_params, state.params))
    
    return state, loss

train_step = jax.jit(_train_step, static_argnums=5)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--ckpt-every", type=int, default=10)
    args = parser.parse_args()
    main(args)
