
import torch
import jax
import jax.numpy as jnp
import flax
from flax.core import freeze, unfreeze
import numpy as np

from pytorch_diffusion.models import DiT as DiT_pytorch
from jax_diffusion.models import DiT as DiT_jax

def main():
    pytorch_model_path = "/home/jupyter/DiT-in-Jax/pytorch_diffusion/pretrained_models/DiT-XL-2-512x512.pt"
    pytorch_state_dict = torch.load(pytorch_model_path)

    pytorch_model = DiT_pytorch(input_size=64, num_classes=1000)
    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()
    print(f"out_channels: {pytorch_model.out_channels}")
    print(f"final_layer.linear.weight.shape: {pytorch_model.final_layer.linear.weight.shape}")

    jax_model = DiT_jax(input_size=64, num_classes=1000)
    rng = jax.random.PRNGKey(0)

    dummy_x = jnp.ones((1, 4, 64, 64))
    dummy_t = jnp.ones((1,), dtype=jnp.int32)
    dummy_y = jnp.ones((1,), dtype=jnp.int32)
    initial_params = jax_model.init(rng, dummy_x, dummy_t, dummy_y)['params']

    with open("/home/jupyter/DiT-in-Jax/jax_diffusion/DiT-XL-2-512x512.msgpack", 'rb') as f:
        loaded_params = flax.serialization.from_bytes(initial_params, f.read())

    with open("/home/jupyter/DiT-in-Jax/jax_diffusion/DiT-XL-2-512x512.msgpack", 'rb') as f:
        jax_params = flax.serialization.from_bytes(initial_params, f.read())

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = jax.random.PRNGKey(seed)

    x_input_torch = torch.randn(1, 4, 64, 64)
    t_input_torch = torch.tensor([100])
    y_input_torch = torch.tensor([207])

    x_input_jax = jnp.asarray(x_input_torch.numpy())
    t_input_jax = jnp.asarray(t_input_torch.numpy())
    y_input_jax = jnp.asarray(y_input_torch.numpy())

    with torch.no_grad():
        output_pytorch = pytorch_model(x_input_torch, t_input_torch, y_input_torch)

    output_jax = jax_model.apply({'params': jax_params}, x_input_jax, t_input_jax, y_input_jax)

    np.testing.assert_allclose(output_pytorch.numpy(), output_jax, rtol=1e-4, atol=1e-4)
    print("success")

if __name__ == "__main__":
    main()
