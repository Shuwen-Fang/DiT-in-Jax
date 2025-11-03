import pytest
import jax
import jax.numpy as jnp
import optax
from copy import deepcopy
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jax_diffusion.models import DiT_models
from jax_diffusion.diffusion import create_diffusion
from jax_diffusion.train import TrainState

@pytest.fixture
def setup():
    """Provides a basic setup for the training tests."""
    latent_size = 32  # Corresponds to image_size 256
    batch_size = 2
    
    model = DiT_models['DiT-XL/2'](
        input_size=latent_size,
        num_classes=1000
    )
    
    opt = optax.adamw(learning_rate=1e-4, weight_decay=0)

    rng = jax.random.PRNGKey(0)
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
    
    diffusion = create_diffusion(timestep_respacing="")
    
    dummy_x_batch = jnp.ones((batch_size, 4, latent_size, latent_size))
    dummy_t_batch = jnp.ones((batch_size,), dtype=jnp.int32)
    dummy_y_batch = jnp.ones((batch_size,), dtype=jnp.int32)
    dummy_model_kwargs = dict(y=dummy_y_batch)

    return {
        "state": state,
        "model": model,
        "diffusion": diffusion,
        "dummy_x": dummy_x_batch,
        "dummy_t": dummy_t_batch,
        "dummy_kwargs": dummy_model_kwargs,
        "rng": rng
    }

def test_loss_function_and_gradients(setup):
    """
    Tests that the loss function produces a scalar output and that gradients
    can be computed and have the correct shape.
    """
    state = setup["state"]
    model = setup["model"]
    diffusion = setup["diffusion"]
    dummy_x = setup["dummy_x"]
    dummy_t = setup["dummy_t"]
    dummy_kwargs = setup["dummy_kwargs"]
    rng = setup["rng"]

    def loss_fn(params, x_start, t, model_kwargs, rng_key):
        model_fn = lambda x, t, y: model.apply({'params': params}, x, t, y)
        terms = diffusion.training_losses(model_fn, x_start, t, rng_key, model_kwargs)
        return terms["loss"].mean()

    rng, loss_rng = jax.random.split(rng)
    loss_value = loss_fn(state.params, dummy_x, dummy_t, dummy_kwargs, loss_rng)
    assert loss_value.shape == (), f"Loss function should return a scalar, but got shape {loss_value.shape}"

    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    rng, grad_rng = jax.random.split(rng)
    loss_value, grads = grad_fn(state.params, dummy_x, dummy_t, dummy_kwargs, grad_rng)
    
    shapes_match = jax.tree_util.tree_map(lambda g, p: g.shape == p.shape, grads, state.params)
    assert jax.tree_util.tree_all(shapes_match), "Gradient shapes do not match parameter shapes"

    total_grad_norm = jax.tree_util.tree_reduce(lambda acc, g: acc + jnp.sum(jnp.abs(g)), grads, 0.0)
    assert total_grad_norm > 0, "Gradients are all zeros!"

def test_parameter_update(setup):
    """
    Tests that applying gradients changes the model parameters.
    """
    state = setup["state"]
    model = setup["model"]
    diffusion = setup["diffusion"]
    dummy_x = setup["dummy_x"]
    dummy_t = setup["dummy_t"]
    dummy_kwargs = setup["dummy_kwargs"]
    rng = setup["rng"]

    def loss_fn(params, x_start, t, model_kwargs, rng_key):
        model_fn = lambda x, t, y: model.apply({'params': params}, x, t, y)
        terms = diffusion.training_losses(model_fn, x_start, t, rng_key, model_kwargs)
        return terms["loss"].mean()
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=False)
    rng, grad_rng = jax.random.split(rng)
    _, grads = grad_fn(state.params, dummy_x, dummy_t, dummy_kwargs, grad_rng)

    initial_params = state.params
    new_state = state.apply_gradients(grads=grads)
    
    params_changed = False
    for p1, p2 in zip(jax.tree_util.tree_leaves(new_state.params), jax.tree_util.tree_leaves(initial_params)):
        if not jnp.array_equal(p1, p2):
            params_changed = True
            break
    assert params_changed, "Parameters did not change after update!"
