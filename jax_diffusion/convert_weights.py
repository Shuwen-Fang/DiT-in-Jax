import torch
import numpy as np
from jax_diffusion.models import DiT_XL_2
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
import flax

def main():
    pytorch_model = torch.load("/home/jupyter/DiT-in-Jax/pytorch_diffusion/pretrained_models/DiT-XL-2-512x512.pt")
    
    rng = jax.random.PRNGKey(0)
    model_jax = DiT_XL_2(input_size=64)
    params = model_jax.init(rng, jnp.ones((1, 4, 64, 64)), jnp.ones((1,), dtype=jnp.int32), jnp.ones((1,), dtype=jnp.int32))['params']

    new_params = unfreeze(params)

    new_params['LabelEmbedder_0']['Embed_0']['embedding'] = pytorch_model['y_embedder.embedding_table.weight'].numpy()
    new_params['TimestepEmbedder_0']['Dense_0']['kernel'] = pytorch_model['t_embedder.mlp.0.weight'].numpy().T
    new_params['TimestepEmbedder_0']['Dense_0']['bias'] = pytorch_model['t_embedder.mlp.0.bias'].numpy()
    new_params['TimestepEmbedder_0']['Dense_1']['kernel'] = pytorch_model['t_embedder.mlp.2.weight'].numpy().T
    new_params['TimestepEmbedder_0']['Dense_1']['bias'] = pytorch_model['t_embedder.mlp.2.bias'].numpy()

    new_params['PatchEmbed_0']['Conv_0']['kernel'] = pytorch_model['x_embedder.proj.weight'].numpy().transpose(2, 3, 1, 0)
    new_params['PatchEmbed_0']['Conv_0']['bias'] = pytorch_model['x_embedder.proj.bias'].numpy()
    new_params['pos_embed'] = pytorch_model['pos_embed'].numpy()

    for i in range(28):
        new_params[f'DiTBlock_{i}']['Attention_0']['Dense_0']['kernel'] = pytorch_model[f'blocks.{i}.attn.qkv.weight'].numpy().T
        new_params[f'DiTBlock_{i}']['Attention_0']['Dense_0']['bias'] = pytorch_model[f'blocks.{i}.attn.qkv.bias'].numpy()
        new_params[f'DiTBlock_{i}']['Attention_0']['Dense_1']['kernel'] = pytorch_model[f'blocks.{i}.attn.proj.weight'].numpy().T
        new_params[f'DiTBlock_{i}']['Attention_0']['Dense_1']['bias'] = pytorch_model[f'blocks.{i}.attn.proj.bias'].numpy()
        new_params[f'DiTBlock_{i}']['Mlp_0']['Dense_0']['kernel'] = pytorch_model[f'blocks.{i}.mlp.fc1.weight'].numpy().T
        new_params[f'DiTBlock_{i}']['Mlp_0']['Dense_0']['bias'] = pytorch_model[f'blocks.{i}.mlp.fc1.bias'].numpy()
        new_params[f'DiTBlock_{i}']['Mlp_0']['Dense_1']['kernel'] = pytorch_model[f'blocks.{i}.mlp.fc2.weight'].numpy().T
        new_params[f'DiTBlock_{i}']['Mlp_0']['Dense_1']['bias'] = pytorch_model[f'blocks.{i}.mlp.fc2.bias'].numpy()
        new_params[f'DiTBlock_{i}']['Dense_0']['kernel'] = pytorch_model[f'blocks.{i}.adaLN_modulation.1.weight'].numpy().T
        new_params[f'DiTBlock_{i}']['Dense_0']['bias'] = pytorch_model[f'blocks.{i}.adaLN_modulation.1.bias'].numpy()

    print(f"pytorch_model['final_layer.linear.weight'].numpy().T.shape: {pytorch_model['final_layer.linear.weight'].numpy().T.shape}")
    new_params['FinalLayer_0']['Dense_0']['kernel'] = pytorch_model['final_layer.linear.weight'].numpy().T
    new_params['FinalLayer_0']['Dense_0']['bias'] = pytorch_model['final_layer.linear.bias'].numpy()
    new_params['FinalLayer_0']['Dense_1']['kernel'] = pytorch_model['final_layer.adaLN_modulation.1.weight'].numpy().T
    new_params['FinalLayer_0']['Dense_1']['bias'] = pytorch_model['final_layer.adaLN_modulation.1.bias'].numpy()

    with open('/home/jupyter/DiT-in-Jax/jax_diffusion/DiT-XL-2-512x512.msgpack', 'wb') as f:
        f.write(flax.serialization.to_bytes(freeze(new_params)))

if __name__ == "__main__":
    main()