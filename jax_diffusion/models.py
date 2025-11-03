
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import linen as nn
import jax.numpy as jnp

class Attention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = False

    @nn.compact
    def __call__(self, x):
        B, N, C = x.shape
        head_dim = self.dim // self.num_heads
        qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose((0, 1, 3, 2))) * (head_dim ** -0.5)
        attn = nn.softmax(attn, axis=-1)

        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)
        x = nn.Dense(self.dim)(x)
        return x

class Mlp(nn.Module):
    in_features: int
    hidden_features: int = None
    out_features: int = None
    act_layer: callable = nn.gelu
    drop: float = 0.0

    @nn.compact
    def __call__(self, x):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features
        x = nn.Dense(features=hidden_features)(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=True)
        x = nn.Dense(features=out_features)(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=True)
        return x

class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        B, C, H, W = x.shape
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(features=self.embed_dim, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size), padding='VALID', use_bias=self.bias)(x)
        x = x.reshape(B, -1, self.embed_dim)
        return x


def modulate(x, shift, scale):
    return x * (1 + jnp.expand_dims(scale, 1)) + jnp.expand_dims(shift, 1)


class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256

    @nn.compact
    def __call__(self, t):
        mlp = nn.Sequential([
            nn.Dense(self.hidden_size),
            nn.silu,
            nn.Dense(self.hidden_size),
        ])
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = mlp(t_freq)
        return t_emb

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half
        )
        args = jnp.expand_dims(t, -1) * freqs
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float

    @nn.compact
    def __call__(self, labels, train, force_drop_ids=None):
        use_cfg_embedding = self.dropout_prob > 0
        embedding_table = nn.Embed(self.num_classes + use_cfg_embedding, self.hidden_size)
        if (train and use_cfg_embedding) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)
        return embeddings

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = jax.random.bernoulli(self.make_rng('dropout'), self.dropout_prob, labels.shape)
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels


class DiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    @nn.compact
    def __call__(self, x, c):
        norm1 = nn.LayerNorm(epsilon=1e-6, use_bias=False)
        attn = Attention(self.hidden_size, num_heads=self.num_heads, qkv_bias=True)
        norm2 = nn.LayerNorm(epsilon=1e-6, use_bias=False)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        mlp = Mlp(in_features=self.hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.gelu, drop=0)
        adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(6 * self.hidden_size)
        ])

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(adaLN_modulation(c), 6, axis=1)
        x = x + jnp.expand_dims(gate_msa, 1) * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + jnp.expand_dims(gate_mlp, 1) * mlp(modulate(norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):
        norm_final = nn.LayerNorm(epsilon=1e-6, use_bias=False)
        linear = nn.Dense(self.patch_size * self.patch_size * self.out_channels)
        adaLN_modulation = nn.Sequential([
            nn.silu,
            nn.Dense(2 * self.hidden_size)
        ])

        shift, scale = jnp.split(adaLN_modulation(c), 2, axis=1)
        x = modulate(norm_final(x), shift, scale)
        x = linear(x)
        return x


class DiT(nn.Module):
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    learn_sigma: bool = True

    @nn.compact
    def __call__(self, x, t, y, train=False):
        out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels
        x_embedder = PatchEmbed(img_size=self.input_size, patch_size=self.patch_size, in_chans=self.in_channels, embed_dim=self.hidden_size, bias=True)
        t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        y_embedder = LabelEmbedder(num_classes=self.num_classes, hidden_size=self.hidden_size, dropout_prob=self.class_dropout_prob)
        
        num_patches = (self.input_size // self.patch_size) ** 2
        pos_embed = self.param('pos_embed', nn.initializers.zeros, (1, num_patches, self.hidden_size))

        x = x_embedder(x) + pos_embed
        t = t_embedder(t)
        y = y_embedder(y, train)
        c = t + y

        for _ in range(self.depth):
            x = DiTBlock(hidden_size=self.hidden_size, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio)(x, c)

        x = FinalLayer(hidden_size=self.hidden_size, patch_size=self.patch_size, out_channels=out_channels)(x, c)
        x = self.unpatchify(x, out_channels)
        return x

    def unpatchify(self, x, out_channels):
        c = out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape((x.shape[0], c, h * p, w * p))
        return imgs

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,
}
