import os
from PIL import Image

import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.optim import Adam
from icecream import ic
from tqdm import tqdm

from functools import partial

import wandb

class Generator(nn.Module):
    '''
    Generator network for a DCGAN. Takes 128 Dimensional latent vector as input and outputs a 64x64x3 image.
    '''
    training: bool

    @nn.compact
    def __call__(self, z):
        '''
        z: (n, 1, 1, 64) dimensional latent vector

        returns:
            (n, 64, 64, 3) image
        '''
        x = nn.Dense(128)(z)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(512)(z) # (n, 1, 1, 64) -> (n, 1, 1, 512)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)
        x = nn.Dense(1024)(x)
        x = nn.relu(x)
        x = nn.Dense(2048)(x) # (n, 1, 1, 512) -> (n, 1, 1, 2048)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features = 64, kernel_size = (4, 4), strides = (1, 1), padding = 'VALID')(x) # (n, 1, 1, 2048) -> (n, 4, 4, 64)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 4, 4, 64) -> (n, 8, 8, 32)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features = 16, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 8, 8, 32) -> (n, 16, 16, 16)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features = 8, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 16, 16, 16) -> (n, 32, 32, 3)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features = 3, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 32, 32, 3) -> (n, 64, 64, 3)
        x = nn.tanh(x)

        return x


class Discriminator(nn.Module):
    '''
    Discriminator network for a DCGAN. Takes 64x64x3 image as input and outputs a 1-dimensional probability.
    '''
    training: bool

    @nn.compact
    def __call__(self, image):
        '''
        x: (n, 64, 64, 3) image
        '''
        x = nn.Conv(features = 16, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(image) # (n, 64, 64, 3) -> (n, 32, 32, 16)
        x = nn.leaky_relu(x)
        x = nn.Conv(features = 24, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 32, 32, 16) -> (n, 16, 16, 32)
        x = nn.leaky_relu(x)
        x = nn.Conv(features = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 16, 16, 32) -> (n, 8, 8, 64)
        x = nn.leaky_relu(x)

        # Flatten the image
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(1024)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(256)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(64)(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(1)(x)

        return x

@jax.vmap
def bce_logits_loss(logit, label):
    return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))

def loss_generator(params_G, params_D, batch, rng_key, variables_G, variables_D):
    z = jax.random.normal(rng_key, shape=(batch.shape[0], 1, 1, 64))

    fake_batch, variables_G = Generator(training = True).apply({
            'params' : params_G,
            'batch_stats' : variables_G['batch_stats'],
    }, z, mutable=['batch_stats'])

    fake_logits = Discriminator(training = True).apply({
            'params' : params_D,
    }, fake_batch)

    real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
    return jnp.mean(bce_logits_loss(fake_logits, real_labels)), (variables_G, variables_D)

def loss_discriminator(params_D, params_G, batch, rng_key, variables_G, variables_D):
    z = jax.random.normal(rng_key, shape=(batch.shape[0], 1, 1, 64))

    fake_batch, variables_G = Generator(training = True).apply({
            'params' : params_G,
            'batch_stats' : variables_G['batch_stats'],
    }, z, mutable=['batch_stats'])

    real_logits = Discriminator(training = True).apply({
            'params' : params_D,
    }, batch)

    fake_logits = Discriminator(training = True).apply({
            'params' : params_D,
    }, fake_batch)

    real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
    real_loss = bce_logits_loss(real_logits, real_labels)

    fake_labels = jnp.zeros((batch.shape[0],), dtype=jnp.int32)
    fake_loss = bce_logits_loss(fake_logits, fake_labels)

    return jnp.mean(real_loss + fake_loss), (variables_G, variables_D)

def train_step(rng_key, variables_G, variables_D, optimizer_G, optimizer_D, batch):
    rng_key, rng_G, rng_D = jax.random.split(rng_key, 3)

    (G_loss, (variables_G, variables_D)), grad_G = jax.value_and_grad(loss_generator, has_aux=True)(
            optimizer_G.target, optimizer_D.target, batch, rng_G, variables_G, variables_D)

    optimizer_G = optimizer_G.apply_gradient(grad_G)

    (D_loss, (variables_G, variables_D)), grad_D = jax.value_and_grad(loss_discriminator, has_aux=True)(
            optimizer_D.target, optimizer_G.target, batch, rng_D, variables_G, variables_D)

    optimizer_D = optimizer_D.apply_gradient(grad_D)

    return rng_key, variables_G, variables_D, optimizer_G, optimizer_D, G_loss, D_loss

def make_dataset(folder_path, batch_size):
    images = []

    for file in os.listdir(folder_path):
        if file.endswith(".png"):
            images.append(jnp.array(Image.open(os.path.join(folder_path, file)).resize((64, 64))))

    images = jnp.array(images)
    images = images.astype(jnp.float32) / 255.0

    # Shuffle the images
    rng = jax.random.PRNGKey(0)
    rng, permutation = jax.random.split(rng, 2)
    images = jax.random.permutation(permutation, images, independent=True)

    # Split the images into batches
    dataset = []

    for i in range(0, images.shape[0], batch_size):
        dataset.append(images[i:i+batch_size])

    return dataset

def main():
    dataset = make_dataset('./Shoe Images', 210)

    rng_key = jax.random.PRNGKey(0)
    rng_key, rng_G, rng_D = jax.random.split(rng_key, 3)

    init_batch_G = jnp.ones((1, 1, 1, 64), dtype=jnp.float32)
    variables_G = Generator(training = True).init(rng_G, init_batch_G)

    init_batch_D = jnp.ones((1, 64, 64, 3), dtype=jnp.float32)
    variables_D = Discriminator(training = True).init(rng_D, init_batch_D)

    optimizer_G = Adam(learning_rate=1e-4, beta1=0.5, beta2=0.999).create(variables_G['params'])
    optimizer_D = Adam(learning_rate=1e-4, beta1=0.5, beta2=0.999).create(variables_D['params'])

    test_latent_dim = jax.random.normal(rng_key, shape=(1, 1, 1, 64))

    run = wandb.init(project='ShoeGAN')

    with tqdm(range(2000)) as progress_bar:
        for _ in progress_bar:
            losses_G, losses_D = [], []

            for batch in dataset:
                rng_key, variables_G, variables_D, optimizer_G, optimizer_D, G_loss, D_loss = train_step(
                        rng_key, variables_G, variables_D, optimizer_G, optimizer_D, batch
                )

                losses_G.append(G_loss)
                losses_D.append(D_loss)

            progress_bar.set_postfix(losses_G=jnp.mean(jnp.array(losses_G)), losses_D=jnp.mean(jnp.array(losses_D)))

            # Log Prediction to qualitatively see how the generator is doing
            prediction = Generator(training = False).apply({
                    'params' : optimizer_G.target,
                    'batch_stats' : variables_G['batch_stats'],
            }, test_latent_dim)

            # Convert prediction JAX Array to Numpy Array
            prediction = np.array(prediction)

            run.log({
                    'Generator Loss' : jnp.mean(jnp.array(losses_G)),
                    'Discriminator Loss' : jnp.mean(jnp.array(losses_D)),
                    'Generator Image' : wandb.Image(prediction[0, :, :, :])
            })



if __name__ == '__main__':
    main()
