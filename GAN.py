import os
from PIL import Image
from flax import training

import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.optim import Adam
from flax.training import train_state
import optax
from optax import adam

from typing import Any

from tqdm import tqdm
import wandb
from icecream import ic

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
        x = nn.Conv(features = 32, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 32, 32, 16) -> (n, 16, 16, 32)
        x = nn.leaky_relu(x)
        x = nn.Conv(features = 64, kernel_size = (4, 4), strides = (2, 2), padding = 'SAME')(x) # (n, 16, 16, 32) -> (n, 8, 8, 64)
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

class BatchNormTrainState(train_state.TrainState):
    batch_stats: Any

def initialize_train_state(model_type, input_shape, tx, rng_key, **kwargs):
    model = model_type(**kwargs)
    variables = model.init(rng_key, jnp.ones(input_shape))

    return BatchNormTrainState.create(
            apply_fn = model.apply,
            tx = tx,
            params = variables.get('params'),
            batch_stats = variables.get('batch_stats')
    )

@jax.vmap
def bce_logits_loss(logit, label):
    return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))

class GANTrainer:
    def __init__(self, rng_key):
        self.optimizer_G = adam(learning_rate=1e-4, b1=0.5, b2=0.999)
        self.optimizer_D = adam(learning_rate=1e-4, b1=0.5, b2=0.999)

        self.rng = rng_key

    def loss_generator(self, params_G, params_D, batch, variables_G, variables_D):
        self.rng, rng_G = jax.random.split(self.rng, 2)
        z = jax.random.normal(rng_G, shape=(batch.shape[0], 1, 1, 64))

        fake_batch, mutables = Generator(training = True).apply({ # TODO batch stats update
                'params' : params_G,
                'batch_stats' : variables_G.batch_stats
        }, z, mutable=['batch_stats'])

        fake_logits = Discriminator(training = True).apply({
                'params' : params_D,
        }, fake_batch)

        real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
        return jnp.mean(bce_logits_loss(fake_logits, real_labels)), mutables

    def loss_discriminator(self, params_D, params_G, batch, variables_G, variables_D):
        self.rng, rng_D = jax.random.split(self.rng, 2)
        z = jax.random.normal(rng_D, shape=(batch.shape[0], 1, 1, 64))

        fake_batch, _ = Generator(training = True).apply({ # TODO batch stats update
                'params' : params_G,
                'batch_stats' : variables_G.batch_stats,
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

        return jnp.mean(real_loss + fake_loss)

    def train_step(self, variables_G, variables_D, batch):
        (G_loss, mutables), grad_G = jax.value_and_grad(self.loss_generator, has_aux=True)(
                variables_G.params, variables_D.params, batch, variables_G, variables_D
        )

        variables_G = variables_G.apply_gradients(grads=grad_G, batch_stats=mutables['batch_stats'])

        D_loss, grad_D = jax.value_and_grad(self.loss_discriminator, has_aux=False)(
                variables_D.params, variables_G.params, batch, variables_G, variables_D
        )

        variables_D = variables_D.apply_gradients(grads=grad_D)

        return variables_G, variables_D, G_loss, D_loss

def make_dataset(folder_path, batch_size):
    images = []

    for file in tqdm(os.listdir(folder_path), desc = 'Loading Files'):
        if file.endswith(".jpg") or file.endswith('.png'):
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
    dataset = make_dataset('./Shoe Images', 420)

    print(f"Loaded Dataset of Shape : {len(dataset)}, {dataset[0].shape}")

    rng_key = jax.random.PRNGKey(42)
    rng_key, rng_G, rng_D = jax.random.split(rng_key, 3)

    trainer = GANTrainer(rng_key=rng_key)

    variables_G = initialize_train_state(Generator, (1, 1, 1, 64), trainer.optimizer_G, rng_G, training=True)
    variables_D = initialize_train_state(Discriminator, (1, 64, 64, 3), trainer.optimizer_D, rng_D, training=True)

    test_latent_dim = jax.random.normal(rng_key, shape=(1, 1, 1, 64))

    run = wandb.init(project='ShoeGAN')

    with tqdm(range(2000)) as progress_bar:
        for _ in progress_bar:
            losses_G, losses_D = [], []

            for batch in dataset:
                variables_G, variables_D, G_loss, D_loss = trainer.train_step(
                        variables_G, variables_D, batch
                )

                losses_G.append(G_loss)
                losses_D.append(D_loss)

            progress_bar.set_postfix(losses_G=jnp.mean(jnp.array(losses_G)), losses_D=jnp.mean(jnp.array(losses_D)))

            # Log Prediction to qualitatively see how the generator is doing
            prediction = Generator(training = False).apply({
                    'params' : variables_G.params,
                    'batch_stats' : variables_G.batch_stats,
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
