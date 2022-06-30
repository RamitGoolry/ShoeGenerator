import os
from PIL import Image

import numpy as np

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.optim import Adam, RMSProp
from tqdm import tqdm
import time
from icecream import ic

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

@jax.vmap
def bce_logits_loss(logit, label):
    return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))

class GANTrainer:
    def __init__(self, variables_G, variables_D, rng_key = None, wandb_run = None):
        self.optimizer_G = RMSProp(learning_rate=1e-4).create(variables_G['params'])
        self.optimizer_D = RMSProp(learning_rate=1e-4).create(variables_D['params'])

        self.rng = rng_key if rng_key is not None else jax.random.PRNGKey(int(time.time()))

        if wandb_run:
            self.wandb_run = wandb_run

    def loss_generator(self, params_G, params_D, batch, variables_G, variables_D):
        G_rng, self.rng = jax.random.split(self.rng, 2)
        z = jax.random.normal(G_rng, shape=(batch.shape[0], 1, 1, 64))

        fake_batch, variables_G = Generator(training = True).apply({
                'params' : params_G,
                'batch_stats' : variables_G['batch_stats'],
        }, z, mutable=['batch_stats'])

        fake_logits = Discriminator(training = True).apply({
                'params' : params_D,
        }, fake_batch)

        return jnp.mean(-fake_logits), (variables_G, variables_D)

    def loss_discriminator(self, params_D, params_G, batch, variables_G, variables_D):
        D_rng, self.rng = jax.random.split(self.rng, 2)
        z = jax.random.normal(D_rng, shape=(batch.shape[0], 1, 1, 64))

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

        return jnp.mean(real_logits) - jnp.mean(fake_logits), (variables_G, variables_D)

    def _clip_tree(self, tree, clip):
        return jax.tree_util.tree_map(lambda x: jnp.clip(x, -clip, clip), tree)

    def train_step(self, variables_G, variables_D, batch, n_critic, clip):
        avg_D_loss = 0

        # Train the discriminator
        for _ in range(n_critic):
            (D_loss, (variables_G, variables_D)), grad_D = jax.value_and_grad(self.loss_discriminator, has_aux=True)(
                    self.optimizer_D.target, self.optimizer_G.target, batch, variables_G, variables_D
            )
            avg_D_loss += D_loss
            self.optimizer_D = self.optimizer_D.apply_gradient(grad_D)

            # Clip the discriminator weights
            variables_D['params'] = self._clip_tree(variables_D['params'], clip)

        avg_D_loss /= n_critic

        (G_loss, (variables_G, variables_D)), grad_G = jax.value_and_grad(self.loss_generator, has_aux=True)(
                self.optimizer_G.target, self.optimizer_D.target, batch, variables_G, variables_D)
        self.optimizer_G = self.optimizer_G.apply_gradient(grad_G)

        if self.wandb_run:
            histograms = {}

            for layer, param in self.optimizer_D.target.items():
                if type(param) == dict or type(param) == flax.core.frozen_dict.FrozenDict:
                    for param_name, param_value in param.items():
                        histograms[f'Discriminator/{layer}:{param_name}'] = wandb.Histogram(np.array(param_value))
                elif type(param) == jnp.DeviceArray:
                    histograms[f'Discriminator/{layer}'] = np.array(param)
                else:
                    print("ERROR : Unknown type for Discriminator parameter:", type(param))

            for layer, param in self.optimizer_G.target.items():
                if type(param) == dict or type(param) == flax.core.frozen_dict.FrozenDict:
                    for param_name, param_value in param.items():
                        histograms[f'Generator/{layer}:{param_name}'] = wandb.Histogram(np.array(param_value))
                elif type(param) == jnp.DeviceArray:
                    histograms[f'Generator/{layer}'] = np.array(param)
                else:
                    print("ERROR : Unknown type for Generator parameter:", type(param))

            self.wandb_run.log(histograms, commit=False)

        return variables_G, variables_D, G_loss, avg_D_loss

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

    init_batch_G = jnp.ones((1, 1, 1, 64), dtype=jnp.float32)
    variables_G = Generator(training = True).init(rng_G, init_batch_G)

    init_batch_D = jnp.ones((1, 64, 64, 3), dtype=jnp.float32)
    variables_D = Discriminator(training = True).init(rng_D, init_batch_D)

    variables_G = variables_G.unfreeze()
    variables_D = variables_D.unfreeze()

    run = wandb.init(project='ShoeGAN')

    trainer = GANTrainer(variables_G, variables_D, wandb_run = run)

    test_latent_dim = jax.random.normal(rng_key, shape=(1, 1, 1, 64))

    with tqdm(range(100)) as progress_bar:
        for epoch in progress_bar:
            losses_G, losses_D = [], []

            for batch in dataset:
                variables_G, variables_D, G_loss, D_loss = trainer.train_step(
                        variables_G, variables_D, batch, n_critic = 1, clip = 0.001
                )

                losses_G.append(G_loss)
                losses_D.append(D_loss)

            progress_bar.set_postfix(losses_G=jnp.mean(jnp.array(losses_G)), losses_D=jnp.mean(jnp.array(losses_D)))

            wandb_dict = {
                    'Generator Loss' : jnp.mean(jnp.array(losses_G)),
                    'Discriminator Loss' : jnp.mean(jnp.array(losses_D))
            }

            if epoch % 10 == 0:
                # Log Prediction to qualitatively see how the generator is doing
                prediction = Generator(training = False).apply({
                        'params' : trainer.optimizer_G.target,
                        'batch_stats' : variables_G['batch_stats'],
                }, test_latent_dim)

                # Convert prediction JAX Array to Numpy Array
                prediction = np.array(prediction)

                wandb_dict.update({
                    'Prediction' : wandb.Image(prediction[0, : , :, :])
                })

            run.log(wandb_dict)

if __name__ == '__main__':
    main()
