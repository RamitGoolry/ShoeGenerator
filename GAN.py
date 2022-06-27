import os
from PIL import Image

import numpy as np

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.optim import Adam
import optax
from optax import adam

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

@jax.vmap
def bce_logits_loss(logit, label):
    return jnp.maximum(logit, 0) - logit * label + jnp.log(1 + jnp.exp(-jnp.abs(logit)))

class GANTrainer:
    def __init__(self, variables_G, variables_D, rng_key):
        self.optimizer_G = adam(learning_rate=1e-4, b1=0.5, b2=0.999)
        self.optimizer_D = adam(learning_rate=1e-4, b1=0.5, b2=0.999)

        self.opt_G_state = self.optimizer_G.init(variables_G['params'])
        self.opt_D_state = self.optimizer_D.init(variables_D['params'])

        self.rng = rng_key

    def loss_generator(self, params_G, params_D, batch, variables_G, variables_D):
        self.rng, rng_G = jax.random.split(self.rng, 2)
        z = jax.random.normal(rng_G, shape=(batch.shape[0], 1, 1, 64))

        fake_batch, variables_G_batch = Generator(training = True).apply({ # TODO batch stats update
                'params' : params_G,
                'batch_stats' : variables_G['batch_stats'],
        }, z, mutable=['batch_stats'])

        fake_logits = Discriminator(training = True).apply({
                'params' : params_D,
        }, fake_batch)

        real_labels = jnp.ones((batch.shape[0],), dtype=jnp.int32)
        return jnp.mean(bce_logits_loss(fake_logits, real_labels)), (variables_G, variables_D)

    def loss_discriminator(self, params_D, params_G, batch, variables_G, variables_D):
        self.rng, rng_D = jax.random.split(self.rng, 2)
        z = jax.random.normal(rng_D, shape=(batch.shape[0], 1, 1, 64))

        fake_batch, variables_G_batch = Generator(training = True).apply({ # TODO batch stats update
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

    def train_step(self, variables_G, variables_D, batch):
        (G_loss, (variables_G, variables_D)), grad_G = jax.value_and_grad(self.loss_generator, has_aux=True)(
                variables_G['params'], variables_D['params'], batch, variables_G, variables_D
        )

        # Grad_G is a FrozenDict
        # opt_G_state is a tuple
        ic(type(grad_G))
        ic(type(self.opt_G_state[0]))
        ic(type(self.opt_G_state[1]))

        ic(jax.tree_map(lambda x: x.shape, variables_G['params'])) # Display shapes of all outputs
        ic(jax.tree_map(lambda x: x.shape, self.opt_G_state[0]))

        try:
            self.opt_G_state, updates = self.optimizer_G.update(grad_G, self.opt_G_state)
            variables_G['params'] = optax.apply_updates(variables_G['params'], updates)
        except Exception as e:
            print(str(e)[:1000])
            exit(1)

        (D_loss, (variables_G, variables_D)), grad_D = jax.value_and_grad(self.loss_discriminator, has_aux=True)(
                variables_D['params'], variables_D['params'], batch, variables_G, variables_D
        )
        self.opt_D_state, updates = self.optimizer_D.update(grad_D, self.opt_D_state)
        variables_D['params'] = optax.apply_updates(variables_D['params'], updates)

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

    init_batch_G = jnp.ones((1, 1, 1, 64), dtype=jnp.float32)
    variables_G = Generator(training = True).init(rng_G, init_batch_G)

    init_batch_D = jnp.ones((1, 64, 64, 3), dtype=jnp.float32)
    variables_D = Discriminator(training = True).init(rng_D, init_batch_D)

    variables_G = variables_G.unfreeze()
    variables_D = variables_D.unfreeze()

    trainer = GANTrainer(variables_G, variables_D, rng_key=rng_key)

    test_latent_dim = jax.random.normal(rng_key, shape=(1, 1, 1, 64))

    run = wandb.init(project='ShoeGAN', mode='disabled')

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
                    'params' : trainer.optimizer_G.target,
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
