from PIL import Image
import os

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Conv2D
from tensorflow.keras.layers import Reshape, Flatten, Input, Activation, Conv2DTranspose

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import re
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2

from icecream import ic

class DCGAN(keras.Model):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        
        self.gan = Sequential()
        
        self.gan.add(self.generator)
        self.gan.add(self.discriminator)
        
    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean()
        self.g_loss_metric = tf.keras.metrics.Mean()

    def build_generator(self) -> Model:        
        model = Sequential(name='generator')
        
        model.add(Dense(
            units = 4 * 4 * 512,
            kernel_initializer = 'glorot_uniform',
            input_shape = (1, 1, 100))
        )
        model.add(Reshape(target_shape=(4, 4, 512)))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))
        
        model.add(Conv2DTranspose(
            filters = 256, kernel_size = (5,5),
            strides = (2, 2), padding = 'same',
            data_format = 'channels_last',
            kernel_initializer='glorot_uniform'
        ))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))
        
        model.add(Conv2DTranspose(
            filters = 128, kernel_size = (5,5),
            strides = (2, 2), padding = 'same',
            data_format = 'channels_last',
            kernel_initializer='glorot_uniform'
        ))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))
        
        model.add(Conv2DTranspose(
            filters = 64, kernel_size = (5,5),
            strides = (2, 2), padding = 'same',
            data_format = 'channels_last',
            kernel_initializer='glorot_uniform'
        ))
        model.add(BatchNormalization(momentum=0.5))
        model.add(Activation('relu'))
        
        model.add(Conv2DTranspose(
            filters = 3, kernel_size = (5,5),
            strides = (2, 2), padding = 'same',
            data_format = 'channels_last',
            kernel_initializer='glorot_uniform'
        ))
        model.add(Activation('tanh'))
        
        return model
    
    def build_discriminator(self) -> Model:
        img_shape = (self.img_rows, self.img_cols, self.channels)
        
        model = Sequential(name='discriminator')
        
        model.add(Conv2D(filters=64, kernel_size=(5, 5),
                        strides = (2, 2), padding = 'same',
                        data_format = 'channels_last',
                        kernel_initializer = 'glorot_uniform',
                        input_shape = img_shape))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2D(filters=128, kernel_size=(5, 5),
                        strides = (2, 2), padding = 'same',
                        data_format = 'channels_last',
                        kernel_initializer = 'glorot_uniform'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2D(filters=256, kernel_size=(5, 5),
                        strides = (2, 2), padding = 'same',
                        data_format = 'channels_last',
                        kernel_initializer = 'glorot_uniform'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        
        model.add(Conv2D(filters=512, kernel_size=(5, 5),
                        strides = (2, 2), padding = 'same',
                        data_format = 'channels_last',
                        kernel_initializer = 'glorot_uniform'))
        model.add(BatchNormalization(momentum=0.5))
        model.add(LeakyReLU(0.2))
        
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        return model

    def summary(self):
        self.generator.summary()
        self.discriminator.summary()

    def train_step(self, real_images):
        batch_size = real_images.shape[0]

        # Sample noise as generator input
        random_latent_noise = tf.random.normal(shape=(batch_size, 1, 1, 100))

        # Generate a batch of images
        generated_images = self.generator(random_latent_noise)

        # Combine with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Labels for generated and real images
        labels = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

        # Adding random noise to the labels
        labels += 0.05 * tf.random.uniform(labels.shape)

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        
        grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))

        # Sample random points in the latent space
        random_latent_noise = tf.random.normal(shape=(batch_size, 1, 1, 100))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_noise))
            g_loss = self.loss_fn(misleading_labels, predictions)
        
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {
            'd_loss': self.d_loss_metric.result(),
            'g_loss': self.g_loss_metric.result(),
        }

def train(model, directory, epochs, batch_size):
    images = []
    
    for img_name in os.listdir(directory):
        if 'png' in img_name:
            images.append(np.array(Image.open(os.path.join(directory, img_name))))
    
    images = np.array([img / 255.0 for img in images])
    images = np.array([cv2.resize(img, (64, 64)) for img in images])

    model.compile(
        d_optimizer = Adam(learning_rate=0.0002, beta_1=0.5),
        g_optimizer = Adam(learning_rate=0.0002, beta_1=0.5),
        loss_fn = tf.keras.losses.BinaryCrossentropy()
    )

    model.fit(images, epochs=epochs, batch_size=batch_size)

                

def main():
    gan = DCGAN()

    train(gan, './Shoe Images', epochs=10, batch_size=20)

if __name__ == '__main__':
    main()