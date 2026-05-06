
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    GRU,
    Dense,
    Flatten,
    RepeatVector,
    TimeDistributed
)

from tensorflow.keras.optimizers import Adam

from metrics.discriminative_metrics import discriminative_score_metrics


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def extract_time(data):
    time = list()
    max_seq_len = 0

    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i]))
        time.append(len(data[i]))

    return time, max_seq_len


def random_generator(batch_size, z_dim, seq_len):
    return np.random.uniform(0., 1., [batch_size, seq_len, z_dim]).astype(np.float32)


# ------------------------------------------------------------
# Model Components
# ------------------------------------------------------------

class Embedder(Model):

    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        self.rnn = tf.keras.Sequential([
            GRU(
                hidden_dim,
                return_sequences=True
            )
            for _ in range(num_layers)
        ])

        self.dense = Dense(hidden_dim)

    def call(self, x):

        x = self.rnn(x)

        return self.dense(x)


class Recovery(Model):

    def __init__(self, hidden_dim, num_layers, dim):
        super().__init__()

        self.rnn = tf.keras.Sequential([
            GRU(
                hidden_dim,
                return_sequences=True
            )
            for _ in range(num_layers)
        ])

        self.dense = Dense(dim)

    def call(self, x):

        x = self.rnn(x)

        return self.dense(x)


class Generator(Model):

    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        self.rnn = tf.keras.Sequential([
            GRU(
                hidden_dim,
                return_sequences=True
            )
            for _ in range(num_layers)
        ])

        self.dense = Dense(hidden_dim)

    def call(self, z):

        z = self.rnn(z)

        return self.dense(z)


class Supervisor(Model):

    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        self.rnn = tf.keras.Sequential([
            GRU(
                hidden_dim,
                return_sequences=True
            )
            for _ in range(num_layers - 1)
        ])

        self.dense = Dense(hidden_dim)

    def call(self, x):

        x = self.rnn(x)

        return self.dense(x)


class Discriminator(Model):

    def __init__(self, hidden_dim, num_layers):
        super().__init__()

        self.rnn = tf.keras.Sequential([
            GRU(
                hidden_dim,
                return_sequences=True
            )
            for _ in range(num_layers)
        ])

        self.flatten = Flatten()

        self.dense = Dense(1)

    def call(self, x):

        x = self.rnn(x)

        x = self.flatten(x)

        return self.dense(x)


# ------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------

def discriminator_loss(real_output, fake_output):

    real_loss = tf.reduce_mean(
        tf.square(real_output - 1)
    )

    fake_loss = tf.reduce_mean(
        tf.square(fake_output)
    )

    return real_loss + fake_loss



def generator_loss(fake_output):

    return tf.reduce_mean(
        tf.square(fake_output - 1)
    )



def supervised_loss(h, h_hat):

    return tf.reduce_mean(
        tf.square(h[:, 1:, :] - h_hat[:, :-1, :])
    )



def reconstruction_loss(x, x_tilde):

    return tf.reduce_mean(
        tf.square(x - x_tilde)
    )


# ------------------------------------------------------------
# Main SeriesGAN Function
# ------------------------------------------------------------


def seriesgan(ori_data, parameters, num_samples='same'):

    ori_data = np.asarray(ori_data).astype(np.float32)

    no, seq_len, dim = ori_data.shape

    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']

    z_dim = dim

    # ------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------

    min_val = np.min(np.min(ori_data, axis=0), axis=0)
    ori_data = ori_data - min_val

    max_val = np.max(np.max(ori_data, axis=0), axis=0)
    ori_data = ori_data / (max_val + 1e-7)

    # ------------------------------------------------------------
    # Models
    # ------------------------------------------------------------

    embedder = Embedder(hidden_dim, num_layers)

    recovery = Recovery(hidden_dim, num_layers, dim)

    generator = Generator(hidden_dim, num_layers)

    supervisor = Supervisor(hidden_dim, num_layers)

    discriminator = Discriminator(hidden_dim, num_layers)

    # ------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------

    e_optimizer = Adam()

    g_optimizer = Adam()

    d_optimizer = Adam()

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    print('Start Training')

    for epoch in range(iterations):

        # Shuffle data manually (avoids tf.data eager-mode requirement)
        idx = np.random.permutation(no)
        shuffled_data = ori_data[idx]

        for start in range(0, no, batch_size):

            X_mb = tf.convert_to_tensor(
                shuffled_data[start:start + batch_size],
                dtype=tf.float32
            )

            batch_current = X_mb.shape[0]

            Z_mb = random_generator(
                batch_current,
                z_dim,
                seq_len
            )

            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            # ------------------------------------------------------------
            # Train Embedder
            # ------------------------------------------------------------

            with tf.GradientTape() as tape:

                H = embedder(X_mb)

                X_tilde = recovery(H)

                e_loss = reconstruction_loss(X_mb, X_tilde)

            e_vars = (
                embedder.trainable_variables +
                recovery.trainable_variables
            )

            e_grads = tape.gradient(e_loss, e_vars)

            e_optimizer.apply_gradients(
                zip(e_grads, e_vars)
            )

            # ------------------------------------------------------------
            # Train Generator
            # ------------------------------------------------------------

            with tf.GradientTape() as tape:

                E_hat = generator(Z_mb)

                H_hat = supervisor(E_hat)

                X_hat = recovery(H_hat)

                Y_fake = discriminator(H_hat)

                g_loss_u = generator_loss(Y_fake)

                H_real = embedder(X_mb)

                h_hat_supervise = supervisor(H_real)

                g_loss_s = supervised_loss(
                    H_real,
                    h_hat_supervise
                )

                g_loss = g_loss_u + 100 * g_loss_s

            g_vars = (
                generator.trainable_variables +
                supervisor.trainable_variables
            )

            g_grads = tape.gradient(g_loss, g_vars)

            g_optimizer.apply_gradients(
                zip(g_grads, g_vars)
            )

            # ------------------------------------------------------------
            # Train Discriminator
            # ------------------------------------------------------------

            with tf.GradientTape() as tape:

                H_real = embedder(X_mb)

                E_hat = generator(Z_mb)

                H_hat = supervisor(E_hat)

                Y_real = discriminator(H_real)

                Y_fake = discriminator(H_hat)

                d_loss = discriminator_loss(
                    Y_real,
                    Y_fake
                )

            d_vars = discriminator.trainable_variables

            d_grads = tape.gradient(d_loss, d_vars)

            d_optimizer.apply_gradients(
                zip(d_grads, d_vars)
            )

        if epoch % 10 == 0:

            print(
                f'Epoch {epoch}/{iterations} | '
                f'E_loss: {float(e_loss):.4f} | '
                f'G_loss: {float(g_loss):.4f} | '
                f'D_loss: {float(d_loss):.4f}'
            )

    print('Finish Training')

    # ------------------------------------------------------------
    # Generate Synthetic Data
    # ------------------------------------------------------------

    if num_samples == 'same':
        num_samples = no

    Z_mb = random_generator(
        num_samples,
        z_dim,
        seq_len
    )

    Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

    E_hat = generator(Z_mb)

    H_hat = supervisor(E_hat)

    generated_data = recovery(H_hat)

    generated_data = tf.keras.backend.eval(generated_data)

    # Renormalization
    generated_data = generated_data * max_val
    generated_data = generated_data + min_val

    return generated_data


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == '__main__':

    dummy_data = np.random.rand(100, 24, 5).astype(np.float32)

    parameters = {
        'hidden_dim': 24,
        'num_layer': 3,
        'iterations': 20,   # smoke test — change to 1000 for full training
        'batch_size': 32
    }

    generated = seriesgan(
        dummy_data,
        parameters,
        100
    )

    print(generated.shape)

