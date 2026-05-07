
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow.keras import Model
from tensorflow.keras.layers import GRU, Dense, Flatten
from tensorflow.keras.optimizers import Adam

from metrics.discriminative_metrics import discriminative_score_metrics
from drive_sync import pull_checkpoints, push_checkpoints


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------

def extract_time(data):
    time = []
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i]))
        time.append(len(data[i]))
    return time, max_seq_len


# ------------------------------------------------------------
# Model Components
# ------------------------------------------------------------

class Embedder(Model):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn   = tf.keras.Sequential([GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)])
        self.dense = Dense(hidden_dim)

    def call(self, x):
        return self.dense(self.rnn(x))


class Recovery(Model):
    def __init__(self, hidden_dim, num_layers, dim):
        super().__init__()
        self.rnn   = tf.keras.Sequential([GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)])
        self.dense = Dense(dim)

    def call(self, x):
        return self.dense(self.rnn(x))


class Generator(Model):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn   = tf.keras.Sequential([GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)])
        self.dense = Dense(hidden_dim)

    def call(self, z):
        return self.dense(self.rnn(z))


class Supervisor(Model):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn   = tf.keras.Sequential([GRU(hidden_dim, return_sequences=True) for _ in range(num_layers - 1)])
        self.dense = Dense(hidden_dim)

    def call(self, x):
        return self.dense(self.rnn(x))


class Discriminator(Model):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.rnn     = tf.keras.Sequential([GRU(hidden_dim, return_sequences=True) for _ in range(num_layers)])
        self.flatten = Flatten()
        self.dense   = Dense(1)

    def call(self, x):
        return self.dense(self.flatten(self.rnn(x)))


# ------------------------------------------------------------
# Loss Functions
# ------------------------------------------------------------

def discriminator_loss(real_output, fake_output):
    return tf.reduce_mean(tf.square(real_output - 1)) + tf.reduce_mean(tf.square(fake_output))

def generator_loss(fake_output):
    return tf.reduce_mean(tf.square(fake_output - 1))

def supervised_loss(h, h_hat):
    return tf.reduce_mean(tf.square(h[:, 1:, :] - h_hat[:, :-1, :]))

def reconstruction_loss(x, x_tilde):
    return tf.reduce_mean(tf.square(x - x_tilde))


# ------------------------------------------------------------
# Main SeriesGAN Function
# ------------------------------------------------------------

def seriesgan(ori_data, parameters, num_samples='same'):

    ori_data = np.asarray(ori_data, dtype=np.float32)
    no, seq_len, dim = ori_data.shape

    hidden_dim    = parameters['hidden_dim']
    num_layers    = parameters['num_layer']
    iterations    = parameters['iterations']
    batch_size    = parameters['batch_size']

    checkpoint_dir   = parameters.get('checkpoint_dir', './seriesgan_checkpoints')
    checkpoint_every = parameters.get('checkpoint_every', 50)
    drive_folder_id  = parameters.get('drive_folder_id', None)
    gdrive_secret    = parameters.get('gdrive_secret', 'GDRIVE_SERVICE_ACCOUNT')

    os.makedirs(checkpoint_dir, exist_ok=True)

    z_dim = dim

    # --------------------------------------------------------
    # Normalization (persisted to survive restarts)
    # --------------------------------------------------------

    norm_path = os.path.join(checkpoint_dir, 'norm_stats.npz')

    if os.path.exists(norm_path):
        norm    = np.load(norm_path)
        min_val = norm['min_val']
        max_val = norm['max_val']
        print(f'[Checkpoint] Loaded normalization stats from {norm_path}')
    else:
        min_val = np.min(np.min(ori_data, axis=0), axis=0)
        max_val = np.max(np.max(ori_data - min_val, axis=0), axis=0)
        np.savez(norm_path, min_val=min_val, max_val=max_val)

    ori_data = (ori_data - min_val) / (max_val + 1e-7)

    # --------------------------------------------------------
    # Models & Optimizers
    # --------------------------------------------------------

    embedder      = Embedder(hidden_dim, num_layers)
    recovery      = Recovery(hidden_dim, num_layers, dim)
    generator     = Generator(hidden_dim, num_layers)
    supervisor    = Supervisor(hidden_dim, num_layers)
    discriminator = Discriminator(hidden_dim, num_layers)

    e_optimizer = Adam()
    g_optimizer = Adam()
    d_optimizer = Adam()

    # Warm-up: build all weights before checkpoint restore
    _d = tf.zeros([1, seq_len, dim])
    _z = tf.zeros([1, seq_len, z_dim])
    _h = embedder(_d);  recovery(_h)
    _e = generator(_z); _hs = supervisor(_e); discriminator(_hs)

    # Pre-compute variable lists (weights are built after warm-up)
    e_vars = embedder.trainable_variables + recovery.trainable_variables
    g_vars = generator.trainable_variables + supervisor.trainable_variables
    d_vars = discriminator.trainable_variables

    # --------------------------------------------------------
    # Checkpoint Setup
    # --------------------------------------------------------

    ckpt = tf.train.Checkpoint(
        embedder=embedder, recovery=recovery,
        generator=generator, supervisor=supervisor,
        discriminator=discriminator,
        e_optimizer=e_optimizer,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
    )

    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory=checkpoint_dir,
        max_to_keep=20, checkpoint_name='seriesgan_ckpt'
    )

    # Pull latest checkpoint from Google Drive (Kaggle only; no-op on Colab)
    pull_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        try:
            step_num    = int(ckpt_manager.latest_checkpoint.split('-')[-1])
            start_epoch = step_num * checkpoint_every
        except ValueError:
            start_epoch = 0
        print(f'[Checkpoint] Resumed from epoch {start_epoch}  ({ckpt_manager.latest_checkpoint})')
    else:
        print('[Checkpoint] No checkpoint found — starting fresh.')

    # --------------------------------------------------------
    # tf.data pipeline  (FIX 1: eliminates Python batch loop)
    # --------------------------------------------------------

    dataset = (
        tf.data.Dataset.from_tensor_slices(ori_data)
        .cache()
        .shuffle(buffer_size=no, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # --------------------------------------------------------
    # Compiled training step  (FIX 2: @tf.function = TF graph)
    # FIX 3: single forward pass per component per step
    # --------------------------------------------------------

    @tf.function
    def train_step(X_mb):
        Z_mb = tf.random.uniform(tf.shape(X_mb))   # noise, same shape as X

        # ── Embedder ──────────────────────────────────────
        with tf.GradientTape() as tape:
            H       = embedder(X_mb)
            X_tilde = recovery(H)
            e_loss  = reconstruction_loss(X_mb, X_tilde)
        e_optimizer.apply_gradients(zip(tape.gradient(e_loss, e_vars), e_vars))

        # ── Generator ─────────────────────────────────────
        with tf.GradientTape() as tape:
            E_hat    = generator(Z_mb)
            H_hat    = supervisor(E_hat)
            Y_fake   = discriminator(H_hat)
            g_loss_u = generator_loss(Y_fake)
            H_real   = embedder(X_mb)
            h_sup    = supervisor(H_real)
            g_loss_s = supervised_loss(H_real, h_sup)
            g_loss   = g_loss_u + 100.0 * g_loss_s
        g_optimizer.apply_gradients(zip(tape.gradient(g_loss, g_vars), g_vars))

        # ── Discriminator ─────────────────────────────────
        with tf.GradientTape() as tape:
            H_real  = embedder(X_mb)
            H_hat   = supervisor(generator(Z_mb))
            Y_real  = discriminator(H_real)
            Y_fake  = discriminator(H_hat)
            d_loss  = discriminator_loss(Y_real, Y_fake)
        d_optimizer.apply_gradients(zip(tape.gradient(d_loss, d_vars), d_vars))

        return e_loss, g_loss, d_loss

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------

    if start_epoch >= iterations:
        print(f'[Checkpoint] Already complete ({start_epoch}/{iterations}). Skipping to generation.')
    else:
        print(f'Start Training  (epochs {start_epoch} → {iterations})')

        for epoch in range(start_epoch, iterations):

            e_metric = tf.keras.metrics.Mean()
            g_metric = tf.keras.metrics.Mean()
            d_metric = tf.keras.metrics.Mean()

            for X_mb in dataset:
                e_loss, g_loss, d_loss = train_step(X_mb)
                e_metric.update_state(e_loss)
                g_metric.update_state(g_loss)
                d_metric.update_state(d_loss)

            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch:05d}/{iterations} | '
                    f'E_loss: {e_metric.result():.4f} | '
                    f'G_loss: {g_metric.result():.4f} | '
                    f'D_loss: {d_metric.result():.4f}'
                )

            if (epoch + 1) % checkpoint_every == 0:
                save_path = ckpt_manager.save()
                print(f'[Checkpoint] Saved at epoch {epoch + 1}  →  {save_path}')
                push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

        # Final save
        ckpt_manager.save()
        push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)
        print('Finish Training')

    # --------------------------------------------------------
    # Generate Synthetic Data
    # --------------------------------------------------------

    if num_samples == 'same':
        num_samples = no

    Z_mb           = tf.random.uniform([num_samples, seq_len, z_dim])
    E_hat          = generator(Z_mb)
    H_hat          = supervisor(E_hat)
    generated_data = recovery(H_hat).numpy()

    # Renormalize
    generated_data = generated_data * max_val + min_val

    return generated_data


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------

if __name__ == '__main__':

    dummy_data = np.random.rand(100, 24, 5).astype(np.float32)

    parameters = {
        'hidden_dim':      24,
        'num_layer':       3,
        'iterations':      20,
        'batch_size':      32,
        'checkpoint_dir':  './seriesgan_checkpoints',
        'checkpoint_every': 5,
        'drive_folder_id': None,
        'gdrive_secret':   'GDRIVE_SERVICE_ACCOUNT',
    }

    generated = seriesgan(dummy_data, parameters, 100)
    print(generated.shape)
