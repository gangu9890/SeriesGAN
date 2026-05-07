"""
SeriesGAN — faithful implementation of arXiv:2410.21203
with checkpoint saving and Google Drive sync.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from utils import extract_time, random_generator, batch_generator
from metrics.discriminative_metrics import discriminative_score_metrics
from drive_sync import pull_checkpoints, push_checkpoints


def seriesgan(ori_data, parameters, num_samples):
    """SeriesGAN function.
    Args:
      - ori_data: original time-series data
      - parameters: network parameters
      - num_samples: number of synthetic samples to generate ('same' or int)
    Returns:
      - generated_data: generated time-series data
    """
    tf.compat.v1.reset_default_graph()

    # Basic Parameters
    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    # Normalization
    min_val = np.min(np.min(ori_data, axis=0), axis=0)
    ori_data = ori_data - min_val
    max_val = np.max(np.max(ori_data, axis=0), axis=0)
    ori_data = ori_data / (max_val + 1e-7)

    # Network Parameters
    if parameters['hidden_dim'] == 'same':
        hidden_dim = dim
    else:
        hidden_dim = parameters['hidden_dim']

    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    z_dim = dim
    gamma = 1
    beta = 1
    temporal_dimension = 16

    # Checkpoint parameters
    checkpoint_dir   = parameters.get('checkpoint_dir', './seriesgan_checkpoints')
    checkpoint_every = parameters.get('checkpoint_every', 500)
    drive_folder_id  = parameters.get('drive_folder_id', None)
    gdrive_secret    = parameters.get('gdrive_secret', 'GDRIVE_SERVICE_ACCOUNT')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save normalization stats
    norm_path = os.path.join(checkpoint_dir, 'norm_stats.npz')
    np.savez(norm_path, min_val=min_val, max_val=max_val)

    # Placeholders
    X = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, dim], name="myinput_x")
    Z = tf.compat.v1.placeholder(tf.float32, [None, max_seq_len, z_dim], name="myinput_z")
    T = tf.compat.v1.placeholder(tf.int32, [None], name="myinput_t")

    # ================================================================
    # Network Definitions
    # ================================================================

    def temporal_embedder(X, T):
        with tf.compat.v1.variable_scope("temporal_embedder", reuse=tf.compat.v1.AUTO_REUSE):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            e_outputs, e_last_states = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
            H = tf.compat.v1.layers.dense(e_last_states[-1], temporal_dimension, activation=None)
        return H

    def temporal_recovery(H_t, T):
        with tf.compat.v1.variable_scope("temporal_recovery", reuse=tf.compat.v1.AUTO_REUSE):
            expanded_H = tf.compat.v1.layers.dense(H_t, max_seq_len * dim, activation=None)
            expanded_H = tf.reshape(expanded_H, [-1, max_seq_len, dim])
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            r_outputs, _ = tf.compat.v1.nn.dynamic_rnn(r_cell, expanded_H, dtype=tf.float32)
            X_tilde = tf.compat.v1.layers.dense(r_outputs, dim, activation=None)
        return X_tilde

    def embedder(X, T):
        with tf.compat.v1.variable_scope("embedder", reuse=tf.compat.v1.AUTO_REUSE):
            e_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            e_outputs, _ = tf.compat.v1.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length=T)
            H = tf.compat.v1.layers.dense(e_outputs, hidden_dim, activation=None)
        return H

    def recovery(H, T):
        with tf.compat.v1.variable_scope("recovery", reuse=tf.compat.v1.AUTO_REUSE):
            r_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            r_outputs, _ = tf.compat.v1.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length=T)
            X_tilde = tf.compat.v1.layers.dense(r_outputs, hidden_dim, activation=None)
        return X_tilde

    def generator(Z, T):
        with tf.compat.v1.variable_scope("generator", reuse=tf.compat.v1.AUTO_REUSE):
            g_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            g_outputs, _ = tf.compat.v1.nn.dynamic_rnn(g_cell, Z, dtype=tf.float32, sequence_length=T)
            E = tf.compat.v1.layers.dense(g_outputs, hidden_dim, activation=None)
        return E

    def supervisor(H, T):
        with tf.compat.v1.variable_scope("supervisor", reuse=tf.compat.v1.AUTO_REUSE):
            s_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers - 1)])
            s_outputs, _ = tf.compat.v1.nn.dynamic_rnn(s_cell, H, dtype=tf.float32, sequence_length=T)
            S = tf.compat.v1.layers.dense(s_outputs, hidden_dim, activation=None)
        return S

    def discriminator(H, T):
        with tf.compat.v1.variable_scope("discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            d_outputs, _ = tf.compat.v1.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length=T)
            Y_hat = tf.compat.v1.layers.dense(d_outputs, hidden_dim, activation=None)
        return Y_hat

    def ae_discriminator(X, T):
        with tf.compat.v1.variable_scope("ae_discriminator", reuse=tf.compat.v1.AUTO_REUSE):
            d_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
                [tf.compat.v1.nn.rnn_cell.GRUCell(hidden_dim, activation=tf.nn.tanh) for _ in range(num_layers)])
            d_outputs, _ = tf.compat.v1.nn.dynamic_rnn(d_cell, X, dtype=tf.float32, sequence_length=T)
            flattened = tf.keras.layers.Flatten()(d_outputs)
            Y_hat_ae = tf.compat.v1.layers.dense(flattened, hidden_dim, activation=None)
        return Y_hat_ae

    # ================================================================
    # Build Computation Graph
    # ================================================================

    # Embedder & Recovery
    H = embedder(X, T)
    X_tilde = recovery(H, T)
    Y_ae_fake = ae_discriminator(X_tilde, T)
    Y_ae_real = ae_discriminator(X, T)

    # Generator
    E_hat = generator(Z, T)
    H_hat = supervisor(E_hat, T)
    H_hat_supervise = supervisor(H, T)

    # Synthetic data
    X_hat = recovery(H_hat, T)
    Y_ae_fake_e = ae_discriminator(X_hat, T)
    X_tilde_fake_second = recovery(E_hat, T)
    Y_ae_fake_e_second = ae_discriminator(X_tilde_fake_second, T)

    # Discriminator
    Y_fake = discriminator(H_hat, T)
    Y_real = discriminator(H, T)
    Y_fake_e = discriminator(E_hat, T)

    # Loss function autoencoder
    H_t = temporal_embedder(X, T)
    X_t = temporal_recovery(H_t, T)
    H_t_hat = temporal_embedder(X_hat, T)

    # ================================================================
    # Variables
    # ================================================================

    e_t_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('temporal_embedder')]
    r_t_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('temporal_recovery')]
    e_vars   = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('embedder')]
    r_vars   = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('recovery')]
    d_ae_vars = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('ae_discriminator')]
    g_vars   = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('generator')]
    s_vars   = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('supervisor')]
    d_vars   = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith('discriminator')]

    # ================================================================
    # Loss Functions (LSGAN — least squares)
    # ================================================================

    # Latent Discriminator loss
    D_loss_real   = tf.reduce_mean(tf.math.squared_difference(Y_real, tf.ones_like(Y_real)))
    D_loss_fake   = tf.reduce_mean(tf.square(Y_fake))
    D_loss_fake_e = tf.reduce_mean(tf.square(Y_fake_e))
    D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

    # Feature (AE) Discriminator loss
    D_ae_loss_real          = tf.reduce_mean(tf.math.squared_difference(Y_ae_real, tf.ones_like(Y_ae_real)))
    D_ae_loss_fake          = tf.reduce_mean(tf.square(Y_ae_fake))
    D_ae_loss_fake_e        = tf.reduce_mean(tf.square(Y_ae_fake_e))
    D_ae_loss_fake_e_second = tf.reduce_mean(tf.square(Y_ae_fake_e_second))
    D_ae_loss = D_ae_loss_real + D_ae_loss_fake
    D_ae_loss_real_second = tf.reduce_mean(tf.math.squared_difference(Y_ae_fake, tf.ones_like(Y_ae_fake)))
    D_ae_loss_second = D_ae_loss_real + D_ae_loss_real_second + beta * (D_ae_loss_fake_e + gamma * D_ae_loss_fake_e_second)

    # Generator loss
    G_loss_U      = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_fake), Y_fake))
    G_loss_U_e    = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_fake_e), Y_fake_e))
    G_loss_U_ae   = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e))
    G_loss_U_ae_e = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second))
    G_loss_U_totall = G_loss_U + G_loss_U_e + G_loss_U_ae + G_loss_U_ae_e

    # Supervised loss (2-step-ahead as per paper)
    G_loss_S = tf.reduce_mean(tf.math.squared_difference(H[:, 2:, :], H_hat_supervise[:, :-2, :]))

    # Time Series Characteristics loss (L_TS)
    mean_H_t     = tf.reduce_mean(H_t, axis=0)
    mean_H_t_hat = tf.reduce_mean(H_t_hat, axis=0)
    mse_mean     = tf.reduce_mean(tf.square(mean_H_t - mean_H_t_hat))
    std_H_t      = tf.math.reduce_std(H_t, axis=0)
    std_H_t_hat  = tf.math.reduce_std(H_t_hat, axis=0)
    mse_std      = tf.reduce_mean(tf.square(std_H_t - std_H_t_hat))
    G_loss_ts    = mse_mean + mse_std

    # Moment matching loss (L_V)
    G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)))
    G_loss_V2 = tf.reduce_mean(tf.abs(tf.nn.moments(X_hat, [0])[0] - tf.nn.moments(X, [0])[0]))
    G_loss_V  = G_loss_V1 + G_loss_V2

    # Combined Generator loss (Eq. 3 from paper)
    G_loss = (G_loss_U + gamma * G_loss_U_e
              + beta * (G_loss_U_ae + gamma * G_loss_U_ae_e)
              + 20 * tf.sqrt(G_loss_S) + 10 * G_loss_V + 20 * G_loss_ts)

    # Embedder loss
    lambda_c = 0.001
    E_loss_T00 = tf.compat.v1.losses.mean_squared_error(X, X_tilde)
    E_loss_U   = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake), Y_ae_fake))
    E_loss_U_e = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e), Y_ae_fake_e))
    E_loss_U_e_second = tf.reduce_mean(tf.math.squared_difference(tf.ones_like(Y_ae_fake_e_second), Y_ae_fake_e_second))

    E_loss_T0 = E_loss_T00 + lambda_c * E_loss_U
    E_loss_T0_second = E_loss_T00 + 0.1 * (lambda_c * E_loss_U + lambda_c * beta * 0.1 * (E_loss_U_e + gamma * E_loss_U_e_second))
    E_loss0 = tf.sqrt(E_loss_T0)
    E_loss  = tf.sqrt(E_loss_T0_second) + 0.01 * G_loss_S

    # Temporal AE loss
    E_loss_temporal = tf.compat.v1.losses.mean_squared_error(X, X_t)

    # ================================================================
    # Optimizers
    # ================================================================

    E_solver_temporal = tf.compat.v1.train.AdamOptimizer().minimize(E_loss_temporal, var_list=e_t_vars + r_t_vars)
    E0_solver         = tf.compat.v1.train.AdamOptimizer().minimize(E_loss0, var_list=e_vars + r_vars)
    E_solver          = tf.compat.v1.train.AdamOptimizer().minimize(E_loss, var_list=e_vars + r_vars)
    D_ae_solver       = tf.compat.v1.train.AdamOptimizer().minimize(D_ae_loss, var_list=d_ae_vars)
    D_ae_solver_second = tf.compat.v1.train.AdamOptimizer().minimize(D_ae_loss_second, var_list=d_ae_vars)
    D_solver          = tf.compat.v1.train.AdamOptimizer().minimize(D_loss, var_list=d_vars)
    G_solver          = tf.compat.v1.train.AdamOptimizer().minimize(G_loss, var_list=g_vars + s_vars)
    GS_solver         = tf.compat.v1.train.AdamOptimizer().minimize(G_loss_S, var_list=g_vars + s_vars)

    # ================================================================
    # Checkpoint Setup (tf.compat.v1.train.Saver)
    # ================================================================

    saver = tf.compat.v1.train.Saver(max_to_keep=20)
    ckpt_prefix = os.path.join(checkpoint_dir, 'seriesgan_ckpt')

    # Pull any existing checkpoints from Google Drive (Kaggle only)
    pull_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    # ================================================================
    # Session
    # ================================================================

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Try to restore from checkpoint
    latest_ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_ckpt:
        saver.restore(sess, latest_ckpt)
        print(f'[Checkpoint] Restored from {latest_ckpt}')

    final_generated = []
    global_summing = 5

    # ================================================================
    # Phase 1: Loss Function Autoencoder Training (0.5× iterations)
    # ================================================================

    phase1_iters = int(iterations * 0.5)
    print(f'Phase 1: Autoencoder Training for Loss ({phase1_iters} iters)')

    for itt in range(phase1_iters):
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            _, step_e_loss = sess.run([E_solver_temporal, E_loss_temporal],
                                      feed_dict={X: X_mb, T: T_mb})

        if itt % 500 == 0 or itt == phase1_iters - 1:
            print(f'  step: {itt*2}/{iterations}, AE_loss: {np.round(step_e_loss, 4)}')

    print('Phase 1 Complete')

    # ================================================================
    # Phase 2: Embedding Network Training (0.5× iterations)
    # ================================================================

    phase2_iters = int(iterations * 0.5)
    print(f'Phase 2: Embedding Network Training ({phase2_iters} iters)')

    step_d_ae_loss = 0.0
    for itt in range(phase2_iters):
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            _, step_e_loss = sess.run([E0_solver, E_loss0],
                                      feed_dict={X: X_mb, T: T_mb})

        check_d_ae_loss = sess.run(D_ae_loss, feed_dict={X: X_mb, T: T_mb})
        if check_d_ae_loss > 0.15:
            _, step_d_ae_loss = sess.run([D_ae_solver, D_ae_loss],
                                          feed_dict={X: X_mb, T: T_mb})

        if itt % 500 == 0 or itt == phase2_iters - 1:
            print(f'  step: {itt*2}/{iterations}, AE_loss: {np.round(step_e_loss, 4)}'
                  f', AE_D_loss: {np.round(step_d_ae_loss, 4)}')

    print('Phase 2 Complete')

    # ================================================================
    # Phase 3: Supervised Loss Only Training (1× iterations)
    # ================================================================

    print(f'Phase 3: Supervised Loss Only ({iterations} iters)')

    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        _, step_g_loss_s = sess.run([GS_solver, G_loss_S],
                                     feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

        if itt % 1000 == 0 or itt == iterations - 1:
            print(f'  step: {itt}/{iterations}, S_loss: {np.round(step_g_loss_s, 4)}')

    print('Phase 3 Complete')

    # Save checkpoint after pre-training phases
    save_path = saver.save(sess, ckpt_prefix, global_step=0)
    print(f'[Checkpoint] Pre-training saved → {save_path}')
    push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    # ================================================================
    # Phase 4: Joint Training (1× iterations)
    # ================================================================

    print(f'Phase 4: Joint Training ({iterations} iters)')

    step_d_loss = 0.0
    step_d_ae_loss = 0.0

    for itt in range(iterations):
        # Generator training (2× per discriminator step)
        for kk in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
            Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

            _, step_g_loss_u, step_g_loss_s, step_g_loss, step_g_loss_ts = sess.run(
                [G_solver, G_loss_U_totall, G_loss_S, G_loss, G_loss_ts],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

            _, step_e_loss_t0 = sess.run(
                [E_solver, E_loss],
                feed_dict={Z: Z_mb, X: X_mb, T: T_mb})

        # Discriminator training (with threshold)
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)

        check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if check_d_loss > 0.15:
            _, step_d_loss = sess.run([D_solver, D_loss],
                                      feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

        check_d_ae_loss = sess.run(D_ae_loss_second, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
        if check_d_ae_loss > 0.15:
            _, step_d_ae_loss = sess.run([D_ae_solver_second, D_ae_loss_second],
                                          feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

        # Logging
        if itt % 1000 == 0 or itt == iterations - 1:
            print(f'  step: {itt}/{iterations}'
                  f', D: {np.round(step_d_loss, 4)}'
                  f', G: {np.round(step_g_loss, 4)}'
                  f', G_u: {np.round(step_g_loss_u, 4)}'
                  f', G_s: {np.round(step_g_loss_s, 4)}'
                  f', G_ts: {np.round(step_g_loss_ts, 4)}'
                  f', AE: {np.round(step_e_loss_t0, 4)}'
                  f', AE_D: {np.round(step_d_ae_loss, 4)}')

        # Periodic checkpoint
        if (itt + 1) % checkpoint_every == 0:
            save_path = saver.save(sess, ckpt_prefix, global_step=itt + 1)
            print(f'[Checkpoint] Saved at step {itt+1} → {save_path}')
            push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

        # Early stopping evaluation (after halfway, every 500 steps)
        if (itt >= int(iterations * 0.5)) and (itt % 500 == 0 or itt == iterations - 1):
            Z_mb_gen = random_generator(no, z_dim, ori_time, max_seq_len)
            generated_data_curr = sess.run(X_hat,
                                            feed_dict={Z: Z_mb_gen, X: ori_data, T: ori_time})
            generated_data = []
            for i in range(no):
                temp = generated_data_curr[i, :ori_time[i], :]
                generated_data.append(temp)

            generated_data = generated_data * max_val
            generated_data = generated_data + min_val

            metric_iteration = 6
            discriminative_score = []
            for _ in range(metric_iteration):
                temp_disc = discriminative_score_metrics(ori_data, generated_data)
                discriminative_score.append(temp_disc)

            discriminative_score = np.array(discriminative_score)
            mean_dis_score = np.round(np.min(discriminative_score), 4)
            summing = mean_dis_score

            if summing <= global_summing:
                global_summing = summing
                final_generated = generated_data
                # Save best model
                save_path = saver.save(sess, ckpt_prefix + '_best', global_step=itt)
                print(f'  [EarlyStop] New best score={summing} at step {itt} → {save_path}')
                push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    # Final checkpoint
    save_path = saver.save(sess, ckpt_prefix, global_step=iterations)
    print(f'[Checkpoint] Final save → {save_path}')
    push_checkpoints(checkpoint_dir, drive_folder_id, gdrive_secret)

    print('Finish Joint Training')

    # ================================================================
    # Generate Synthetic Data
    # ================================================================

    if num_samples == "same":
        return final_generated

    else:
        count = max(1, int(num_samples / no))
        all_generated_data = []
        for c in range(count):
            Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
            generated_data_curr = sess.run(X_hat,
                                            feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
            generated_data = []
            for i in range(no):
                temp = generated_data_curr[i, :ori_time[i], :]
                generated_data.append(temp)

            generated_data = generated_data * max_val
            generated_data = generated_data + min_val
            all_generated_data.append(generated_data)

        all_generated_data = np.concatenate(all_generated_data)
        return all_generated_data
