import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_absolute_error


def predictive_score_metrics(ori_data, generated_data):
    ori_data = np.asarray(ori_data).astype(np.float32)
    generated_data = np.asarray(generated_data).astype(np.float32)

    no, seq_len, dim = ori_data.shape
    hidden_dim = max(dim // 2, 1)

    # Prepare training data
    X_train = generated_data[:, :-1, :(dim - 1)]
    Y_train = generated_data[:, 1:, (dim - 1)]
    Y_train = np.expand_dims(Y_train, axis=-1)

    batch_size = min(128, len(X_train))

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        inp = tf.compat.v1.placeholder(tf.float32, [None, seq_len - 1, dim - 1])
        targets = tf.compat.v1.placeholder(tf.float32, [None, seq_len - 1, 1])

        gru = tf.keras.layers.GRU(hidden_dim, return_sequences=True)(inp)
        logits = tf.keras.layers.Dense(1, activation=None)(gru)
        preds = tf.math.sigmoid(logits)

        loss = tf.reduce_mean(tf.abs(targets - preds))
        optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)

            # Train for 20 epochs
            for epoch in range(20):
                idx = np.random.permutation(len(X_train))
                for i in range(0, len(X_train), batch_size):
                    batch_idx = idx[i:i+batch_size]
                    sess.run(optimizer, feed_dict={inp: X_train[batch_idx], targets: Y_train[batch_idx]})

            # Test on original data
            X_test = ori_data[:, :-1, :(dim - 1)]
            pred_Y = sess.run(preds, feed_dict={inp: X_test})

    Y_test = ori_data[:, 1:, (dim - 1)]
    Y_test = np.expand_dims(Y_test, axis=-1)

    MAE_temp = 0
    for i in range(no):
        MAE_temp += mean_absolute_error(
            Y_test[i].flatten(),
            pred_Y[i].flatten()
        )

    predictive_score = MAE_temp / no
    return predictive_score