import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def discriminative_score_metrics(ori_data, generated_data):
    ori_data = np.asarray(ori_data).astype(np.float32)
    generated_data = np.asarray(generated_data).astype(np.float32)

    no, seq_len, dim = ori_data.shape

    # Labels
    real_labels = np.ones((len(ori_data), 1), dtype=np.float32)
    fake_labels = np.zeros((len(generated_data), 1), dtype=np.float32)

    # Combine
    X = np.concatenate([ori_data, generated_data], axis=0)
    y = np.concatenate([real_labels, fake_labels], axis=0)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    hidden_dim = max(dim // 2, 1)
    batch_size = min(128, len(X_train))

    graph = tf.compat.v1.Graph()
    with graph.as_default():
        inp = tf.compat.v1.placeholder(tf.float32, [None, seq_len, dim])
        labels = tf.compat.v1.placeholder(tf.float32, [None, 1])

        gru = tf.keras.layers.GRU(hidden_dim, return_sequences=False)(inp)
        logits = tf.keras.layers.Dense(1, activation=None)(gru)
        preds = tf.math.sigmoid(logits)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        optimizer = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        init = tf.compat.v1.global_variables_initializer()

        with tf.compat.v1.Session(graph=graph) as sess:
            sess.run(init)

            # Train for 20 epochs
            for epoch in range(20):
                idx = np.random.permutation(len(X_train))
                for i in range(0, len(X_train), batch_size):
                    batch_idx = idx[i:i+batch_size]
                    sess.run(optimizer, feed_dict={inp: X_train[batch_idx], labels: y_train[batch_idx]})

            # Predict
            y_pred_probs = sess.run(preds, feed_dict={inp: X_test})

    y_pred = (y_pred_probs > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score