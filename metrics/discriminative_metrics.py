import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def discriminative_score_metrics(ori_data, generated_data):

    ori_data = np.asarray(ori_data).astype(np.float32)
    generated_data = np.asarray(generated_data).astype(np.float32)

    no, seq_len, dim = ori_data.shape

    # Labels
    real_labels = np.ones((len(ori_data), 1))
    fake_labels = np.zeros((len(generated_data), 1))

    # Combine
    X = np.concatenate([ori_data, generated_data], axis=0)
    y = np.concatenate([real_labels, fake_labels], axis=0)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    hidden_dim = max(dim // 2, 1)

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(
            hidden_dim,
            return_sequences=False,
            input_shape=(seq_len, dim)
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=128,
        verbose=0
    )

    # Predict
    y_pred = model.predict(X_test, verbose=0)

    y_pred = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)

    discriminative_score = np.abs(0.5 - acc)

    return discriminative_score