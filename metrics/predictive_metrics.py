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

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(
            hidden_dim,
            return_sequences=True,
            input_shape=(seq_len - 1, dim - 1)
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='mae'
    )

    # Train
    model.fit(
        X_train,
        Y_train,
        epochs=20,
        batch_size=128,
        verbose=0
    )

    # Test on original data
    X_test = ori_data[:, :-1, :(dim - 1)]

    Y_test = ori_data[:, 1:, (dim - 1)]
    Y_test = np.expand_dims(Y_test, axis=-1)

    pred_Y = model.predict(X_test, verbose=0)

    MAE_temp = 0

    for i in range(no):

        MAE_temp += mean_absolute_error(
            Y_test[i].flatten(),
            pred_Y[i].flatten()
        )

    predictive_score = MAE_temp / no

    return predictive_score