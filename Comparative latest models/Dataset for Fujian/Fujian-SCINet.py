import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from Dataset_Division import nor_data



class InnerConv1DBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, h: float, kernel_size: int, neg_slope: float = .01, dropout: float = .5,
                 **kwargs):
        if filters <= 0 or h <= 0:
            raise ValueError('filters and h must be positive')

        super().__init__(**kwargs)
        self.conv1d = tf.keras.layers.Conv1D(max(round(h * filters), 1), kernel_size, padding='same')
        self.leakyrelu = tf.keras.layers.LeakyReLU(neg_slope)

        self.dropout = tf.keras.layers.Dropout(dropout)

        self.conv1d2 = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')
        self.tanh = tf.keras.activations.tanh

    def call(self, input_tensor, training=None):
        x = self.conv1d(input_tensor)
        x = self.leakyrelu(x)

        if training:
            x = self.dropout(x)

        x = self.conv1d2(x)
        x = self.tanh(x)
        return x


class SCIBlock(tf.keras.layers.Layer):
    def __init__(self, features: int, kernel_size: int, h: int, name='sciblock', **kwargs):
        """
        :param features: number of features in the output
        :param kernel_size: kernel size of the convolutional layers
        :param h: scaling factor for convolutional module
        """
        super().__init__(name=name, **kwargs)
        self.features = features
        self.kernel_size = kernel_size
        self.h = h

        self.conv1ds = {k: InnerConv1DBlock(filters=self.features, h=self.h, kernel_size=self.kernel_size, name=k)
                        for k in ['psi', 'phi', 'eta', 'rho']}  # regularize?

    def call(self, inputs):
        F_odd, F_even = inputs[:, ::2], inputs[:, 1::2]

        # Interactive learning as described in the paper
        F_s_odd = F_odd * tf.math.exp(self.conv1ds['phi'](F_even))
        F_s_even = F_even * tf.math.exp(self.conv1ds['psi'](F_odd))

        F_prime_odd = F_s_odd + self.conv1ds['rho'](F_s_even)
        F_prime_even = F_s_even - self.conv1ds['eta'](F_s_odd)

        return F_prime_odd, F_prime_even

    def get_config(self):
        config = super().get_config()
        config.update({'features': self.features, 'kernel_size': self.kernel_size, 'h': self.h})
        return config


class Interleave(tf.keras.layers.Layer):
    """A layer used to reverse the even-odd split operation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _interleave(self, slices):
        if not slices:
            return slices
        elif len(slices) == 1:
            return slices[0]

        mid = len(slices) // 2
        even = self._interleave(slices[:mid])
        odd = self._interleave(slices[mid:])

        shape = tf.shape(even)
        return tf.reshape(tf.stack([even, odd], axis=3), (shape[0], shape[1] * 2, shape[2]))

    def call(self, inputs):
        return self._interleave(inputs)


class SCINet(tf.keras.layers.Layer):
    def __init__(self, horizon: int, features: int, levels: int, h: int, kernel_size: int,
                 kernel_regularizer=None, activity_regularizer=None, name='scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param levels: height of the binary tree + 1
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for the fully connected layer at the end
        :param activity_regularizer: activity regularizer for the fully connected layer at the end
        """
        if levels < 1:
            raise ValueError('Must have at least 1 level')

        super().__init__(name=name, **kwargs)
        self.horizon = horizon
        self.features = features
        self.levels = levels
        self.h = h
        self.kernel_size = kernel_size

        self.interleave = Interleave()
        self.flatten = tf.keras.layers.Flatten()

        # tree of sciblocks
        self.sciblocks = [SCIBlock(features=features, kernel_size=self.kernel_size, h=self.h)
                          for _ in range(2 ** self.levels - 1)]
        self.dense = tf.keras.layers.Dense(
            self.horizon * features,
            kernel_regularizer=kernel_regularizer,
            activity_regularizer=activity_regularizer
        )

    def build(self, input_shape):
        if input_shape[1] / 2 ** self.levels % 1 != 0:
            raise ValueError(f'timestamps {input_shape[1]} must be evenly divisible by a tree with '
                             f'{self.levels} levels')
        super().build(input_shape)

    def call(self, inputs):
        # cascade input down a binary tree of sci-blocks
        lvl_inputs = [inputs]  # inputs for current level of the tree
        for i in range(self.levels):
            i_end = 2 ** (i + 1) - 1
            i_start = i_end - 2 ** i
            lvl_outputs = [output for j, tensor in zip(range(i_start, i_end), lvl_inputs)
                           for output in self.sciblocks[j](tensor)]
            lvl_inputs = lvl_outputs

        x = self.interleave(lvl_outputs)
        x += inputs

        # not sure if this is the correct way of doing it. The paper merely said to use a fully connected layer to
        # produce an output. Can't use TimeDistributed wrapper. It would force the layer's timestamps to match that of
        # the input -- something SCINet is supposed to solve
        x = self.flatten(x)
        x = self.dense(x)
        x = tf.reshape(x, (-1, self.horizon, self.features))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({'horizon': self.horizon, 'levels': self.levels})
        return config


class StackedSCINet(tf.keras.layers.Layer):
    """Layer that implements StackedSCINet as described in the paper.

    When called, outputs a tensor of shape (K, -1, n_steps, n_features) containing the outputs of all K internal
    SCINets (e.g., output[k-1] is the output of the kth SCINet, where k is in [1, ..., K]).

    To use intermediate supervision, pass the layer's output to StackedSCINetLoss as a separate model output.
    """

    def __init__(self, horizon: int, features: int, stacks: int, levels: int, h: int, kernel_size: int,
                 kernel_regularizer=None, activity_regularizer=None, name='stacked_scinet', **kwargs):
        """
        :param horizon: number of time stamps in output
        :param stacks: number of stacked SCINets
        :param levels: number of levels for each SCINet
        :param h: scaling factor for convolutional module in each SCIBlock
        :param kernel_size: kernel size of convolutional module in each SCIBlock
        :param kernel_regularizer: kernel regularizer for each SCINet
        :param activity_regularizer: activity regularizer for each SCINet
        """
        if stacks < 2:
            raise ValueError('Must have at least 2 stacks')

        super().__init__(name=name, **kwargs)
        self.stacks = stacks
        self.scinets = [SCINet(horizon=horizon, features=features, levels=levels, h=h,
                               kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                               activity_regularizer=activity_regularizer) for _ in range(stacks)]

    def call(self, inputs):  # sample_weights=None
        outputs = []
        for scinet in self.scinets:
            x = scinet(inputs)
            outputs.append(x)  # keep each stack's output for intermediate supervision
            inputs = tf.concat([x, inputs[:, x.shape[1]:, :]], axis=1)  # X_hat_k concat X_(t-(T-tilda)+1:t)
        return tf.stack(outputs)

    def get_config(self):
        config = super().get_config()
        config.update({'stacks': self.stacks})
        return config


class Identity(tf.keras.layers.Layer):
    """Identity layer used solely for the purpose of naming model outputs and properly displaying outputs when plotting
    some multi-output models.

    Returns input without changing them.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.identity(inputs)


class StackedSCINetLoss(tf.keras.losses.Loss):
    """Compute loss for a Stacked SCINet via intermediate supervision.

    `loss = sum of mean normalised difference between each stack's output and ground truth`

    `y_pred` should be the output of a StackedSCINet layer.
    """

    def __init__(self, name='stacked_scienet_loss', **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        stacked_outputs = y_pred
        horizon = stacked_outputs.shape[2]

        errors = stacked_outputs - y_true
        loss = tf.linalg.normalize(errors, axis=3)[1]
        loss = tf.reduce_sum(loss, 2)
        loss /= horizon
        loss = tf.reduce_sum(loss)

        return loss


# class NetConcatenate(tf.keras.layer.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.concatenate = tf.keras.layers.Concatenate(axis=1)
#
#     def call(self, intermediates, inputs):
#         return self.concatenate([intermediates, inputs[:, intermediates.shape[1]:, :]])


def make_simple_scinet(input_shape, horizon: int, L: int, h: int, kernel_size: int, learning_rate: float,
                       kernel_regularizer=None, activity_regularizer=None, diagram_path=None):
    """Compiles a simple SCINet and saves model diagram if given a path.

    Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
    """
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='inputs'),
        SCINet(horizon, features=input_shape[-1], levels=L, h=h, kernel_size=kernel_size,
               kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer)
    ])

    # model.summary()
    if diagram_path:
        tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mse',
                  metrics=['mse', 'mae']
                  )

    return model


def make_simple_stacked_scinet(input_shape, horizon: int, K: int, L: int, h: int, kernel_size: int,
                               learning_rate: float, kernel_regularizer=None, activity_regularizer=None,
                               diagram_path=None):
    """Compiles a simple StackedSCINet and saves model diagram if given a path.

    Intended to be a demonstration of simple model construction. See paper for details on the hyperparameters.
    """
    inputs = tf.keras.Input(shape=(input_shape[1], input_shape[2]), name='lookback_window')
    x = StackedSCINet(horizon=horizon, features=input_shape[-1], stacks=K, levels=L, h=h,
                      kernel_size=kernel_size, kernel_regularizer=kernel_regularizer,
                      activity_regularizer=activity_regularizer)(inputs)
    outputs = Identity(name='outputs')(x[-1])
    intermediates = Identity(name='intermediates')(x)
    model = tf.keras.Model(inputs=inputs, outputs=[outputs, intermediates])

    model.summary()
    if diagram_path:
        tf.keras.utils.plot_model(model, to_file=diagram_path, show_shapes=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss={
                      # 'outputs': 'mse',
                      'intermediates': StackedSCINetLoss()
                  },
                  metrics={'outputs': ['mse', 'mae']}
                  )

    return model


original_data = pd.read_excel(io="Wind speed data of Fujian.xlsx", sheet_name="Sheet1")
data = original_data.values[0:8760]

outlag = 3
scalarY, x_train, x_test, y_train, y_test = nor_data(data, 1752, 2, 8, outlag)


timesteps = 8
features = 1
batch_size = 32
input_shape = (batch_size, timesteps, features)
horizon = 3
L = 2
h = 2
kernel_size = 3
learning_rate = 0.001


# Creating SCINet model
scinet_model = make_simple_scinet(input_shape, horizon, L, h, kernel_size, learning_rate)

# Training the model
scinet_model.fit(x_train, y_train, epochs=100, batch_size=batch_size, validation_split=0.2, verbose=0)

# Prediction
y_test_prediction = scinet_model.predict(x_test)
y_test_predicted_values = scalarY.inverse_transform(y_test_prediction.reshape(-1, outlag))
y_test_real_values = scalarY.inverse_transform(y_test.reshape(-1, outlag))

save_y_test_predicted_values = pd.DataFrame(y_test_predicted_values)
save_y_test_predicted_values.to_excel('Predicted values of Fujian-SCINet.xlsx', index=False)
save_y_test_real_values = pd.DataFrame(y_test_real_values)
save_y_test_real_values.to_excel('Real values of Fujian-SCINet.xlsx', index=False)


def ia(y: np.array, y_pre: np.array) -> float:
    """
    :param y: the actual values
    :param y_pre: the prediction values
    :return: Index of agreement (IA)
    """
    arr_y = y.ravel()
    arr_y_pre = y_pre.ravel()
    mean_y = arr_y.mean()

    arr_1 = np.square(arr_y - arr_y_pre)
    arr_2 = np.square(np.abs(arr_y_pre - mean_y) + np.abs(arr_y - mean_y))

    IA = 1 - (arr_1.sum() / arr_2.sum())
    return IA


def tic(y: np.array, y_pre: np.array) -> float:
    """
    :param y: the actual values
    :param y_pre: the prediction values
    :return: Theil's inequality coefficient (TIC)
    """
    arr_y = y.ravel()
    arr_y_pre = y_pre.ravel()

    TIC = math.sqrt(np.square(arr_y_pre - arr_y).mean())/(math.sqrt(np.square(arr_y).mean()) + math.sqrt(np.square(arr_y_pre).mean()))

    return TIC


eva_test = np.zeros((5, 3))
for step in range(3):
    def Evaluation_indexes(predicted_values, real_values, eva_test):
        rmse = math.sqrt(mean_squared_error(real_values, predicted_values))
        mae = mean_absolute_error(real_values, predicted_values)
        mape = np.mean(np.abs((predicted_values - real_values) / real_values))
        IA = ia(real_values, predicted_values)
        TIC = tic(real_values, predicted_values)
        print('RMSE: %.4f' % rmse)
        print('MAE: %.4f' % mae)
        print('MAPE: %.6f' % mape)
        print('IA: %.4f' % IA)
        print('TIC: %.4f' % TIC)
        eva_test[0, step] = round(rmse, 4)
        eva_test[1, step] = round(mae, 4)
        eva_test[2, step] = round(float(mape), 4)
        eva_test[3, step] = round(IA, 4)
        eva_test[4, step] = round(TIC, 4)
        return eva_test


    sumpredicted = y_test_predicted_values[:, step]
    sumreal = y_test_real_values[:, step]

    print(f'{step + 1}-step prediction accuracy')
    eva_test = Evaluation_indexes(sumpredicted, sumreal, eva_test)

eva_test = pd.concat(
    [pd.DataFrame(['RMSE', 'MAE', 'MAPE', 'IA', 'TIC']), pd.DataFrame(eva_test)], axis=1)
eva_test.columns = ['Evaluation metrics', '1-step', '2-step', '3-step']
eva_test.to_excel('Fujian-SCINet.xlsx', index=False)
