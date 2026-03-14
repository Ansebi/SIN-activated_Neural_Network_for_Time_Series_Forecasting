import tensorflow as tf
import numpy as np
import logging


def least_squares(x: np.array, y: np.array):
    """
    returns a, b for y = ax + b given x and y values
    """
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_x2 = np.sum(x**2)
    sum_xy = np.sum(x * y)
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
    b = (sum_y * sum_x2 - sum_x * sum_xy) / (n * sum_x2 - sum_x**2)
    return a, b


def sinenn_weights(model):
    return [
        f'{layer}: {weight:.4f}' for layer, weight in
        zip(
            [i.path.replace('/kernel', '') for i in model.weights],
            [i[0][0] for i in model.get_weights()]
        )
    ]


def auto_n_relative_wavelengths(n: int):
    if n == 1:
        return [1]
    even = 0
    if not n % 2:
        warning = 'It is preferable to have odd number of waves'
        warning += ' so that 1 (single wave) in the middle is also returned.'
        logging.warning(warning)
        even = 1
    waves = list(np.ones(n))
    for i in list(range(1, n // 2 + 1)):
        waves[n // 2 + i - even] = i + 1
        waves[n // 2 - i] = 1 / (i + 1)
    return waves


class SineNN:
    """
    Sin-wave model with linear component\n\n

    example_trainability_map = {\n
      'linear_rotation': True,\n
      'linear_rotation_handler': True,\n
      'y_shift': True,\n
      'y_shift_amplifier': True,\n
      'frequency': False,\n
      'phase_shift': True,\n
      'phase_shift_amplifier': True,\n
      'sin': False,\n
      'amplitude': True,\n
      'output': False\n
    }\n
    _______________________________________\n
    Usage:\n
    sin_model = SineNN(
        X_train,\n
        y_train,\n
        learning_rate=0.1,\n
        linear_trend_trainable=False,\n
        wave_components_trainable=True,\n
        waves=[2, 1, 1/2],\n
        show_summary=True\n
    )()\n
    sin_model.fit(\n
      X_train,\n
      y_train,\n
      epochs=10,\n
      verbose=False\n
    )\n
    y_pred = sin_model.predict(X_test).flatten()\n
    """

    def auto_init(self):
        if any(
            [
                self.init_linear_rotation == 'auto',
                self.init_y_shift == 'auto'
            ]
        ):
            gradient, y_intercept = least_squares(
                self.x.flatten(), self.y
            )
        if self.init_x_shift == 'auto':
            self.init_x_shift = 1,
        if self.init_y_shift == 'auto':
            self.init_y_shift = y_intercept,
        if self.init_linear_rotation == 'auto':
            self.init_linear_rotation = gradient,
        if self.init_frequency == 'auto':
            self.init_frequency = 2 * np.pi / self.wavelen
        if self.init_amplitude == 'auto':
            self.init_amplitude = (self.y.max() - self.y.min()) / 2
        if self.model_name is None:
            self.model_name = 'SineNN'
        if not self.trainability_map:
            self.trainability_map = self.default_trainability_map

    def build_wave(self, wave):
        input_layer = tf.keras.layers.Input(
            name=f'input_{self.wave_count}',
            shape=[self.x.shape[1]]
        )

        frequency_layer = tf.keras.layers.Dense(
            1,
            name=f'frequency_{self.wave_count}',
            kernel_initializer=tf.keras.initializers.
            Constant(value=self.init_frequency / wave),
            use_bias=False
        )(input_layer)

        phase_shift_layer = tf.keras.layers.Dense(
            1,
            name=f'phase_shift_{self.wave_count}',
            activation=lambda x: 0 * x + self.init_x_shift,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.
            Constant(value=self.init_x_shift)
        )(input_layer)

        phase_shift_amplifier_layer = tf.keras.layers.Dense(
            1,
            name=f'phase_shift_amplifier_{self.wave_count}',
            use_bias=False,
            kernel_initializer=tf.keras.initializers.
            Constant(value=1)
        )(phase_shift_layer)

        sin_input_layer = tf.keras.layers.Add(name='sin_input')(
            [
                frequency_layer,
                phase_shift_amplifier_layer
            ]
        )

        sin_layer = tf.keras.layers.Dense(
            1,
            name=f'sin_{self.wave_count}',
            activation=lambda sin_input: tf.math.sin(sin_input),
            kernel_initializer=tf.keras.initializers.
            Constant(value=1),
            use_bias=False
        )(sin_input_layer)

        sin_amplitude_layer = tf.keras.layers.Dense(
            1,
            name=f'amplitude_{self.wave_count}',
            kernel_initializer=tf.keras.initializers.
            Constant(value=self.init_amplitude),
            use_bias=False
        )(sin_layer)

        wave_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=sin_amplitude_layer,
            name=f'Wave_{self.wave_count}'
        )
        if self.wave_components_trainable:
            for layer_name in self.wave_components:
                trainability = self.trainability_map[layer_name]
                layer_name = f'{layer_name}_{self.wave_count}'
                wave_model.get_layer(name=layer_name).trainable = trainability
        else:
            for layer_name in self.wave_components:
                layer_name = f'{layer_name}_{self.wave_count}'
                wave_model.get_layer(name=layer_name).trainable = False
        return wave_model

    def build_trend(self):
        input_layer = tf.keras.layers.Input(
            name='input',
            shape=[self.x.shape[1]]
        )
        linear_rotation_layer = tf.keras.layers.Dense(
            1,
            name='linear_rotation',
            kernel_initializer=tf.keras.initializers.
            Constant(value=self.init_linear_rotation),
            use_bias=False
        )(input_layer)

        linear_rotation_handler_layer = tf.keras.layers.Dense(
            1,
            name='linear_rotation_handler',
            kernel_initializer=tf.keras.initializers.
            Constant(value=1),
            use_bias=False
        )(linear_rotation_layer)

        y_shift_layer = tf.keras.layers.Dense(
            1,
            name='y_shift',
            activation=lambda x: 0 * x + self.init_y_shift,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.
            Constant(value=self.init_y_shift)
        )(input_layer)

        y_shift_amplifier_layer = tf.keras.layers.Dense(
            1,
            name='y_shift_amplifier',
            kernel_initializer=tf.keras.initializers.
            Constant(value=1),
            use_bias=False,
        )(y_shift_layer)

        linear_component_layer = tf.keras.layers.Add(name='linear_component')(
            [
                linear_rotation_handler_layer,
                y_shift_amplifier_layer
            ]
        )
        linear_model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=linear_component_layer,
            name=f'Trend'
        )
        if self.linear_trend_trainable:
            for layer_name in self.linear_components:
                trainability = self.trainability_map[layer_name]
                linear_model.get_layer(
                    name=layer_name).trainable = trainability
        else:
            for layer_name in self.linear_components:
                linear_model.get_layer(name=layer_name).trainable = False
        return linear_model

    def build(self):
        self.wave_count = 0
        input_layer = tf.keras.layers.Input(
            name='input',
            shape=[self.x.shape[1]]
        )

        linear_trend_layer = self.build_trend()(input_layer)
        wave_models = {}
        for wave in self.waves:
            self.wave_count += 1
            wave_models[wave] = self.build_wave(wave)(input_layer)
        add_waves_layer = tf.keras.layers.Add(
            name='add_waves')(list(wave_models.values()))
        waves_output_layer = tf.keras.layers.Lambda(
            lambda x: x / self.wave_count,
            name='waves_output'
        )(add_waves_layer)
        output_layer = tf.keras.layers.Add(name='output')(
            [
                linear_trend_layer,
                waves_output_layer
            ]
        )

        self.model = tf.keras.models.Model(
            inputs=input_layer,
            outputs=output_layer,
            name=self.model_name
        )
        adam = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=adam, loss='mse')
        if self.show_summary:
            self.model.summary()
        return self.model

    def __init__(
        self,
        x, y,
        waves=(1,),
        wavelen=365.25,
        init_x_shift='auto',
        init_y_shift='auto',
        init_linear_rotation='auto',
        init_frequency='auto',
        init_amplitude='auto',
        learning_rate=10**-1,
        linear_trend_trainable=True,
        wave_components_trainable=True,
        trainability_map=None,
        model_name=None,
        show_summary=True
    ) -> None:
        self.x = x
        self.y = y
        self.wavelen = wavelen
        self.waves = waves
        self.init_x_shift = init_x_shift
        self.init_y_shift = init_y_shift
        self.init_linear_rotation = init_linear_rotation
        self.init_frequency = init_frequency
        self.init_amplitude = init_amplitude
        self.learning_rate = learning_rate
        self.linear_trend_trainable = linear_trend_trainable
        self.wave_components_trainable = wave_components_trainable
        self.trainability_map = trainability_map
        self.model_name = model_name
        self.show_summary = show_summary
        self.default_trainability_map = {
            'linear_rotation': True,
            'linear_rotation_handler': True,
            'y_shift': True,
            'y_shift_amplifier': True,
            'frequency': False,
            'phase_shift': True,
            'phase_shift_amplifier': True,
            'sin': False,
            'amplitude': True,
            'output': False
        }
        self.linear_components = [
            'linear_rotation',
            'linear_rotation_handler',
            'y_shift',
            'y_shift_amplifier'
        ]
        self.wave_components = [
            'frequency',
            'phase_shift',
            'phase_shift_amplifier',
            'sin',
            'amplitude'
        ]
        self.auto_init()
        self.build()

    def __call__(self):
        return self.model
