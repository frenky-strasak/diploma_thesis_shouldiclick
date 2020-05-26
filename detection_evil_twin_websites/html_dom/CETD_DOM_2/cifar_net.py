#!/usr/bin/env python3
# Team: František Střasák, Daniel Šmolík
# 40419513-4364-11e9-b0fd-00505601122b
# 3f638e58-4364-11e9-b0fd-00505601122b
#
import os
import datetime
import math
import re

import tensorflow as tf



# The neural network model
class CifarNet(tf.keras.Model):
    def __init__(self, cifar, args):
        self.cifar = cifar

        # REGULARIZER
        self.regularizer = None if args.l2 == 0 else tf.keras.regularizers.L1L2(0.0, args.l2)

        inputs = tf.keras.layers.Input(shape=[cifar.H, cifar.W, cifar.C])
        hidden = self._create_hidden_layers(args.cnn, inputs)
        # Add the final output layer
        assert cifar.LABELS == 2, 'Error: the number of labels is differnet.'
        outputs = tf.keras.layers.Dense(cifar.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        if args.from_file:
            self.load_weights(args.from_file)

        # DECAY
        if args.decay:
            decay_steps = math.ceil(cifar.train.size / args.batch_size) * args.epochs
            if args.decay == 'polynomial':
                decay = tf.keras.optimizers.schedules.PolynomialDecay(args.learning_rate, decay_steps,
                                                                      args.learning_rate_final)
            else:  # exponential decay
                decay_rate = args.learning_rate_final / args.learning_rate
                decay = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate, decay_steps, decay_rate)
            learning_rate = decay
        else:
            learning_rate = args.learning_rate

        # OPTIMIZER
        if args.optimizer == 'SGD':
            momentum = args.momentum if args.momentum else 0.0
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        else:  # Adam
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.batch_size = args.batch_size
        self.epochs = args.epochs

        model_file_name = "{}-{}".format(
            os.path.basename(__file__),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))

        self.save_callback = tf.keras.callbacks.ModelCheckpoint('{}.h5'.format(model_file_name), save_best_only=True,
                                                                period=5)
        self.tb_callback = tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None

    def train(self, train_data_gen=None):
        if train_data_gen is None:
            train_data_gen = self.cifar.train.batches(self.batch_size)
        self.fit_generator(train_data_gen,
                           steps_per_epoch=math.ceil(self.cifar.train.size / self.batch_size),
                           epochs=self.epochs,
                           validation_data=self.cifar.dev.batches(self.batch_size),
                           validation_steps=math.ceil(self.cifar.dev.size / self.batch_size),
                           callbacks=[self.save_callback])

    def validation_accuracy(self):
        test_logs = self.evaluate_generator(self.cifar.dev.batches(self.batch_size),
                                            steps=math.ceil(self.cifar.dev.size / self.batch_size))
        return test_logs[self.metrics_names.index("accuracy")]

    def _create_hidden_layers(self, args_str, inputs):
        prev_layer = inputs
        next_layer_info_string, layers_info_string = self._next_layer_info(args_str)
        while next_layer_info_string:
            prev_layer = self._layer_from_str(prev_layer, next_layer_info_string)
            next_layer_info_string, layers_info_string = self._next_layer_info(layers_info_string)
        return prev_layer

    def _next_layer_info(self, layers_str):
        if not layers_str:
            return '', ''
        bracket_level = 0
        for i, ch in enumerate(layers_str):
            if ch == ',':
                if bracket_level == 0:
                    return layers_str[:i], layers_str[i + 1:]
            elif ch == '[':
                bracket_level += 1
            elif ch == ']':
                bracket_level -= 1
        return layers_str, ''

    def _layer_type_and_info(self, layer_str):
        layer_type = ''
        for i, ch in enumerate(layer_str):
            if ch == '-' or ch == ',':
                return layer_type, layer_str[i + 1:]
            else:
                layer_type += ch
        return layer_type, ''

    def _create_repeated_layers(self, prev_layer, repeated_layers_info_str):
        info = repeated_layers_info_str.split('-', 1)
        repetions, repeated_layers_str = int(info[0]), info[1].lstrip('[').rstrip(']')
        last_layer = prev_layer
        for _ in range(repetions):
            last_layer = self._create_hidden_layers(repeated_layers_str, last_layer)
        return last_layer

    def _create_wide_layer(self, prev_layer, layer_info_str):
        info = layer_info_str.split('-', 1)
        width, wide_layer_str = int(info[0]), info[1].lstrip('[').rstrip(']')
        parallel_layers = []
        for _ in range(width):
            parallel_layers.append(self._create_hidden_layers(wide_layer_str, prev_layer))
        return tf.keras.layers.Add()(parallel_layers)

    def _create_layer_conv(self, prev_layer, layer_info_str, batch_norm=False):
        info = layer_info_str.split('-')
        filters, kernel_size, stride, padding = int(info[0]), int(info[1]), int(info[2]), info[3]
        activation = None if batch_norm else 'relu'
        conv_layer = tf.keras.layers.Conv2D(filters, kernel_size, stride, padding, activation=activation,
                                            use_bias=not batch_norm, kernel_regularizer=self.regularizer,
                                            bias_regularizer=self.regularizer)(prev_layer)
        if batch_norm:
            batch_norm_layer = tf.keras.layers.BatchNormalization()(conv_layer)
            output_layer = tf.keras.layers.Activation('relu')(batch_norm_layer)
        else:
            output_layer = conv_layer
        return output_layer

    def _create_layer_maxpool(self, prev_layer, layer_info_str):
        info = layer_info_str.split('-')
        kernel_size, stride = int(info[0]), int(info[1])
        return tf.keras.layers.MaxPool2D(kernel_size, stride)(prev_layer)

    def _create_layer_avgpool(self, prev_layer, layer_info_str):
        info = layer_info_str.split('-')
        kernel_size, stride = int(info[0]), int(info[1])
        return tf.keras.layers.AvgPool2D(kernel_size, stride)(prev_layer)

    def _create_residual_connection(self, prev_layer, connection_info_str):
        connection_info_str = connection_info_str.lstrip('[').rstrip(']')
        layers_in_residual_connection = self._create_hidden_layers(connection_info_str, prev_layer)
        return tf.keras.layers.Add()([layers_in_residual_connection, prev_layer])

    def _create_layer_flatten(self, prev_layer):
        return tf.keras.layers.Flatten()(prev_layer)

    def _create_layer_dense(self, prev_layer, layer_info_str):
        return tf.keras.layers.Dense(int(layer_info_str), activation='relu', kernel_regularizer=self.regularizer,
                                     bias_regularizer=self.regularizer)(prev_layer)

    def _create_layer_dropout(self, prev_layer, layer_info_str):
        return tf.keras.layers.Dropout(float(layer_info_str))(prev_layer)

    def _layer_from_str(self, prev_layer, layer_str):
        layer_type, layer_info_str = self._layer_type_and_info(layer_str)
        if layer_type == 'RP':
            return self._create_repeated_layers(prev_layer, layer_info_str)
        elif layer_type == 'W':
            return self._create_wide_layer(prev_layer, layer_info_str)
        elif layer_type == 'C':
            return self._create_layer_conv(prev_layer, layer_info_str)
        elif layer_type == 'CB':
            return self._create_layer_conv(prev_layer, layer_info_str, batch_norm=True)
        elif layer_type == 'M':
            return self._create_layer_maxpool(prev_layer, layer_info_str)
        elif layer_type == 'A':
            return self._create_layer_avgpool(prev_layer, layer_info_str)
        elif layer_type == 'R':
            return self._create_residual_connection(prev_layer, layer_info_str)
        elif layer_type == 'F':
            return self._create_layer_flatten(prev_layer)
        elif layer_type == 'D':
            return self._create_layer_dense(prev_layer, layer_info_str)
        elif layer_type == 'DO':
            return self._create_layer_dropout(prev_layer, layer_info_str)
        else:
            raise ValueError("{} is not a valid hidden layer type".format(layer_type))
