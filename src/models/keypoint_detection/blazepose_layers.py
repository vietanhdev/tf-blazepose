import tensorflow as tf


class ChannelPadding(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(ChannelPadding, self).__init__()
        self.channels = channels

    def build(self, input_shapes):
        self.pad_shape = tf.constant(
            [[0, 0], [0, 0], [0, 0], [0, self.channels - input_shapes[-1]]])

    def call(self, input):
        return tf.pad(input, self.pad_shape)


class BlazeBlock(tf.keras.Model):
    def __init__(self, block_num=3, channel=48, channel_padding=1):
        super(BlazeBlock, self).__init__()

        self.downsample_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=(
                2, 2), padding='same', activation=None),
            tf.keras.layers.Conv2D(
                filters=channel, kernel_size=1, activation=None)
        ])
        if channel_padding:
            self.downsample_b = tf.keras.models.Sequential([
                tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
                ChannelPadding(channels=channel)
            ])
        else:
            self.downsample_b = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

        self.conv = list()
        for i in range(block_num):
            self.conv.append(tf.keras.models.Sequential([
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=3, padding='same', activation=None),
                tf.keras.layers.Conv2D(
                    filters=channel, kernel_size=1, activation=None)
            ]))

    def call(self, x):
        x = tf.keras.activations.relu(
            self.downsample_a(x) + self.downsample_b(x))
        for i in range(len(self.conv)):
            x = tf.keras.activations.relu(x + self.conv[i](x))
        return x
