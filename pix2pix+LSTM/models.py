import tensorflow as tf
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TimeDistributed, Conv2D, Dropout, Reshape, UpSampling2D, Concatenate
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.compat.v1.keras.initializers import glorot_normal

class discriminator(tf.keras.Model):
    def __init__(self, name=None):
        super(discriminator, self).__init__(name=name)
        self.conv1 = TimeDistributed(conv2d(filters=64, f_size=4, normalization=False,
                                            alpha=0.2, dropout_rate=0))
        self.conv2 = TimeDistributed(conv2d(filters=128, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=0))
        self.conv3 = TimeDistributed(conv2d(filters=256, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=0))
        self.conv4 = TimeDistributed(conv2d(filters=512, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=0))
        self.conv5 = TimeDistributed(Conv2D(1, kernel_size=4, strides=1, padding='same'))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        validity = self.conv5(x)

        return validity

class generator(tf.keras.Model):
    def __init__(self, frames, out_channels, dropout_rate, name=None):
        super(generator, self).__init__(name=name)
        self.conv1 = TimeDistributed(conv2d(filters=64, f_size=4, normalization=False,
                                            alpha=0.2, dropout_rate=0))
        self.conv2 = TimeDistributed(conv2d(filters=128, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=0))
        self.conv3 = TimeDistributed(conv2d(filters=256, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=0))
        self.conv4 = TimeDistributed(conv2d(filters=512, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=dropout_rate))
        self.conv5 = TimeDistributed(conv2d(filters=512, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=dropout_rate))
        self.conv6 = TimeDistributed(conv2d(filters=512, f_size=4, normalization=True,
                                            alpha=0.2, dropout_rate=dropout_rate))
        self.conv7 = TimeDistributed(conv2d(filters=512, f_size=2, normalization=True,
                                            alpha=0.2, dropout_rate=dropout_rate))
        self.conv8 = TimeDistributed(conv2d(filters=512, f_size=2, normalization=False,
                                            alpha=0.2, dropout_rate=dropout_rate))
        self.resh1 = TimeDistributed(Reshape((512, )))
        self.lstm  = CuDNNLSTM(512, batch_input_shape=(None, frames, 512),
                               kernel_initializer=glorot_normal(seed=1),
                               return_sequences=True, stateful=False)
        self.resh2 = TimeDistributed(Reshape((1, 1, 512, )))
        self.deco1 = TimeDistributed(deconv2d(filters=512, f_size=2, dropout_rate=dropout_rate))
        self.deco2 = TimeDistributed(deconv2d(filters=512, f_size=2, dropout_rate=dropout_rate))
        self.deco3 = TimeDistributed(deconv2d(filters=512, f_size=4, dropout_rate=dropout_rate))
        self.deco4 = TimeDistributed(deconv2d(filters=512, f_size=4, dropout_rate=dropout_rate))
        self.deco5 = TimeDistributed(deconv2d(filters=256, f_size=4, dropout_rate=0))
        self.deco6 = TimeDistributed(deconv2d(filters=128, f_size=4, dropout_rate=0))
        self.deco7 = TimeDistributed(deconv2d(filters=64, f_size=4, dropout_rate=0))
        self.upsa  = TimeDistributed(UpSampling2D(size=2))
        self.conv9 = TimeDistributed(Conv2D(out_channels, kernel_size=4, strides=1, padding='same', activation='tanh'))

    def call(self, x):
        d1 = self.conv1(x)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)
        d6 = self.conv6(d5)
        d7 = self.conv7(d6)
        d8 = self.conv8(d7)

        y = self.resh1(d8)
        z = self.lstm(y)
        x = self.resh2(z)

        x = self.deco1(x)
        u1 = Concatenate()([x, d7])
        x = self.deco2(u1)
        u2 = Concatenate()([x, d6])
        x = self.deco3(u2)
        u3 = Concatenate()([x, d5])
        x = self.deco4(u3)
        u4 = Concatenate()([x, d4])
        x = self.deco5(u4)
        u5 = Concatenate()([x, d3])
        x = self.deco6(u5)
        u6 = Concatenate()([x, d2])
        x = self.deco7(u6)
        u7 = Concatenate()([x, d1])
        x = self.upsa(u7)
        output_imgs = self.conv9(x)

        return output_imgs

class conv2d(tf.keras.Model):
    def __init__(self, filters, f_size=4, normalization=True, alpha=0.2, dropout_rate=0, name=None):
        super(conv2d, self).__init__(name=name)
        self.conv = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')
        self.norm = InstanceNormalization()
        self.act  = LeakyReLU(alpha=alpha)
        self.drop = Dropout(dropout_rate)
        self.filters = filters
        self.normalization = normalization
        self.dropout_rate = dropout_rate

    def call(self, x):
        y = self.conv(x)
        if self.normalization:
            y = self.norm(y)
        y = self.act(y)
        if self.dropout_rate:
            y = self.drop(y)
        return y

    def compute_output_shape(self, input_shape):
        input_batch = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]
        output_shape = [input_batch, input_height // 2, input_width // 2, self.filters]
        return output_shape

class deconv2d(tf.keras.Model):
    def __init__(self, filters, f_size=4, dropout_rate=0, name=None):
        super(deconv2d, self).__init__(name=name)
        self.upsa = UpSampling2D(size=2)
        self.conv = Conv2D(filters, kernel_size=f_size, strides=1, padding='same')
        self.norm = InstanceNormalization()
        self.act  = ReLU()
        self.drop = Dropout(dropout_rate)
        self.filters = filters
        self.dropout_rate = dropout_rate

    def call(self, x):
        y = self.upsa(x)
        y = self.conv(y)
        y = self.norm(y)
        y = self.act(y)
        if self.dropout_rate:
            y = self.drop(y)
        return y

    def compute_output_shape(self, input_shape):
        input_batch = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]
        output_shape = [input_batch, 2 * input_height, 2 * input_width, self.filters]
        return output_shape
