import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dense, PReLU, UpSampling1D, LayerNormalization, Dropout
import custom_function as cf
import numpy as np

class Encoder(tf.keras.Model):
    def __init__(self, frame_size, latent_size, default_float='float32'):
        super(Encoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.encoder = []
        self.encoder.append(Dense(frame_size))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size*2))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size*2))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size*2))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size*2))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size*2))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size*2))
        self.encoder.append(PReLU())
        self.encoder.append(Dense(frame_size//2))
        self.encoder.append(PReLU())


    def call(self, x):
        output = x
        for f in self.encoder:
            output = f(output)

        return output


class Decoder(tf.keras.Model):
    def __init__(self, latent_size, channel_size=128, default_float='float32'):
        super(Decoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.latent_size = latent_size
        self.channel_size = channel_size
        self.style_block = [StyleBlock(pow(2, i+1), self.channel_size) for i in range(9)]
        self.conv_output = Conv1D(1, 3, padding='same', activation='tanh')

    def call(self, latent, step):
        input = tf.constant(0.1, shape=[latent.shape[0], 2, self.channel_size])
        for i in range(step):
            input = self.style_block[i](input, latent)
        output = self.conv_output(input)
        return output

class StyleBlock(tf.Module):
    def __init__(self, latent_size, channel_size, default_float='float32'):
        super(StyleBlock, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.latent_size = latent_size
        self.channel_size = channel_size
        self.alpha = tf.Variable(0.5)
        self.up = UpSampling1D(2)
        self.conv1 = Conv1D(self.channel_size, 3, padding='same')
        self.affine1_scale = Dense(self.channel_size)
        self.affine1_offset = Dense(self.channel_size)
        self.conv2 = Conv1D(self.channel_size, 3, padding='same')
        self.affine2_scale = Dense(self.channel_size)
        self.affine2_offset = Dense(self.channel_size)
        self.prelu = PReLU()
        self.norm = LayerNormalization(axis=1)

    def __call__(self, x, latent):
        latent_sliced = tf.slice(latent, [0, 0], [latent.shape[0], self.latent_size])
        scale1 = tf.expand_dims(self.affine1_scale(latent_sliced), 1)
        offset1 = tf.expand_dims(self.affine1_offset(latent_sliced), 1)
        scale2 = tf.expand_dims(self.affine2_scale(latent_sliced), 1)
        offset2 = tf.expand_dims(self.affine2_offset(latent_sliced), 1)

        upsampled = self.up(x)
        after_conv = self.prelu(self.conv1(upsampled))
        normalized = self.norm(after_conv)
        styled = normalized*scale1 + offset1

        after_conv = self.prelu(self.conv2(styled))
        normalized = self.norm(after_conv)
        styled = normalized*scale2 + offset2

        output = styled + upsampled * self.alpha

        return output