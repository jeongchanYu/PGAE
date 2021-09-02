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
        self.encoder.append(Dense(frame_size))
        self.encoder.append(PReLU())


    def call(self, x):
        output = x
        for f in self.encoder:
            output = f(output)

        return output


class Decoder(tf.keras.Model):
    def __init__(self, latent_size, default_float='float32'):
        super(Decoder, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.latent_size = latent_size
        self.style_block = [StyleBlock(pow(2, i+2)) for i in range(8)]
        self.conv_output = Conv1D(1, 3, activation='tanh')

    def call(self, latent, step):
        input = tf.zeros([latent.shape[0], 4, 128])
        for i in range(step):
            input = self.style_block[i](input, latent)
        output = self.conv_output(input)
        return output

class StyleBlock(tf.Module):
    def __init__(self, latent_size, default_float='float32'):
        super(StyleBlock, self).__init__()
        tf.keras.backend.set_floatx(default_float)
        self.latent_size = latent_size
        self.alpha = tf.Variable(0.5)
        self.up = UpSampling1D(2)
        self.conv1 = Conv1D(128, 3, padding='same')
        self.affine1_scale = Dense(128)
        self.affine1_offset = Dense(128)
        self.conv2 = Conv1D(128, 3, padding='same')
        self.affine2_scale = Dense(128)
        self.affine2_offset = Dense(128)
        self.prelu = PReLU()
        self.norm = LayerNormalization(axis=1)

    def call(self, x, latent):
        latent_sliced = tf.slice(latent, [0, 0], [latent.shape[0], self.latent_size])
        latent_sliced = tf.expand_dims(latent_sliced, 1)
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

        output = styled + x * self.alpha

        return output