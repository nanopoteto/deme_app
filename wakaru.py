import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from keras.optimizers import Adam, SGD
from keras import backend as K

from .cond_mask_base import CondMaskBaseModel
from .layers import *
from .utils import *

from IPython import embed
from IPython.terminal.embed import InteractiveShellEmbed

GENE_LOSS_NORM_P = 'l2'

class ClassifierLossLayer(Layer):
    __name__ = 'classifier_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(ClassifierLossLayer, self).__init__(**kwargs)

    def lossfun(self, c_true, c_fake, c_pred_real, c_pred_fake_f, c_pred_fake_p):
        real_loss = K.mean(keras.metrics.binary_crossentropy(c_true, c_pred_real))
        fake_loss_f = K.mean(keras.metrics.binary_crossentropy(c_fake, c_pred_fake_f))
        fake_loss_p = K.mean(keras.metrics.binary_crossentropy(c_fake, c_pred_fake_p))
        return real_loss + fake_loss_f + fake_loss_p

    def call(self, inputs):
        c_true = inputs[0]
        c_fake = inputs[1]
        c_pred_real = inputs[2]
        c_pred_fake_f = inputs[3]
        c_pred_fake_p = inputs[4]
        loss = self.lossfun(c_true, c_fake, c_pred_real, c_pred_fake_f, c_pred_fake_p)
        self.add_loss(loss, inputs=inputs)

        return c_true

class DiscriminatorLossLayer(Layer):
    __name__ = 'discriminator_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real_a, y_real_b, y_fake_a, y_fake_b, y_fake_ab, y_fake_ba):
        y_pos = K.ones_like(y_real_a)
        y_neg = K.zeros_like(y_real_a)
        loss_real_a = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real_a))
        loss_real_b = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real_b))
        loss_fake_a = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake_a))
        loss_fake_b = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake_b))
        loss_fake_ab = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake_ab))
        loss_fake_ba = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake_ba))
        return loss_real_a + loss_real_b + loss_fake_a + loss_fake_b + loss_fake_ab + loss_fake_ba

    def call(self, inputs):
        y_real_a  = inputs[0]
        y_real_b  = inputs[1]
        y_fake_a  = inputs[2]
        y_fake_b  = inputs[3]
        y_fake_ab = inputs[4]
        y_fake_ba = inputs[5]
        loss = self.lossfun(y_real_a, y_real_b, y_fake_a, y_fake_b, y_fake_ab, y_fake_ba)
        self.add_loss(loss, inputs=inputs)

        return y_real_a

class VaeLossLayer(Layer):
    __name__ = 'simple_loss_layer'

    def __init__(self, metric='l2', **kwargs):
        self.is_placeholder = True
        self.metric = metric
        self.metric_fun = None
        super(VaeLossLayer, self).__init__(**kwargs)

    def build(self, input):
        if self.metric == 'l1':
            self.metric_fun = lambda x, y: K.abs(x - y)
        elif self.metric == 'l2':
            self.metric_fun = lambda x, y: K.square(x - y)
        else:
            raise Exception('Unknown norm type: %s' % p)

    def lossfun(self, x_r_hair, x_r_face, x_f_hair, x_f_face):
        rec_loss_hair = K.mean(K.sum(self.metric_fun(x_r_hair, x_f_hair), axis=[1, 2, 3]))
        rec_loss_face = K.mean(K.sum(self.metric_fun(x_r_face, x_f_face), axis=[1, 2, 3]))

        return rec_loss_hair + rec_loss_face

    def call(self, inputs):
        x_r_hair = inputs[0]
        x_r_face = inputs[1]
        x_f_hair = inputs[2]
        x_f_face = inputs[3]
        loss = self.lossfun(x_r_hair, x_r_face, x_f_hair, x_f_face)
        self.add_loss(loss, inputs=inputs)

        return x_r_hair

class GeneratorLossLayer(Layer):
    __name__ = 'generator_loss_layer'

    def __init__(self, w_rec=1.0, w_D_face=1.0, w_D_hair=1.0, w_C=1.0, metric='l2', **kwargs):
        self.is_placeholder = True
        self.w_rec = K.variable(w_rec)
        self.w_D_face = K.variable(w_D_face)
        self.w_D_hair = K.variable(w_D_hair)
        self.w_C = K.variable(w_C)
        self.metric = metric
        self.metric_fun = None
        super(GeneratorLossLayer, self).__init__(**kwargs)

    def build(self, input):
        if self.metric == 'l1':
            self.metric_fun = lambda x, y: K.abs(x - y)
        elif self.metric == 'l2':
            self.metric_fun = lambda x, y: K.square(x - y)
        else:
            raise Exception('Unknown norm type: %s' % p)

    def lossfun(self, x_r_a, x_f_a, f_D_x_r_a, f_D_x_f_a, f_D_x_r_face_a, f_D_x_r_hair_a, f_D_x_f_face_a, f_D_x_f_hair_a, f_D_x_f_face_ba, f_D_x_f_hair_ab):
        rec_loss = K.mean(K.sum(self.metric_fun(x_r_a, x_f_a), axis=[1, 2, 3]))
        f_d_loss = K.mean(K.sum(K.square(f_D_x_r_a - f_D_x_f_a), axis=-1))
        f_d_loss_hair_a = K.mean(K.sum(K.square(f_D_x_r_hair_a - f_D_x_f_hair_a), axis=-1))
        f_d_loss_face_a = K.mean(K.sum(K.square(f_D_x_r_face_a - f_D_x_f_face_a), axis=-1))
        f_d_loss_face_ba = K.mean(K.sum(K.square(f_D_x_r_face_a - f_D_x_f_face_ba), axis=-1))
        f_d_loss_hair_ab = K.mean(K.sum(K.square(f_D_x_r_hair_a - f_D_x_f_hair_ab), axis=-1))

        return rec_loss + f_d_loss + f_d_loss_hair_a + f_d_loss_face_a + f_d_loss_face_ba + f_d_loss_hair_ab
        # return self.w_rec * rec_loss + self.w_D_face * f_d_loss_face + self.w_D_hair * f_d_loss_hair + self.w_C * f_c_loss

    def call(self, inputs):
        x_r_a = inputs[0]
        x_f_a = inputs[1]
        f_D_x_r_a = inputs[2]
        f_D_x_f_a = inputs[3]
        f_D_x_r_face_a = inputs[4]
        f_D_x_r_hair_a = inputs[5]
        f_D_x_f_face_a = inputs[6]
        f_D_x_f_hair_a = inputs[7]
        f_D_x_f_face_ba = inputs[8]
        f_D_x_f_hair_ab = inputs[9]
        loss = self.lossfun(x_r_a, x_f_a, f_D_x_r_a, f_D_x_f_a, f_D_x_r_face_a, f_D_x_r_hair_a, f_D_x_f_face_a, f_D_x_f_hair_a, f_D_x_f_face_ba, f_D_x_f_hair_ab)
        self.add_loss(loss, inputs=inputs)

        return x_r_a

class FeatureMatchingLayer(Layer):
    __name__ = 'feature_matching_layer'

    def __init__(self, lmbda=1.0, **kwargs):
        self.is_placeholder = True
        self.lmbda = K.variable(lmbda)
        super(FeatureMatchingLayer, self).__init__(**kwargs)

    def lossfun(self, f1, f2):
        f1_avg = K.mean(f1, axis=0)
        f2_avg = K.mean(f2, axis=0)
        return 0.5 * K.mean(K.sum(K.square(f1_avg - f2_avg), axis=-1)) * self.lmbda

    def call(self, inputs):
        f1 = inputs[0]
        f2 = inputs[1]
        loss = self.lossfun(f1, f2)
        self.add_loss(loss, inputs=inputs)

        return f1

class KLLossLayer(Layer):
    __name__ = 'kl_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(KLLossLayer, self).__init__(**kwargs)

    def lossfun(self, z_avg, z_log_var):
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
        return kl_loss

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        loss = self.lossfun(z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return z_avg

def discriminator_accuracy(x_r, x_f, x_p):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_r)
        x_neg = K.zeros_like(x_r)
        loss_r = K.mean(keras.metrics.binary_accuracy(x_pos, x_r))
        loss_p = K.mean(keras.metrics.binary_accuracy(x_neg, x_p))
        loss_f = K.mean(keras.metrics.binary_accuracy(x_neg, x_f))
        return (loss_r + loss_p + loss_f) / 3.0

    return accfun

def z_discriminator_accuracy(x_f, x_p):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_f)
        x_neg = K.zeros_like(x_f)
        loss_f = K.mean(keras.metrics.binary_accuracy(x_pos, x_f))
        loss_p = K.mean(keras.metrics.binary_accuracy(x_neg, x_p))
        return 0.5 * (loss_f + loss_p)

    return accfun

def generator_accuracy(x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_f)
        loss_f = K.mean(keras.metrics.binary_accuracy(x_pos, x_f))
        return loss_f

    return accfun


def classifier_accuracy(c_true, c_pred):
    def accfun(y0, y1):
        loss = K.mean(keras.metrics.binary_accuracy(c_true, c_pred))
        return loss

    return accfun

def ExtractionLayer(s, e, output_shape):
    def fun(inputs):
        return Lambda(lambda x: x[:, s:e], output_shape=output_shape)(inputs)

    return fun

class HAIRGAN(CondMaskBaseModel):
    def __init__(self,
        input_shape=(64, 64, 3),
        z_dims = 256,
        name='hairgan',
        **kwargs
    ):
        super(HAIRGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.input_shape = input_shape
        self.z_dims = z_dims

        self.f_enc = None
        self.f_face_dec = None
        self.f_hair_dec = None
        self.f_dis = None
        self.f_face_dis = None
        self.f_hair_dis = None

        self.enc_trainer = None
        self.gen_trainer = None
        self.dis_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):

        x_r_a, x_r_hair_a, x_r_face_a, x_r_b, x_r_hair_b, x_r_face_b = x_batch

        batchsize = len(x_r_a)
        z_p = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')

        x_dummy = np.zeros(x_r_a.shape, dtype='float32')
        z_dummy = np.zeros(z_p.shape, dtype='float32')
        y_dummy = np.zeros((batchsize, 1), dtype='float32')
        f_dummy = np.zeros((batchsize, 1024), dtype='float32')

        # Train autoencoder
        self.enc_trainer.train_on_batch([x_r_a, x_r_b, x_r_hair_a, x_r_hair_b, x_r_face_a, x_r_face_b],
                                        [x_dummy, x_dummy, x_dummy, x_dummy, z_dummy, z_dummy, z_dummy, z_dummy])


        # Train generator
        g_losses = self.gen_trainer.train_on_batch([x_r_a, x_r_b, x_r_hair_a, x_r_hair_b, x_r_face_a, x_r_face_b],
                                                   [x_dummy, x_dummy, x_dummy, x_dummy])
        g_loss  = g_losses[0]
        g_acc = g_losses[5]

        # Train discriminator
        d_losses = self.dis_trainer.train_on_batch([x_r_a, x_r_b, x_r_hair_a, x_r_hair_b, x_r_face_a, x_r_hair_b], [y_dummy, y_dummy, y_dummy])

        d_loss = d_losses[0]

        losses = [
            ('g_loss'     , g_loss),
            ('g_acc'      , g_acc),
            ('d_loss'     , d_loss),
        ]
        return losses

    def predict(self, z_samples):
        z = z_samples
        x = self.f_dec.predict([z, c])
        return x

    def reconstruct(self, x_r):
        x_f = self.f_rec.predict(x_r)
        x_f_hair, x_f_face = self.f_rec_part.predict(x_r)
        return x_f, x_f_hair, x_f_face

    def swapface(self, x_batch):
        x_r_a, x_r_b = x_batch
        x_f_ab, x_f_ba = self.f_swap.predict([x_r_a, x_r_b])
        return x_f_ab, x_f_ba

    # def get_z_params(self, x_batch):
    #     x_r_hair, x_r_face, c, c_angle = x_batch
    #
    #     z_f_hair_avg, z_f_hair_log_var = self.f_ext_z_hair.predict([x_r_hair, c])
    #     z_f_face_avg, z_f_face_log_var = self.f_ext_z_face.predict([x_r_face, c_angle])
    #
    #     z_f_hair = np.concatenate([z_f_hair_avg, z_f_hair_log_var], axis=-1)
    #     z_f_face = np.concatenate([z_f_face_avg, z_f_face_log_var], axis=-1)
    #
    #     return z_f_hair, z_f_face

    # def swapface(self, x_batch):
    #     x_hair_1, x_face_1, x_hair_2, x_face_2, c_1, c_2, c_angle_1, c_angle_2 = x_batch
    #
    #     z_f_hair_avg_1, z_f_hair_log_var_1 = self.f_ext_z_hair.predict([x_hair_1, c_1])
    #     z_f_face_avg_1, z_f_face_log_var_1 = self.f_ext_z_face.predict([x_face_1, c_angle_1])
    #
    #     z_f_hair_avg_2, z_f_hair_log_var_2 = self.f_ext_z_hair.predict([x_hair_2, c_2])
    #     z_f_face_avg_2, z_f_face_log_var_2 = self.f_ext_z_face.predict([x_face_2, c_angle_2])
    #
    #     z_avg_1 = np.concatenate([z_f_hair_avg_1, z_f_face_avg_2], axis=-1)
    #     z_avg_2 = np.concatenate([z_f_hair_avg_2, z_f_face_avg_1], axis=-1)
    #
    #     x_rec_1 = self.f_dec.predict([z_avg_1, c_1])
    #     x_rec_2 = self.f_dec.predict([z_avg_2, c_2])

        # return x_rec_1, x_rec_2

    def build_model(self):
        self.f_enc      = self.build_encoder(output_dims=self.z_dims*2)
        self.f_dec      = self.build_decoder()
        self.f_face_dec = self.build_half_decoder()
        self.f_hair_dec = self.build_half_decoder()
        self.f_dis      = self.build_discriminator()
        self.f_face_dis = self.build_pair_discriminator()
        self.f_hair_dis = self.build_pair_discriminator()

        # Algorithm
        x_r_a      = Input(shape=self.input_shape, name='x_r_a')
        x_r_hair_a = Input(shape=self.input_shape, name='x_r_hair_a')
        x_r_face_a = Input(shape=self.input_shape, name='x_r_face_a')
        x_r_b      = Input(shape=self.input_shape, name='x_r_b')
        x_r_hair_b = Input(shape=self.input_shape, name='x_r_hair_b')
        x_r_face_b = Input(shape=self.input_shape, name='x_r_face_b')

        z_params_a = self.f_enc([x_r_a])
        z_params_b = self.f_enc([x_r_b])

        z_hair_avg_a     = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_params_a)
        z_hair_log_var_a = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_params_a)
        z_face_avg_a     = ExtractionLayer(self.z_dims*2//2, self.z_dims*3//2, (self.z_dims//2,))(z_params_a)
        z_face_log_var_a = ExtractionLayer(self.z_dims*3//2, self.z_dims*4//2, (self.z_dims//2,))(z_params_a)
        z_hair_avg_b     = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_params_b)
        z_hair_log_var_b = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_params_b)
        z_face_avg_b     = ExtractionLayer(self.z_dims*2//2, self.z_dims*3//2, (self.z_dims//2,))(z_params_b)
        z_face_log_var_b = ExtractionLayer(self.z_dims*3//2, self.z_dims*4//2, (self.z_dims//2,))(z_params_b)

        kl_loss_hair_a = KLLossLayer()([z_hair_avg_a, z_hair_log_var_a])
        kl_loss_face_a = KLLossLayer()([z_face_avg_a, z_face_log_var_a])
        kl_loss_hair_b = KLLossLayer()([z_hair_avg_b, z_hair_log_var_b])
        kl_loss_face_b = KLLossLayer()([z_face_avg_b, z_face_log_var_b])

        # z concatenate
        z_avg_a     = Concatenate(axis=-1)([z_hair_avg_a, z_face_avg_a])
        z_log_var_a = Concatenate(axis=-1)([z_hair_log_var_a, z_face_log_var_a])
        z_avg_b     = Concatenate(axis=-1)([z_hair_avg_b, z_face_avg_b])
        z_log_var_b = Concatenate(axis=-1)([z_hair_log_var_b, z_face_log_var_b])

        # z_p  = Input(shape=(self.z_dims,))
        z_f_a  = SampleNormal()([z_avg_a, z_log_var_a])
        z_f_b  = SampleNormal()([z_avg_b, z_log_var_b])

        # z_p_hair = Input(shape=(self.z_dims//2,))
        z_f_hair_a = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_f_a)
        z_f_hair_b = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_f_b)

        # z_p_face = Input(shape=(self.z_dims//2,))
        z_f_face_a = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_f_a)
        z_f_face_b = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_f_b)

        # hair fast
        z_f_ab = Concatenate(axis=-1)([z_f_hair_a, z_f_face_b])
        z_f_ba = Concatenate(axis=-1)([z_f_hair_b, z_f_face_a])

        # x_p = self.f_dec([z_p, c])
        x_f_a  = self.f_dec(z_f_a)
        x_f_b  = self.f_dec(z_f_b)
        x_f_ab = self.f_dec(z_f_ab)
        x_f_ba = self.f_dec(z_f_ba)

        x_f_hair_a  = self.f_hair_dec(z_f_hair_a)
        x_f_face_a  = self.f_face_dec(z_f_face_a)
        x_f_hair_b  = self.f_hair_dec(z_f_hair_b)
        x_f_face_b  = self.f_face_dec(z_f_face_b)

        x_r_concate_face_a  = Concatenate(axis=-1)([x_r_a,  x_r_face_a])
        x_f_concate_face_a  = Concatenate(axis=-1)([x_f_a,  x_r_face_a])
        x_f_concate_face_ab = Concatenate(axis=-1)([x_f_ba, x_r_face_a])
        x_r_concate_face_b  = Concatenate(axis=-1)([x_r_b,  x_r_face_b])
        x_f_concate_face_b  = Concatenate(axis=-1)([x_f_b,  x_r_face_b])
        x_f_concate_face_ba = Concatenate(axis=-1)([x_f_ab, x_r_face_b])

        x_r_concate_hair_a  = Concatenate(axis=-1)([x_r_a,  x_r_hair_a])
        x_f_concate_hair_a  = Concatenate(axis=-1)([x_f_a,  x_r_hair_a])
        x_f_concate_hair_ab = Concatenate(axis=-1)([x_f_ab, x_r_hair_a])
        x_r_concate_hair_b  = Concatenate(axis=-1)([x_r_b,  x_r_hair_b])
        x_f_concate_hair_b  = Concatenate(axis=-1)([x_f_b,  x_r_hair_b])
        x_f_concate_hair_ba = Concatenate(axis=-1)([x_f_ba, x_r_hair_b])

        y_r_a,  f_D_x_r_a  = self.f_dis(x_r_a)
        y_r_b,  f_D_x_r_b  = self.f_dis(x_r_b)
        y_f_a,  f_D_x_f_a  = self.f_dis(x_f_a)
        y_f_b,  f_D_x_f_b  = self.f_dis(x_f_b)
        y_f_ab, f_D_x_f_ab = self.f_dis(x_f_ab)
        y_f_ba, f_D_x_f_ba = self.f_dis(x_f_ba)

        y_r_face_a,  f_D_x_r_face_a   = self.f_face_dis(x_r_concate_face_a)
        y_r_face_b,  f_D_x_r_face_b   = self.f_face_dis(x_r_concate_face_b)
        y_f_face_a,  f_D_x_f_face_a   = self.f_face_dis(x_f_concate_face_a)
        y_f_face_b,  f_D_x_f_face_b   = self.f_face_dis(x_f_concate_face_b)
        y_f_face_ab, f_D_x_f_face_ab  = self.f_face_dis(x_f_concate_face_ab)
        y_f_face_ba, f_D_x_f_face_ba  = self.f_face_dis(x_f_concate_face_ba)

        y_r_hair_a,  f_D_x_r_hair_a   = self.f_hair_dis(x_r_concate_hair_a)
        y_r_hair_b,  f_D_x_r_hair_b   = self.f_hair_dis(x_r_concate_hair_b)
        y_f_hair_a,  f_D_x_f_hair_a   = self.f_hair_dis(x_f_concate_hair_a)
        y_f_hair_b,  f_D_x_f_hair_b   = self.f_hair_dis(x_f_concate_hair_b)
        y_f_hair_ab, f_D_x_f_hair_ab  = self.f_hair_dis(x_f_concate_hair_ab)
        y_f_hair_ba, f_D_x_f_hair_ba  = self.f_hair_dis(x_f_concate_hair_ba)

        d_loss      = DiscriminatorLossLayer()([y_r_a, y_r_b, y_f_a, y_f_b, y_f_ab, y_f_ba])
        d_loss_face = DiscriminatorLossLayer()([y_r_face_a, y_r_face_b, y_f_face_a, y_f_face_b, y_f_face_ab, y_f_face_ba])
        d_loss_hair = DiscriminatorLossLayer()([y_r_hair_a, y_r_hair_b, y_f_hair_a, y_f_hair_b, y_f_hair_ab, y_f_hair_ba])

        g_loss_a  = GeneratorLossLayer(metric=GENE_LOSS_NORM_P, w_C=1.0)([x_r_a, x_f_a, f_D_x_r_a, f_D_x_f_a, f_D_x_r_face_a, f_D_x_r_hair_a, f_D_x_f_face_a, f_D_x_f_face_b, f_D_x_f_face_ba, f_D_x_f_hair_ab])
        g_loss_b  = GeneratorLossLayer(metric=GENE_LOSS_NORM_P, w_C=1.0)([x_r_b, x_f_b, f_D_x_r_b, f_D_x_f_b, f_D_x_r_face_b, f_D_x_r_hair_b, f_D_x_f_face_b, f_D_x_f_face_a, f_D_x_f_face_ab, f_D_x_f_hair_ba])

        l2_loss_a = VaeLossLayer()([x_r_hair_a, x_r_face_a, x_f_hair_a, x_f_face_a])
        l2_loss_b = VaeLossLayer()([x_r_hair_b, x_r_face_b, x_f_hair_b, x_f_face_b])

        # Build discriminator trainer
        set_trainable(self.f_enc,      False)
        set_trainable(self.f_dec,      False)
        set_trainable(self.f_face_dec, False)
        set_trainable(self.f_hair_dec, False)
        set_trainable(self.f_dis,      True )
        set_trainable(self.f_face_dis, True )
        set_trainable(self.f_hair_dis, True )

        self.dis_trainer = Model(inputs=[x_r_a, x_r_b, x_r_hair_a, x_r_hair_b, x_r_face_a, x_r_face_b],
                                 outputs=[d_loss, d_loss_face, d_loss_hair],
                                 name='discriminator')
        self.dis_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
                                 optimizer=Adam(lr=2.0e-5, beta_1=0.1),)
                                #  metrics=[discriminator_accuracy(y_r_face, y_f_face, y_p_face)])
        self.dis_trainer.summary()

        # Build encoder trainer
        set_trainable(self.f_enc,      True )
        set_trainable(self.f_dec,      False)
        set_trainable(self.f_face_dec, False)
        set_trainable(self.f_hair_dec, False)
        set_trainable(self.f_dis,      False)
        set_trainable(self.f_face_dis, False)
        set_trainable(self.f_hair_dis, False)

        self.enc_trainer = Model(inputs=[x_r_a, x_r_b, x_r_hair_a, x_r_hair_b, x_r_face_a, x_r_face_b],
                                 outputs=[l2_loss_a, l2_loss_b, g_loss_a, g_loss_b, kl_loss_hair_a, kl_loss_face_a, kl_loss_hair_b, kl_loss_face_b],
                                 name='encoder')
        self.enc_trainer.compile(loss=[zero_loss] * 8,
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.enc_trainer.summary()


        # Build generator
        set_trainable(self.f_enc,      False)
        set_trainable(self.f_dec,      True )
        set_trainable(self.f_face_dec, True )
        set_trainable(self.f_hair_dec, True )
        set_trainable(self.f_dis,      False)
        set_trainable(self.f_face_dis, False)
        set_trainable(self.f_hair_dis, False)

        self.gen_trainer = Model(inputs=[x_r_a, x_r_b, x_r_hair_a, x_r_hair_b, x_r_face_a, x_r_face_b],
                                 outputs=[l2_loss_a, l2_loss_b, g_loss_a, g_loss_b],
                                 name='generator')
        self.gen_trainer.compile(loss=[zero_loss] * 4,
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_f_a)])
        self.gen_trainer.summary()

        # #Functions for editing applications
        self.f_rec = Model(inputs=[x_r_a],
                           outputs=[x_f_a],
                           name='f_rec')

        self.f_rec_part = Model(inputs=[x_r_a],
                           outputs=[x_f_hair_a, x_f_face_a],
                           name='f_rec_part')

        self.f_swap = Model(inputs=[x_r_a, x_r_b],
                           outputs=[x_f_ab, x_f_ba],
                           name='f_swap')
        #
        # # Functions for editing applications
        # self.f_ext_z_hair = Model(inputs=[x_r_hair, c],
        #                           outputs=[z_hair_avg, z_hair_log_var],
        #                           name=['f_ext_z_hair'])
        #
        # self.f_ext_z_face = Model(inputs=[x_r_face, c_angle],
        #                           outputs=[z_face_avg, z_face_log_var],
        #                           name=['f_ext_z_face'])

        # Store trainers
        self.store_to_save('enc_trainer')
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')

    def build_encoder(self, output_dims):
        x_inputs = Input(shape=self.input_shape)
        # c_inputs = Input(shape=(self.num_attrs,))

        # c = Reshape((1, 1, self.num_attrs))(c_inputs)
        # c = UpSampling2D(size=self.input_shape[:2])(c)
        # x = Concatenate(axis=-1)([x_inputs, c])

        x = BasicConvLayer(filters=128, strides=(2, 2))(x_inputs)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(output_dims)(x)
        x = Activation('linear')(x)

        return Model(x_inputs, x)

    def build_decoder(self):
        z_inputs = Input(shape=(self.z_dims,))

        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(z_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(z_inputs, x)

    def build_half_decoder(self):
        z_inputs = Input(shape=(self.z_dims//2,))

        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(z_inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(z_inputs, x)

    def build_pair_discriminator(self):
        w, h, c = self.input_shape
        inputs = Input(shape=(w, h, c * 2))

        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(1, 1))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(1, 1))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        f = Activation('relu')(x)

        x = Dense(1)(f)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])

    def build_discriminator(self):
        inputs = Input(shape=(self.input_shape))

        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(1, 1))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(1, 1))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        f = Activation('relu')(x)

        x = Dense(1)(f)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])

    def build_classifier(self):
        w, h, c = self.input_shape
        inputs = Input(shape=(w, h, c * 1))

        x = BasicConvLayer(filters=128, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(1, 1))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(1, 1))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        f = Activation('relu')(x)

        x = Dense(self.num_attrs)(f)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])
