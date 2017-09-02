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

    def lossfun(self, y_real, y_fake_f, y_fake_p):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_real)
        loss_real = K.mean(keras.metrics.binary_crossentropy(y_pos, y_real))
        loss_f_fake = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake_f))
        loss_p_fake = K.mean(keras.metrics.binary_crossentropy(y_neg, y_fake_p))
        return loss_real + loss_f_fake + loss_p_fake

    def call(self, inputs):
        y_real = inputs[0]
        y_fake_f = inputs[1]
        y_fake_p = inputs[2]
        loss = self.lossfun(y_real, y_fake_f, y_fake_p)
        self.add_loss(loss, inputs=inputs)

        return y_real

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

    def lossfun(self, x_r, x_f, f_D_x_r_face, f_D_x_f_face, f_D_x_r_hair, f_D_x_f_hair, f_C_x_r, f_C_x_f):
        rec_loss = K.mean(K.sum(self.metric_fun(x_r, x_f), axis=[1, 2, 3]))
        f_d_loss_face = K.mean(K.sum(K.square(f_D_x_r_face - f_D_x_f_face), axis=-1))
        f_d_loss_hair = K.mean(K.sum(K.square(f_D_x_r_hair - f_D_x_f_hair), axis=-1))
        f_c_loss = K.mean(K.sum(K.square(f_C_x_r - f_C_x_f), axis=-1))
        return self.w_rec * rec_loss + self.w_D_face * f_d_loss_face + self.w_D_hair * f_d_loss_hair + self.w_C * f_c_loss

    def call(self, inputs):
        x_r = inputs[0]
        x_f = inputs[1]
        f_D_x_r_face = inputs[2]
        f_D_x_f_face = inputs[3]
        f_D_x_r_hair = inputs[4]
        f_D_x_f_hair = inputs[5]
        f_C_x_r = inputs[6]
        f_C_x_f = inputs[7]
        loss = self.lossfun(x_r, x_f, f_D_x_r_face, f_D_x_f_face, f_D_x_r_hair, f_D_x_f_hair, f_C_x_r, f_C_x_f)
        self.add_loss(loss, inputs=inputs)

        return x_r

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

def generator_accuracy(x_p, x_f):
    def accfun(y0, y1):
        x_pos = K.ones_like(x_p)
        loss_p = K.mean(keras.metrics.binary_accuracy(x_pos, x_p))
        loss_f = K.mean(keras.metrics.binary_accuracy(x_pos, x_f))
        return 0.5 * (loss_p + loss_f)

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
        num_attrs=40,
        z_dims = 256,
        name='hairgan',
        **kwargs
    ):
        super(HAIRGAN, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.input_shape = input_shape
        self.num_attrs = num_attrs
        self.z_dims = z_dims

        self.f_hair_enc = None
        self.f_face_enc = None
        self.f_dec = None
        self.f_face_dec = None
        self.f_hair_dec = None
        self.f_face_dis = None
        self.f_hair_dis = None
        self.f_cls = None

        self.enc_trainer = None
        self.gen_trainer = None
        self.dis_trainer = None
        self.cls_trainer = None

        self.build_model()

    def train_on_batch(self, x_batch):

        x_r, x_r_hair, x_r_face, m, c, c_angle = x_batch

        batchsize = len(x_r)
        z_p = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')
        z_p_hair = z_p[:, :self.z_dims//2]
        z_p_face = z_p[:, self.z_dims//2:]

        x_dummy = np.zeros(x_r.shape, dtype='float32')
        c_dummy = np.zeros(c.shape, dtype='float32')
        z_dummy = np.zeros(z_p.shape, dtype='float32')
        y_dummy = np.zeros((batchsize, 1), dtype='float32')
        f_dummy = np.zeros((batchsize, 1024), dtype='float32')

        # Train autoencoder
        self.enc_trainer.train_on_batch([x_r, x_r_hair, x_r_face, c, c_angle, z_p, z_p_face, z_p_hair],
                                        [x_dummy, z_dummy, z_dummy])


        # Train generator
        g_losses = self.gen_trainer.train_on_batch([x_r, x_r_hair, x_r_face, c, c_angle, z_p, z_p_face, z_p_hair],
                                                   [x_dummy, f_dummy, f_dummy, f_dummy])
        g_loss  = g_losses[0]
        g_acc = g_losses[5]

        # Train classifier
        c_fake = np.zeros_like(c)
        c_fake[:, -1] = 1.0
        c_loss, c_acc = self.cls_trainer.train_on_batch([x_r, x_r_hair, x_r_face, c, c_angle, c_fake, z_p, z_p_face, z_p_hair], c_dummy)

        # Train discriminator
        d_loss, _, _, d_acc, _ = self.dis_trainer.train_on_batch([x_r, x_r_hair, x_r_face, c, c_angle, z_p, z_p_face, z_p_hair], [y_dummy, y_dummy])

        losses = [
            ('g_loss'     , g_loss),
            ('g_acc'      , g_acc),
            ('d_loss'     , d_loss),
            ('d_acc'      , d_acc),
            ('c_loss'     , c_loss),
            ('c_acc'      , c_acc)
        ]
        return losses

    def predict(self, z_samples):
        z, c = z_samples
        x = self.f_dec.predict([z, c])
        return x

    def reconstruct(self, x_batch):
        x_r_hair, x_r_face, c, c_angle = x_batch
        x_f = self.f_rec.predict([x_r_hair, x_r_face, c, c_angle])
        return x_f

    def get_z_params(self, x_batch):
        x_r_hair, x_r_face, c, c_angle = x_batch

        z_f_hair_avg, z_f_hair_log_var = self.f_ext_z_hair.predict([x_r_hair, c])
        z_f_face_avg, z_f_face_log_var = self.f_ext_z_face.predict([x_r_face, c_angle])

        z_f_hair = np.concatenate([z_f_hair_avg, z_f_hair_log_var], axis=-1)
        z_f_face = np.concatenate([z_f_face_avg, z_f_face_log_var], axis=-1)

        return z_f_hair, z_f_face

    def swapface(self, x_batch):
        x_hair_1, x_face_1, x_hair_2, x_face_2, c_1, c_2, c_angle_1, c_angle_2 = x_batch

        z_f_hair_avg_1, z_f_hair_log_var_1 = self.f_ext_z_hair.predict([x_hair_1, c_1])
        z_f_face_avg_1, z_f_face_log_var_1 = self.f_ext_z_face.predict([x_face_1, c_angle_1])

        z_f_hair_avg_2, z_f_hair_log_var_2 = self.f_ext_z_hair.predict([x_hair_2, c_2])
        z_f_face_avg_2, z_f_face_log_var_2 = self.f_ext_z_face.predict([x_face_2, c_angle_2])

        z_avg_1 = np.concatenate([z_f_hair_avg_1, z_f_face_avg_2], axis=-1)
        z_avg_2 = np.concatenate([z_f_hair_avg_2, z_f_face_avg_1], axis=-1)

        x_rec_1 = self.f_dec.predict([z_avg_1, c_1])
        x_rec_2 = self.f_dec.predict([z_avg_2, c_2])

        return x_rec_1, x_rec_2

    def build_model(self):
        self.f_hair_enc = self.build_encoder(output_dims=self.z_dims)
        self.f_face_enc = self.build_encoder(output_dims=self.z_dims)
        self.f_dec      = self.build_decoder()
        self.f_face_dec = self.build_half_decoder()
        self.f_hair_dec = self.build_half_decoder()
        self.f_face_dis = self.build_discriminator()
        self.f_hair_dis = self.build_discriminator()
        self.f_cls      = self.build_classifier()

        # Algorithm
        x_r = Input(shape=self.input_shape, name='x_r')
        x_r_hair = Input(shape=self.input_shape, name='x_r_hair')
        x_r_face = Input(shape=self.input_shape, name='x_r_face')
        m = Input(shape=self.input_shape, name='m')
        c = Input(shape=(self.num_attrs,), name='c')
        c_angle = Input(shape=(self.num_attrs,), name='c_angle')
        c_fake = Input(shape=(self.num_attrs,), name='c_fake')

        z_params_hair = self.f_hair_enc([x_r_hair, c])
        z_params_face = self.f_face_enc([x_r_face, c_angle])

        z_hair_avg = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_params_hair)
        z_hair_log_var = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_params_hair)
        z_face_avg = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_params_face)
        z_face_log_var = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_params_face)

        kl_loss_hair = KLLossLayer()([z_hair_avg, z_hair_log_var])
        kl_loss_face = KLLossLayer()([z_face_avg, z_face_log_var])

        # z concatenate
        z_avg = Concatenate(axis=-1)([z_hair_avg, z_face_avg])
        z_log_var = Concatenate(axis=-1)([z_hair_log_var, z_face_log_var])

        z_p  = Input(shape=(self.z_dims,))
        z_f  = SampleNormal()([z_avg, z_log_var])

        z_p_face = Input(shape=(self.z_dims//2,))
        z_f_face = ExtractionLayer(self.z_dims*1//2, self.z_dims*2//2, (self.z_dims//2,))(z_f)

        z_p_hair = Input(shape=(self.z_dims//2,))
        z_f_hair = ExtractionLayer(self.z_dims*0//2, self.z_dims*1//2, (self.z_dims//2,))(z_f)

        x_p = self.f_dec([z_p, c])
        x_f = self.f_dec([z_f, c])

        x_p_face = self.f_face_dec([z_p_face, c_angle])
        x_f_face = self.f_face_dec([z_f_face, c_angle])
        x_p_hair = self.f_hair_dec([z_p_hair, c_angle])
        x_f_hair = self.f_hair_dec([z_f_hair, c_angle])

        x_r_concate_face = Concatenate(axis=-1)([x_r, x_r_face])
        x_p_concate_face = Concatenate(axis=-1)([x_p, x_p_face])
        x_f_concate_face = Concatenate(axis=-1)([x_f, x_f_face])

        x_r_concate_hair = Concatenate(axis=-1)([x_r, x_r_hair])
        x_p_concate_hair = Concatenate(axis=-1)([x_p, x_p_hair])
        x_f_concate_hair = Concatenate(axis=-1)([x_f, x_f_hair])

        y_r_face, f_D_x_r_face = self.f_face_dis(x_r_concate_face)
        y_f_face, f_D_x_f_face = self.f_face_dis(x_p_concate_face)
        y_p_face, f_D_x_p_face = self.f_face_dis(x_f_concate_face)

        y_r_hair, f_D_x_r_hair = self.f_hair_dis(x_r_concate_hair)
        y_f_hair, f_D_x_f_hair = self.f_hair_dis(x_p_concate_hair)
        y_p_hair, f_D_x_p_hair = self.f_hair_dis(x_f_concate_hair)

        d_loss_face = DiscriminatorLossLayer()([y_r_face, y_f_face, y_p_face])
        d_loss_hair = DiscriminatorLossLayer()([y_r_hair, y_f_hair, y_p_hair])

        c_r, f_C_x_r = self.f_cls(x_r)
        c_f, f_C_x_f = self.f_cls(x_f)
        c_p, f_C_x_p = self.f_cls(x_p)

        g_loss  = GeneratorLossLayer(metric=GENE_LOSS_NORM_P, w_C=1.0)([x_r, x_f, f_D_x_r_face, f_D_x_f_face, f_D_x_r_hair, f_D_x_f_hair, f_C_x_r, f_C_x_f])
        gd_loss_face = FeatureMatchingLayer()([f_D_x_r_face, f_D_x_p_face])
        gd_loss_hair = FeatureMatchingLayer()([f_D_x_r_hair, f_D_x_p_hair])
        gc_loss = FeatureMatchingLayer(lmbda=1.0)([f_C_x_r, f_C_x_p])

        c_loss = ClassifierLossLayer()([c, c_fake, c_r, c_f, c_p])

        # Build classifier trainer
        set_trainable(self.f_hair_enc, False)
        set_trainable(self.f_face_enc, False)
        set_trainable(self.f_dec,      False)
        set_trainable(self.f_face_dec, False)
        set_trainable(self.f_hair_dec, False)
        set_trainable(self.f_face_dis, False)
        set_trainable(self.f_hair_dis, False)
        set_trainable(self.f_cls,      True)

        self.cls_trainer = Model(inputs=[x_r, x_r_hair, x_r_face, c, c_angle, c_fake, z_p, z_p_face, z_p_hair],
                                 outputs=[c_loss],
                                 name='classifier')
        self.cls_trainer.compile(loss=[zero_loss],
                                 optimizer=Adam(lr=2.0e-5, beta_1=0.1),
                                 metrics=[classifier_accuracy(c, c_r)])
        self.cls_trainer.summary()

        # Build discriminator trainer
        set_trainable(self.f_hair_enc, False)
        set_trainable(self.f_face_enc, False)
        set_trainable(self.f_dec,      False)
        set_trainable(self.f_face_dec, False)
        set_trainable(self.f_hair_dec, False)
        set_trainable(self.f_face_dis, True )
        set_trainable(self.f_hair_dis, True )
        set_trainable(self.f_cls,      False)

        self.dis_trainer = Model(inputs=[x_r, x_r_hair, x_r_face, c, c_angle, z_p, z_p_face, z_p_hair],
                                 outputs=[d_loss_face, d_loss_hair],
                                 name='discriminator')
        self.dis_trainer.compile(loss=[zero_loss, zero_loss],
                                 optimizer=Adam(lr=2.0e-5, beta_1=0.1),
                                 metrics=[discriminator_accuracy(y_r_face, y_f_face, y_p_face)])
        self.dis_trainer.summary()

        # Build encoder trainer
        set_trainable(self.f_hair_enc, True )
        set_trainable(self.f_face_enc, True )
        set_trainable(self.f_dec,      False)
        set_trainable(self.f_face_dec, False)
        set_trainable(self.f_hair_dec, False)
        set_trainable(self.f_face_dis, False)
        set_trainable(self.f_hair_dis, False)
        set_trainable(self.f_cls,      False)

        self.enc_trainer = Model(inputs=[x_r, x_r_hair, x_r_face, c, c_angle, z_p, z_p_face, z_p_hair],
                                 outputs=[g_loss, kl_loss_hair, kl_loss_face],
                                 name='encoder')
        self.enc_trainer.compile(loss=[zero_loss, zero_loss, zero_loss],
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5))
        self.enc_trainer.summary()


        # Build generator
        set_trainable(self.f_hair_enc, False)
        set_trainable(self.f_face_enc, False)
        set_trainable(self.f_dec,      True )
        set_trainable(self.f_face_dec, True )
        set_trainable(self.f_hair_dec, True )
        set_trainable(self.f_face_dis, False)
        set_trainable(self.f_hair_dis, False)
        set_trainable(self.f_cls,      False)

        self.gen_trainer = Model(inputs=[x_r, x_r_hair, x_r_face, c, c_angle, z_p, z_p_face, z_p_hair],
                                 outputs=[g_loss, gd_loss_face, gd_loss_hair, gc_loss],
                                 name='generator')
        self.gen_trainer.compile(loss=[zero_loss] * 4,
                                 optimizer=Adam(lr=2.0e-4, beta_1=0.5),
                                 metrics=[generator_accuracy(y_f_face, y_p_face)])
        self.gen_trainer.summary()

        #Functions for editing applications
        self.f_rec = Model(inputs=[x_r_hair, x_r_face, c, c_angle],
                           outputs=[x_f],
                           name='f_rec')

        # Functions for editing applications
        self.f_ext_z_hair = Model(inputs=[x_r_hair, c],
                                  outputs=[z_hair_avg, z_hair_log_var],
                                  name=['f_ext_z_hair'])

        self.f_ext_z_face = Model(inputs=[x_r_face, c_angle],
                                  outputs=[z_face_avg, z_face_log_var],
                                  name=['f_ext_z_face'])

        # Store trainers
        self.store_to_save('cls_trainer')
        self.store_to_save('enc_trainer')
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')

    def build_encoder(self, output_dims):
        x_inputs = Input(shape=self.input_shape)
        c_inputs = Input(shape=(self.num_attrs,))

        c = Reshape((1, 1, self.num_attrs))(c_inputs)
        c = UpSampling2D(size=self.input_shape[:2])(c)
        x = Concatenate(axis=-1)([x_inputs, c])

        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(output_dims)(x)
        x = Activation('linear')(x)

        return Model([x_inputs, c_inputs], x)

    def build_decoder(self):
        z_inputs = Input(shape=(self.z_dims,))
        c_inputs = Input(shape=(self.num_attrs,))
        z = Concatenate()([z_inputs, c_inputs])

        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model([z_inputs, c_inputs], x)

    def build_half_decoder(self):
        z_inputs = Input(shape=(self.z_dims//2,))
        c_inputs = Input(shape=(self.num_attrs,))
        z = Concatenate()([z_inputs, c_inputs])

        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(z)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model([z_inputs, c_inputs], x)

    def build_discriminator(self):
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
