

from .util import *

class Discriminator(object):
    def __init__(self, name, inputs, is_training, stddev=0.02, center=True, scale=True, reuse=None, attn=True):
        self._is_training = is_training
        self._stddev = stddev
        self._perceptual_fmap = []
        self.sigma_collection = dict()
        self.attn = attn

        with tf.variable_scope(name, initializer=tf.truncated_normal_initializer(stddev=self._stddev), reuse=reuse):
            self._center = center
            self._scale = scale
            self._prob = 0.5
            self._inputs = inputs
            self._discriminator = self._build_discriminator(inputs, reuse=reuse)


    def _build_layer(self, name, inputs, k, bn=True, use_dropout=False):
        layer = dict()
        with tf.variable_scope(name):
            layer['conv'] = snconv3d(inputs, size=5, stride=2, output_dim=k)
            layer['bn'] = batch_norm(layer['conv'], center=self._center, scale=self._scale, training=self._is_training)\
                          if bn else layer['conv']
            layer['dropout'] = tf.nn.dropout(layer['bn'], self._prob) if use_dropout else layer['bn']
            layer['fmap'] = lkrelu(layer['dropout'], slope=0.2)
        return layer

    def _build_discriminator(self, inputs, reuse=None):
        discriminator = dict()
        discriminator['l1'] = self._build_layer('l1', inputs, 64, bn=False)
        self._perceptual_fmap.append(discriminator['l1']['fmap'])
        discriminator['l2'] = self._build_layer('l2', discriminator['l1']['fmap'], 128)
        self._perceptual_fmap.append(discriminator['l2']['fmap'])
        discriminator['l3'] = self._build_layer('l3', discriminator['l2']['fmap'], 256)
        self._perceptual_fmap.append(discriminator['l3']['fmap'])
        if self.attn:
            discriminator['attention'], sigma = sn_attention('attention', discriminator['l3']['fmap'])
            self._perceptual_fmap.append(discriminator['attention'])
            self.sigma_collection['disc_sigma'] = sigma
        else:
            discriminator['attention'] = discriminator['l3']['fmap']

        discriminator['l4'] = self._build_layer('l4', discriminator['attention'], 512)
        self._perceptual_fmap.append(discriminator['l4']['fmap'])



        with tf.variable_scope('l5'):
            l5 = dict()
            l5['conv'] = snconv3d(discriminator['l4']['fmap'], size=5, stride=1, output_dim=1)
            l5['bn'] = batch_norm(l5['conv'], center=self._center, scale=self._scale, training=self._is_training)
            l5['fmap'] = tf.nn.sigmoid(l5['bn'])
            discriminator['l5'] = l5

        return discriminator
