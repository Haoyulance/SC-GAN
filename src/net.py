
import tensorflow as tf
from utils.discriminator import Discriminator
from utils.generator import Generator
from utils.util import *


class SCGAN(object):
    def __init__(self, width, height, depth, ichan, ochan, g_reg=0.001, d_reg=0.001, l1_weight=200, B_weight=200, lr=0.0002, beta1=0.5, floss=False, reg=False, attn=True):
        """

        :param width: image width in pixel
        :param height: image height in pixel
        :param ichan: number of channels used by input images
        :param ochan: number of channels used by output images
        :param l1_weight: l1 loss weight
        :param lr: learning rate for adam FGHJ
        :param beta1: beta1 parameter for adam
        """
        self._is_training = tf.placeholder(tf.bool, name='is_train_holder')
        self._g_inputs = tf.placeholder(tf.float32, [None, width, height, depth, ichan], name='input_holder')
        self._d_inputs_a = tf.placeholder(tf.float32, [None, width, height, depth, ichan])
        self._d_inputs_b = tf.placeholder(tf.float32, [None, width, height, depth,  ochan])
        self._g = Generator(self._g_inputs, self._is_training, ochan, attn=attn)
        self.lr = lr

        self._real_d_2 = Discriminator('Discriminator_2', tf.concat([self._d_inputs_a, self._d_inputs_b], axis=4),
                                       self._is_training, attn=True)
        self._fake_d_2 = Discriminator('Discriminator_2', tf.concat([self._d_inputs_a, self._g._decoder['final']['fmap']], axis=4), self._is_training, reuse=True, attn=True)

        self._f_matching_loss_2 = tf.reduce_mean([tf.reduce_mean(tf.abs(self._real_d_2._perceptual_fmap[i] - self._fake_d_2._perceptual_fmap[i]))
                                 for i in range(len(self._real_d_2._perceptual_fmap))])

        self._pure_g_loss = -tf.reduce_mean(tf.log(self._fake_d_2._discriminator['l5']['fmap']))

        rmse_mask = tf.not_equal(self._d_inputs_b, -1)
        mask = tf.dtypes.cast(rmse_mask, tf.float32)
        masked_input = tf.math.multiply(self._d_inputs_b, mask)
        masked_output = tf.math.multiply(self._g._decoder['final']['fmap'], mask)

        self.psnr_score = tf.reduce_mean(tf.image.psnr(self._d_inputs_b, self._g._decoder['final']['fmap'], max_val=2.0))

        self.ssim_score = tf.reduce_mean(tf.image.ssim(masked_input, masked_output, max_val=2.0))
        self.rmse_score = tf.math.sqrt(tf.reduce_mean(tf.math.squared_difference(tf.boolean_mask(self._d_inputs_b, rmse_mask),
                                                      tf.boolean_mask(self._g._decoder['final']['fmap'], rmse_mask))))

        self._g_loss = self._pure_g_loss + l1_weight * (tf.reduce_mean(
            tf.abs(self._d_inputs_b - self._g._decoder['final']['fmap']))) + B_weight * self.rmse_score

        self._d_loss_2 = -tf.reduce_mean(
            tf.log(self._real_d_2._discriminator['l5']['fmap'] + tf.keras.backend.epsilon()) +
            tf.log(1.0 - self._fake_d_2._discriminator['l5']['fmap'] + tf.keras.backend.epsilon()))
        self._d_loss = self._d_loss_2

        if floss:
            self._f_matching_loss_2 = tf.cond(self._f_matching_loss_2 < 2e-4, lambda:0.0, lambda:self._f_matching_loss_2)
            self._g_loss += 20 * self._f_matching_loss_2
            self.feature_matching_loss_2 = tf.summary.scalar("feature_matching_loss_2", self._f_matching_loss_2)

        if reg:
            vars = tf.trainable_variables(scope='Generator')
            self.l2_reg_g = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * g_reg
            self.l2_reg_g = tf.cond(self.l2_reg_g < 1e-5, lambda:0.0, lambda:self.l2_reg_g)
            self._g_loss += self.l2_reg_g
            tf.summary.scalar("generator_l2", self.l2_reg_g)

            vars = tf.trainable_variables(scope='Discriminator_2')
            self.l2_reg_d = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * d_reg
            self.l2_reg_d = tf.cond(self.l2_reg_d < 5e-5, lambda:0.0, lambda:self.l2_reg_d)
            self._d_loss += self.l2_reg_d
            tf.summary.scalar("discriminator_l2", self.l2_reg_d)




        tf.summary.scalar("pure generator_loss", self._pure_g_loss)
        tf.summary.scalar("generator_loss", self._g_loss)
        tf.summary.scalar('psnr score', self.psnr_score)
        tf.summary.scalar('ssim_score', self.ssim_score)
        tf.summary.scalar('rmse_score', self.rmse_score/2)
        tf.summary.scalar("discriminator_loss", -self._d_loss)



        self.l_rate = tf.placeholder(tf.float32, shape=None)

        if attn:
            tf.summary.scalar("decoder_sigma", self._g.sigma_collection['decoder_sigma'])
            tf.summary.scalar("encoder_sigma", self._g.sigma_collection['encoder_sigma'])
            tf.summary.scalar("disc_fake_sigma", self._fake_d_2.sigma_collection['disc_sigma'])
        tf.summary.scalar("learning rate", self.l_rate)
        self.summary_merge = tf.summary.merge_all()

        # for test metrices
        self.test_psnr = tf.placeholder(tf.float32, shape=None)
        self.test_ssim = tf.placeholder(tf.float32, shape=None)
        self.test_rmse = tf.placeholder(tf.float32, shape=None)

        self.psnr = tf.summary.scalar('psnr test', self.test_psnr)
        self.ssim = tf.summary.scalar('ssim test', self.test_ssim)
        self.rmse = tf.summary.scalar('rmse test', self.test_rmse)

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
        with tf.control_dependencies(g_update_ops):
            self._g_train_step = tf.train.AdamOptimizer(self.lr, beta1=beta1)\
                                    .minimize(self._g_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))



        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator_2')
        with tf.control_dependencies(d_update_ops):
            self._d_train_step_1 = tf.train.AdamOptimizer(self.lr, beta1=beta1)\
                                    .minimize(self._d_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator_2'))


    def train_step(self, sess, g_inputs, d_inputs_a, d_inputs_b, is_training=True, train_d=1):

        _, gloss_curr = sess.run([self._g_train_step, self._g_loss],
                                 feed_dict={self._d_inputs_a: d_inputs_a, self._d_inputs_b: d_inputs_b,
                                            self._g_inputs: g_inputs, self._is_training: is_training})

        if train_d == 1:

            _, dloss_curr = sess.run([self._d_train_step_1, self._d_loss],
                                     feed_dict={self._d_inputs_a: d_inputs_a, self._d_inputs_b: d_inputs_b,
                                                self._g_inputs: g_inputs, self._is_training: is_training})
            summart_temp = sess.run(self.summary_merge, feed_dict={self._d_inputs_a: d_inputs_a, self._d_inputs_b: d_inputs_b,
                                            self._g_inputs: g_inputs, self._is_training: is_training, self.l_rate:self.lr})
            return gloss_curr, dloss_curr, summart_temp

        summart_temp = sess.run(self.summary_merge, feed_dict={self._d_inputs_a: d_inputs_a, self._d_inputs_b: d_inputs_b,
                                            self._g_inputs: g_inputs, self._is_training: is_training, self.l_rate:self.lr})

        return gloss_curr, _, summart_temp


    def sample_generator(self, sess, g_inputs, is_training=False):

        generated = sess.run(self._g._decoder['final']['fmap'],
                            feed_dict={self._g_inputs: g_inputs, self._is_training: is_training})

        return generated
