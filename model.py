import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from GAN import GANTrainer, MultiGPUGANTrainer, SeparateGANTrainer, GANModelDesc
from utils import *

import tensorpack.tfutils.symbolic_functions as symbf

SHAPE = 128
BATCH = 16
TEST_BATCH = 32
NF = 64  # channel size

class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SHAPE, SHAPE, 3), 'inputA'),
                InputDesc(tf.float32, (None, SHAPE, SHAPE, 3), 'inputB')]


    @staticmethod
    def build_res_block(x, name, chan, first=False):
        with tf.variable_scope(name):
            input = x
            l = (LinearWrap(x)
                    .Conv2D('conv0', chan, stride=1, kernel_shape=3)
                    .Conv2D('conv1', chan, stride=1, kernel_shape=3)())
            l = (LinearWrap(tf.concat([l, input], axis=1))
                    .Conv2D('conv2', chan, stride=1, kernel_shape=3)())
            return l

    def generator(self, img):
        with argscope([Conv2D, Deconv2D],
                      nl=INLReLU, kernel_shape=4, stride=2), \
                argscope(Deconv2D, nl=INReLU):

            def res_group(input, name, depth, channels):
                l = input
                for k in range(depth):
                  l = Model.build_res_block(l, name + ('/res%d' % k), channels,
                          first=(k==0))
                return l

            subDepth = 3
            conv0 = Conv2D('conv0', img, NF, nl=LeakyReLU)
            conv1 = Conv2D('conv1', conv0, NF * 2)
            layer1 = res_group(conv1, 'layer1', subDepth, NF*2)
            conv2 = Conv2D('conv2', layer1, NF * 4)
            layer2 = res_group(conv2, 'layer2', subDepth, NF*4)
            conv3 = Conv2D('conv3', layer2, NF * 8)
            l = res_group(conv3, 'layer3', subDepth, NF*8)
            deconv0 = Deconv2D('deconv0', l, NF * 4)
            up1 = tf.concat([deconv0, layer2], axis=1)
            b_layer_2 = res_group(up1, 'blayer2', subDepth, NF * 4)
            deconv1 = Deconv2D('deconv1', b_layer_2, NF * 2)
            up2 = tf.concat([deconv1, layer1], axis=1)
            b_layer_1 = res_group(up2, 'blayer1', subDepth, NF * 2)
            deconv2 = Deconv2D('deconv2', b_layer_1, NF * 1)
            deconv3 = Deconv2D('deconv3', deconv2, 3, nl=tf.sigmoid)
        return deconv3

    def discriminator(self, img):
        with argscope(Conv2D, nl=INLReLU, kernel_shape=4, stride=2):
            l = Conv2D('conv0', img, NF*2, nl=LeakyReLU)
            relu1 = Conv2D('conv1', l, NF * 4)
            relu2 = Conv2D('conv2', relu1, NF * 8)

            relu3 = Conv2D('convf', relu2, NF*8, kernel_shape=3, stride=1)
            atrous = tf.contrib.layers.conv2d(relu3, NF*8, kernel_size=3,
                    data_format='NCHW', rate=2,
                    activation_fn=INLReLU, biases_initializer=None)
            atrous2 = tf.contrib.layers.conv2d(atrous, NF*8, kernel_size=3,
                    data_format='NCHW', rate=4,
                    activation_fn=INLReLU, biases_initializer=None)
            atrous3 = tf.contrib.layers.conv2d(atrous2, NF*8, kernel_size=3,
                    data_format='NCHW', rate=8,
                    activation_fn=INLReLU, biases_initializer=None)
            merge = tf.concat([relu3, atrous3], axis=1)
            clean = Conv2D('mConv', merge, NF*8, kernel_shape=3, stride=1)
            lsgan = Conv2D('lsconv', clean, 1, stride=1, nl=tf.identity,
                    use_bias=False)


        return lsgan, [relu1, relu2, relu3, atrous, atrous2, atrous3, clean]

    def get_feature_match_loss(self, feats_real, feats_fake):
        losses = []
        for real, fake in zip(feats_real, feats_fake):
            loss = tf.reduce_mean(tf.squared_difference(
                tf.reduce_mean(real, 0),
                tf.reduce_mean(fake, 0)),
                name='mse_feat_' + real.op.name)
            losses.append(loss)
        ret = tf.reduce_mean(losses, name='feature_match_loss')
        add_moving_summary(ret)
        return ret

    def loss_normalize(self, loss, update_condition, epsilon=1e-10):
        # Variable used for storing the scalar-value of the loss-function.
        loss_value = tf.Variable(1.0, name='loss_scalar_val_' + loss.op.name)

        loss_value_smooth = (tf.Variable(1.0, name='loss_smooth_' +
                loss.op.name))

        #TODO don't update if is_training
        ma_loss_value = (
            moving_averages.assign_moving_average(
                    loss_value_smooth, loss, 0.9999, zero_debias=False, name='loss_EMA'
                )
            )

        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ma_loss_value)
        # Expression used for either updating the scalar-value or
        # just re-using the old value.
        # Note that when loss_value.assign(loss) is evaluated, it
        # first evaluates the loss-function which is a TensorFlow
        # expression, and then assigns the resulting scalar-value to
        # the loss_value variable.
        loss_value_updated = tf.cond(update_condition,
                                     lambda: loss_value.assign(ma_loss_value),
                                     lambda: loss_value)


        # Expression for the normalized loss-function.
        loss_normalized = loss / (loss_value_updated + epsilon)

        add_moving_summary(tf.identity(loss_value, name='loss_scalar_' + loss.op.name))

        return loss_normalized

    def dragan_penalty(self, inputs):
        mean, var = tf.nn.moments(inputs,
                axes=range(1,inputs.get_shape().ndims), keep_dims=True)
        inputs_p = (inputs + 0.5 * tf.sqrt(var) *
        tf.random_uniform(shape=tf.shape(inputs), minval=-1., maxval=1.))

        alpha = tf.random_uniform(shape=[tf.shape(inputs)[0], 1, 1, 1], minval=0., maxval=1.)
        differences = inputs_p - inputs  # This is different from WGAN-GP
        interpolates = inputs + (alpha * differences)
        D_inter,_,_ = self.discriminator(interpolates)
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2,
        name='grad_penalty_red')
        return tf.identity(gradient_penalty * 0.25, name='grad_penalty')

    def _build_graph(self, inputs):
        A, B = inputs
        A = tf.transpose(A / 255.0, [0, 3, 1, 2])
        B = tf.transpose(B / 255.0, [0, 3, 1, 2])


        def viz3(name, a, b, c):
            im = tf.concat([a, b, c], axis=3)
            im = tf.transpose(im, [0, 2, 3, 1])
            im = (im) * 255
            im = tf.clip_by_value(im, 0, 255)
            im = tf.cast(im, tf.uint8, name='viz_' + name)
            tf.summary.image(name, im, max_outputs=50)


        # use the initializers from torch
        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.contrib.layers.variance_scaling_initializer(factor=0.333, uniform=True),
                      use_bias=False), \
                argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
                argscope([Conv2D, Deconv2D, BatchNorm, InstanceNorm], data_format='NCHW'), \
                argscope(LeakyReLU, alpha=0.2):
            with tf.variable_scope('gen'):
                with tf.variable_scope('B'):
                    AB = self.generator(A)
                with tf.variable_scope('A'):
                    BA = self.generator(B)
                with tf.variable_scope('A', reuse=True):
                    ABA = self.generator(AB)
                with tf.variable_scope('B', reuse=True):
                    BAB = self.generator(BA)

            viz3('A_recon', A, AB, ABA)
            viz3('B_recon', B, BA, BAB)

            with tf.variable_scope('discrim'):
                with tf.variable_scope('A'):
                    A_dis_real, A_feats_real = self.discriminator(A)
                with tf.variable_scope('A', reuse=True):
                    A_dis_fake, A_feats_fake = self.discriminator(BA)

                with tf.variable_scope('B'):
                    B_dis_real, B_feats_real = self.discriminator(B)
                with tf.variable_scope('B', reuse=True):
                    B_dis_fake, B_feats_fake = self.discriminator(AB)


        def LSGAN_losses(real, fake):
            with tf.name_scope('LSGAN_losses'):
                d_real = tf.reduce_mean(tf.squared_difference(real, 0.9), name='d_real')
                d_fake = tf.reduce_mean(tf.square(fake), name='d_fake')
                d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')

                g_loss = tf.reduce_mean(tf.squared_difference(fake, 0.9), name='g_loss')
                add_moving_summary(g_loss, d_loss)
                return g_loss, d_loss

        with tf.name_scope('LossA'):
            # reconstruction loss
            recon_loss_A = tf_dssim(A, ABA)
            recon_loss_A_l = tf.losses.absolute_difference(A,ABA,
                    reduction=tf.losses.Reduction.MEAN)

			# gan loss
            self.build_losses(A_dis_real, A_dis_fake)
            G_loss_A = self.g_loss
            D_loss_A = self.d_loss
            # feature matching loss
            fm_loss_A = self.get_feature_match_loss(A_feats_real, A_feats_fake)


        with tf.name_scope('LossB'):
            recon_loss_B = tf_dssim(B, BAB)

            recon_loss_B_l = tf.losses.absolute_difference(B, BAB,
                    reduction=tf.losses.Reduction.MEAN)


            self.build_losses(B_dis_real, B_dis_fake)
            G_loss_B = self.g_loss
            D_loss_B = self.d_loss# + grad_penalty_B
            fm_loss_B = self.get_feature_match_loss(B_feats_real, B_feats_fake)


        global_step = get_global_step_var()
        rate = tf.train.piecewise_constant(global_step, [np.int64(15000), np.int64(25000), np.int64(50000), np.int64(100000)], [0.01, 0.10, 0.15, 0.20, 0.25])
        rate = tf.identity(rate, name='rate')   # mitigate a TF bug
        loss_update = tf.logical_or(tf.equal(global_step, tf.constant(36,
            dtype=np.int64)), tf.equal(global_step % 90, tf.constant(0, dtype=np.int64)))
        rate = tf.constant(0.33, np.float32, name='static_rate')

        g_loss = tf.add_n([
            (self.loss_normalize(G_loss_A + G_loss_B, loss_update) * 0.7 +
            self.loss_normalize(fm_loss_A + fm_loss_B, loss_update) * 0.3) * (1 - rate),
            (self.loss_normalize((recon_loss_A + recon_loss_B), loss_update) *
                0.7 +
				self.loss_normalize((recon_loss_A_l + recon_loss_B_l),
                    loss_update) * 0.3) * rate], name='G_loss_total')
        d_loss = tf.add_n([D_loss_A, D_loss_B], name='D_loss_total')

        self.collect_variables('gen', 'discrim')

        self.g_loss = g_loss
        self.d_loss = d_loss

        add_moving_summary(recon_loss_A, recon_loss_B, rate, g_loss, d_loss,
                recon_loss_A_l, recon_loss_B_l)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5)
