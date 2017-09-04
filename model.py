import tensorflow as tf
import numpy as np
import time, os
import glob
from operations import *
from utils import *

class DCGAN():
    def __init__(self, args, sess):
        self.args = args
        self.sess = sess

        self.build_model()
        self.args.is_training = True

    def build_model(self):
        self.inputs = tf.placeholder(tf.float32, [self.args.batch_size] + [self.args.target_size, self.args.target_size, self.args.channel_dim], name='inputs')
        self.z = tf.placeholder(tf.float32, [None, self.args.z_dim], name='z')
        self.z_summary = histogram_summary('z', self.z)

        self.g = self.generator(self.z)
        self.g_summary = image_summary('G', self.g)
        self.d_real, self.d_real_logits = self.discriminator(self.inputs)
        self.d_fake, self.d_fake_logits = self.discriminator(self.g, reuse=True)
        self.generated_sample = self.generator(self.z, reuse=True, sampling=True)

        # Define loss functino : cross entropy
        self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_real), logits=self.d_real_logits))
        self.d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.d_fake), logits=self.d_fake_logits))
        self.d_loss = self.d_real_loss + self.d_fake_loss
        self.d_loss_summary = scalar_summary('d_loss', self.d_loss)
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.d_fake), logits=self.d_fake_logits))
        self.g_loss_summary = scalar_summary('g_loss', self.g_loss)

        # Get trainable variables
        self.trvb = tf.trainable_variables()
        for i in self.trvb:
            print(i.op.name)

        self.g_vars = [v for v in self.trvb if v.name.startswith('generator/')]
        self.d_vars = [v for v in self.trvb if v.name.startswith('discriminator/')]

        self.d_optimizer = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optimizer = tf.train.AdamOptimizer(self.args.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()


    # z is prior noise
    def generator(self, z, reuse=False, sampling=False):
        with tf.variable_scope('generator') as scope:
            if reuse:
                scope.reuse_variables()
            if sampling:
                self.args.is_training = False

            fourth_height, fourth_width = int(self.args.target_size/2), int(self.args.target_size/2) # 32,32
            third_height, third_width = int(self.args.target_size/4), int(self.args.target_size/4) # 16,16
            second_height, second_width = int(self.args.target_size/8), int(self.args.target_size/8) #8,8
            first_height, first_width = int(self.args.target_size/16), int(self.args.target_size/16) #4,4
            # self.args.final_dim=128
            self.z_flattened = linear(z, first_height*first_width*(self.args.final_dim*8), name='g_linear')

            self.deconv1 = tf.reshape(self.z_flattened, [self.args.batch_size, first_height, first_width, self.args.final_dim*8]) # [batch, 4, 4, 1024]
            self.deconvolution1 = deconv2d(self.deconv1, [self.args.batch_size, second_height, second_width, self.args.final_dim*4], name='deconv1') # [batch, 8,8,512]
            self.deconvolution1_batch = batchnorm_wrapper(self.deconvolution1, self.args.is_training, name='deconv1_bn')

            self.deconv2 = tf.nn.relu(self.deconvolution1_batch)
            self.deconvolution2 = deconv2d(self.deconv2, [self.args.batch_size, third_height, third_width, self.args.final_dim*2], name='deconv2') # [batch, 16,16, 256]
            self.deconvolution2_batch = batchnorm_wrapper(self.deconvolution2, self.args.is_training, name='deconv2_bn')

            self.deconv3 = tf.nn.relu(self.deconvolution2_batch)
            self.deconvolution3 = deconv2d(self.deconv3, [self.args.batch_size, fourth_height, fourth_width, self.args.final_dim], name='deconv3') # [batch, 32,32,128]
            self.deconvolution3_batch = batchnorm_wrapper(self.deconvolution3, self.args.is_training, name='deconv3_bn')

            self.deconv4 = tf.nn.relu(self.deconvolution3_batch)
            self.deconvolution4 = deconv2d(self.deconv4, [self.args.batch_size, self.args.target_size, self.args.target_size, self.args.channel_dim], name='deconv4') # [batch, 64, 64, 3]
            return tf.nn.tanh(self.deconvolution4)

    # real_fake_image : [batch, 64,64,3]
    def discriminator(self, real_fake_image, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            self.conv1 = conv2d(real_fake_image, self.args.final_dim, name='conv1') # [batch, 32,32, 128]
            self.conv1_batch = batchnorm_wrapper(self.conv1, self.args.is_training, name='conv1_bn')

            self.conv2 = conv2d(lrelu(self.conv1_batch), self.args.final_dim*2, name='conv2') #[batch, 16,16,256]
            self.conv2_batch = batchnorm_wrapper(self.conv2, self.args.is_training, name='conv2_bn')

            self.conv3 = conv2d(lrelu(self.conv2_batch), self.args.final_dim*4, name='conv3') # [batch, 8,8,512]
            self.conv3_batch = batchnorm_wrapper(self.conv3, self.args.is_training, name='conv3_bn')

            self.conv4 = conv2d(lrelu(self.conv3_batch), self.args.final_dim*8, name='conv4') # [batch, 4, 4, 1024]
            self.disc_flattened = tf.reshape(self.conv4, [self.args.batch_size, -1])
            self.prob_logits = linear(self.disc_flattened, 1, name='d_linear')

            return tf.nn.sigmoid(self.prob_logits), self.prob_logits


    def train(self):
        self.sess.run(tf.global_variables_initializer())

        # Merging summary
        self.d_sum = merge_summary([self.z_summary, self.d_loss_summary])
        self.g_sum = merge_summary([self.z_summary, self.g_summary, self.g_loss_summary])
        self.writer = summary_writer('./tensorboard_log', self.sess.graph)

        self.train_count = 0
        start_time = time.time()
        # Globbing all files
        self.real_datas = glob.glob(os.path.join(self.args.data_path,'*.jpg'))  # Return png file as list
        self.sample_z = np.random.uniform(-1,1, [self.args.showing_height*self.args.showing_width, self.args.z_dim])

        if self.load():
            print('Checkpoint loaded')
        else:
            print('Checkpoint load failed')

        for epoch in range(self.args.num_epoch):
            print('Epoch %d' % (epoch+1))
            shuffle_index = np.random.permutation(len(self.real_datas))[:self.args.partition_index]
            batches = np.asarray(self.real_datas)[shuffle_index]
            trainingsteps_per_batch = self.args.partition_index // (self.args.batch_size)
            for index in range(0, trainingsteps_per_batch):
                self.train_count += 1
                batch_inputs = batches[index*self.args.batch_size : (index+1)*self.args.batch_size]
                batch = [get_image(batch_input, self.args.input_size, self.args.target_size) for batch_input in batch_inputs]
                batch_images = np.asarray(batch).astype(np.float32)

                batch_z = np.random.uniform(-1,1, [self.args.batch_size, self.args.z_dim])

                _, d_r_loss, d_f_loss, summary = self.sess.run([self.d_optimizer, self.d_real_loss, self.d_fake_loss, self.d_sum], feed_dict={self.z : batch_z, self.inputs : batch_images})
                self.writer.add_summary(summary, self.train_count)
                _ = self.sess.run([self.g_optimizer], feed_dict={self.z : batch_z})
                # Run g twice to make sure d_loss does not go to zero
                _, gloss, summary = self.sess.run([self.g_optimizer, self.g_loss, self.g_sum], feed_dict={self.z : batch_z})
                self.writer.add_summary(summary, self.train_count)

                print('Step %d in Epoch %d, D real loss : %3.3f, D fake loss :  %3.3f, D total loss : %3.3f, G loss : %3.3f, duration time : %3.3f' % (index+1, epoch+1, d_r_loss, d_f_loss,d_r_loss + d_f_loss, gloss, time.time()-start_time))

            # Every 5 epoch, test and save
            if np.mod(epoch+1, self.args.save_interval) == 0:
                G_sample = self.sess.run(self.generated_sample, feed_dict={self.z:self.sample_z})
                save_image(G_sample, [self.args.showing_height,self.args.showing_width], './{}/train_{:2d}epochs.png'.format(self.args.sample_dir, epoch+1))
                self.save(self.train_count)
                self.args.is_training = True

    def generator_test(self):
        self.load()
        z_test = np.random.uniform(-1, 1, [self.args.showing_height * self.args.showing_width, self.args.z_dim])
        generated = self.sess.run([self.generated_sample], feed_dict={self.z:z_test})
        save_image(generated, [self.args.showing_height, self.args.showing_width], './{}/test'.format(self.args.sample_dir))

    @property
    def model_dir(self):
        return '{}_batchsize_{}_z_dim_for_{}'.format(self.args.batch_size, self.args.z_dim, 'CelebA')

    def save(self, global_step):
        model_name = 'DCGAN'
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=global_step)
        print('Checkpoint saved at %d steps' % (global_step))

    def load(self):
        checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        # Checkpoint prototype
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.train_count = int(ckpt_name.split('-')[-1])  # Get trained steps count
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print('Checkpoint loaded at %d steps' % (self.train_count))
            return True
        else:
            self.train_count = 0
            return False
