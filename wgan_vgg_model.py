# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 16:39:45 2018

@author: yeohyeongyu
"""


"""
* data(in paper... )
    - [X]random extract patch (pairs):  
            train : 100,096 (from 4,000)
            test : 5,056 ( from 2,000)
      -> 따로 저장하지 않고, 모델에 들어가기 진적에 patch 형태로 변환
    - size : 64*64 (procedure에서는 80*80이라고 돼있음, 실험은  64*64로...)
    - [X]remove mostly air images 
      -> 처리 X(원본 이미지를 0~1사이 값으로 변환해서 입력으로...)

-------------------------------------------------
* training detail
 - mini batch size L : 128
 - opt : Adam(alpha = 1e-5, beta1 = 0.5, beta2 = 0.9)
 - discriminator iter : 4
 - lambda(WGAN weight penalty) : 10
 - lambda1(VGG weight) : 0.1
 - [X] epoch : 100
"""

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tcl
import numpy as np
import time
from glob import glob

import inout_util
import wgan_vgg_module as modules


class wganVgg(object):
    def __init__(self, sess, args):
        self.sess = sess    

        #### image placehold  (patch image, whole image)
        self.z_i= tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'LDCT')
        self.x_i = tf.placeholder(tf.float32, [None, args.patch_size, args.patch_size, args.img_channel], name = 'NDCT')
        self.whole_z = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_LDCT')
        self.whole_x = tf.placeholder(tf.float32, [1, args.whole_size, args.whole_size, args.img_channel], name = 'whole_NDCT')


        #### set modules (generator, discriminator, vgg net)
        self.g_net = modules.generator
        self.d_net = modules.discriminator
        self.vgg = modules.Vgg19(vgg_path = args.pretrained_vgg) 
        

        #### generate & discriminate
        #generated images
        self.G_zi = self.g_net(self.z_i, reuse = False)
        self.G_whole_zi = self.g_net(self.whole_z)

        #discriminate
        self.D_xi = self.d_net(self.x_i, reuse = False)
        self.D_G_zi= self.d_net(self.G_zi)

        #### variable list
        self.d_vars = [var for var in tf.global_variables() if 'discriminator' in var.name]
        self.g_vars = [var for var in tf.global_variables() if 'generator' in var.name]


        #### loss define
        #gradients penalty
        self.epsilon = tf.random_uniform([], 0.0, 1.0)
        self.x_hat = self.epsilon * self.x_i + (1 - self.epsilon) * self.G_zi
        self.D_x_hat = self.d_net(self.x_hat)
        self.grad_x_hat = tf.gradients(self.D_x_hat, self.x_hat)[0]
        self.grad_x_hat_l2 = tf.sqrt(tf.reduce_sum(tf.square(self.grad_x_hat), axis=1))
        self.gradient_penalty =  tf.square(self.grad_x_hat_l2 - 1.0)

        #perceptual loss
        self.G_zi_3c = tf.concat([self.G_zi]*3, axis=3)
        self.xi_3c = tf.concat([self.x_i]*3, axis=3)
        [w, h, d] = self.G_zi_3c.get_shape().as_list()[1:]
        self.vgg_perc_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((self.vgg.extract_feature(self.G_zi_3c) -  self.vgg.extract_feature(self.xi_3c))))) / (w*h*d))

        #discriminator loss(WGAN LOSS)
        self.D_loss =  tf.reduce_mean(self.D_G_zi) - tf.reduce_mean(self.D_xi) + args.lambda_ *tf.reduce_mean(self.gradient_penalty )
        #generator loss
        self.G_loss = args.lambda_1 * self.vgg_perc_loss - tf.reduce_mean(self.D_G_zi)


        #### summary
        #loss summary
        self.summary_vgg_perc_loss = tf.summary.scalar("PerceptualLoss_VGG", self.vgg_perc_loss)
        self.summary_d_loss = tf.summary.scalar("DiscriminatorLoss__WGAN", self.D_loss)
        self.summary_g_loss = tf.summary.scalar("GeneratorLoss", self.G_loss)
        self.summary_all_loss = tf.summary.merge([self.summary_vgg_perc_loss, self.summary_d_loss, self.summary_g_loss])

        #image summary
        self.patch_img_summary = tf.concat([self.z_i, self.x_i, self.G_zi], axis = 2)
        self.whole_img_summary = tf.concat([self.whole_z, self.whole_x, self.G_whole_zi], axis = 2)
        self.summary_image1 = tf.summary.image('reulst', self.patch_img_summary)
        self.summary_image2 = tf.summary.image('reulst', self.whole_img_summary, max_outputs=2)


        #### optimizer
        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.D_loss, var_list = self.d_vars)
            self.g_adam = tf.train.AdamOptimizer(learning_rate= args.alpha, beta1 = args.beta1, beta2 = args.beta2).minimize(self.G_loss, var_list = self.g_vars)
                
            

        #model saver
        self.saver = tf.train.Saver(max_to_keep=None)    

    def train(self, args):
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)


        self.start_step = 0
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                

        #image load
        img_loader = inout_util.DataLoader(args.zi_path, args.xi_path, p_h = args.patch_size, p_w = args.patch_size, image_max = args.img_vmax, image_min = args.img_vmin)
        zi_img, xi_img = img_loader(shuffle=False)
        batch_zi, batch_xi = img_loader.generate_image_batch(zi_img, xi_img, args.batch_size, shuffle = True)
        #batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([batchA, batchB], capacity=2 * args.num_gpu)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess = self.sess)

        start_time = time.time()
        for t in range(self.start_step, args.num_iter):
            p_time = 0
            for _ in range(0, args.d_iters):
                #patch sampling
                p_bz, p_bx = self.sess.run([batch_zi, batch_xi])
                
                #discriminator update
                self.sess.run(self.d_adam, feed_dict={self.z_i : p_bz, self.x_i : p_bx})
                
            #patch sampling    
            p_bz, p_bx = self.sess.run([batch_zi, batch_xi])            
            
            #generator update & loss summary
            _, summary_str= self.sess.run([self.g_adam, self.summary_all_loss], feed_dict={self.z_i : p_bz, self.x_i : p_bx})
            self.writer.add_summary(summary_str, t)
                
             
            #print point
            if (t-1) % args.print_freq == 0:
                #print loss & time
                d_loss, g_loss, g_zi_img = self.sess.run([self.D_loss, self.G_loss, self.G_zi], feed_dict={self.z_i : p_bz, self.x_i : p_bx})
                                
                print('Iter {} Time {} d_loss {} g_loss {}'.format(t, time.time() - start_time, d_loss, g_loss))


                summary_str= self.sess.run(self.summary_image1, \
                                         feed_dict={self.z_i : p_bz[0].reshape([1] + self.z_i.get_shape().as_list()[1:]), \
                                                    self.x_i : p_bx[0].reshape([1] + self.x_i.get_shape().as_list()[1:]), \
                                                    self.G_zi : g_zi_img[0].reshape([1] + self.G_zi.get_shape().as_list()[1:])
                                                    })
                self.writer.add_summary(summary_str, t)



                #summary image
                zi_files_dir= sorted(glob(args.test_zi_path + '/*.*'))
                xi_files_dir= sorted(glob(args.test_xi_path + '/*.*'))
                
                sltd_idx = np.random.choice(range(len(zi_files_dir)))
                test_zi, test_xi = inout_util.load_test_image(zi_files_dir[sltd_idx], xi_files_dir[sltd_idx],  image_max = args.img_vmax, image_min = args.img_vmin)
                whole_G_zi = self.sess.run(self.G_whole_zi, feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})
                
                summary_str= self.sess.run(self.summary_image2, \
                                         feed_dict={self.whole_z : test_zi.reshape(self.whole_z.get_shape().as_list()), \
                                                    self.whole_x : test_xi.reshape(self.whole_x.get_shape().as_list()), \
                                                    self.G_whole_zi : whole_G_zi.reshape(self.G_whole_zi.get_shape().as_list()),
                                                    })
                self.writer.add_summary(summary_str, t)
            
            if (t-1) % args.save_freq == 0:
                self.save(t, args.checkpoint_dir)


    def test(self, args):
        self.sess.run(tf.global_variables_initializer())

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ## mk save dir (image & numpy file)    
        img_save_dir = os.path.join('.', args.test_image_save_dir)
        npy_save_dir = os.path.join('.', args.test_npy_save_dir)

        if not os.path.exists(img_save_dir):
            os.makedirs(img_save_dir)

        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)


        ## test
        start_time = time.time()
        zi_files_dir= sorted(glob(args.test_zi_path + '/*.*'))
        xi_files_dir= sorted(glob(args.test_xi_path + '/*.*'))

        print('test data dir \nLDCT : {}\nNDCT : {}'.format(args.test_zi_path,  args.test_xi_path))
        print(len(zi_files_dir), len(xi_files_dir))

        for idx in range(len(zi_files_dir)):
            test_zi, test_xi = inout_util.load_test_image(zi_files_dir[idx], xi_files_dir[idx], \
                image_max = args.img_vmax, image_min = args.img_vmin)
            whole_G_zi = self.sess.run(self.G_whole_zi, feed_dict={self.whole_z: test_zi.reshape(self.whole_z.get_shape().as_list())})
            save_file_nm = 'Gen_from_' + zi_files_dir[idx].split('/')[-1]
            
 
            #image save
            inout_util.save_image(test_zi.reshape(args.whole_size, args.whole_size), test_xi.reshape(args.whole_size, args.whole_size), \
                whole_G_zi.reshape(args.whole_size, args.whole_size), os.path.join(img_save_dir, save_file_nm.replace('npy', '')))
            np.save(os.path.join(npy_save_dir, save_file_nm), whole_G_zi)


                
    def save(self, step, checkpoint_dir = 'checkpoint'):
        model_name = "wgan_vgg.model"
        checkpoint_dir = os.path.join('.', checkpoint_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir = 'checkpoint'):
        print(" [*] Reading checkpoint...")
        checkpoint_dir = os.path.join('.', checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            
            self.start_step = int(ckpt_name.split('-')[-1])
            
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(self.start_step)
            return True
        else:
            return False

