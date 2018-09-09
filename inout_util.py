# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 10:45:16 2018

@author: yeohyeongyu
"""
import os
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class DataLoader(object):
    def __init__(self, img_A_path, img_B_path, height = 512, width = 512, p_h = 64, p_w = 64,  depth = 1, image_max = 3072, image_min = -1024):
        #self.data_dir = data_dir
        self.img_A_path = img_A_path
        self.img_B_path = img_B_path
        #capacity...
        self.min_fraction_of_examples_in_queue = 1
        
        #image params
        self.image_min = image_min
        self.image_max = image_max 
        
        self.patch_h = p_h
        self.patch_w = p_w
        self.height = height
        self.width = width
        self.depth = depth
        self.image_byte_size = self.height * self.width * self.depth *2 + 64*2  #int16 -> 2byte -> *2 & header?? 
        
    #make file name queue
    def __call__(self, shuffle = False):
                
        imgA_files_dir= sorted(glob(self.img_A_path + '/*.*'))
        imgB_files_dir= sorted(glob(self.img_B_path + '/*.*'))
        
        self.num_files_per_epoch = len(imgA_files_dir)
        self.min_queue_examples =  int(self.min_fraction_of_examples_in_queue * self.num_files_per_epoch )
        
        
        imgA_filename_queue = tf.train.string_input_producer(imgA_files_dir, shuffle = shuffle)
        imgB_filename_queue = tf.train.string_input_producer(imgB_files_dir, shuffle = shuffle)
        
        [imgA_dequeue, imgB_dequeue] = self.dequeue_image([imgA_filename_queue, imgB_filename_queue])#, self.dequeue_image(imgB_filename_queue )
        image_pair = tf.concat([imgA_dequeue, imgB_dequeue], axis=2)
        crop_img = tf.random_crop(image_pair, [self.patch_h, self.patch_w, self.depth * 2])
        patch_imgA, patch_imgB  = crop_img[:,:, :self.depth],  crop_img[:,:, self.depth:]
        return patch_imgA, patch_imgB

    def dequeue_image(self, file_queues):
        # define reader
        reader = tf.FixedLengthRecordReader(record_bytes=self.image_byte_size)
        reshaped_images = []
        for file_queue in file_queues:
            key,value = reader.read(file_queue)
            
            #define decoder
            image_bytes = tf.decode_raw(value, tf.int16)
            image_bytes_ = tf.transpose(tf.reshape(tf.strided_slice(image_bytes, [64], [64 + self.image_byte_size]), [self.depth, self.height, self.width]),  [1, 2, 0])
            reshaped_image = tf.cast(image_bytes_, tf.float32)
            reshaped_image = (reshaped_image - self.image_min )/(self.image_max - self.image_min) # 0 ~ 1
            reshaped_images.append(reshaped_image)
        return reshaped_images

    def generate_image_batch(self, A_img_queue, B_img_queue, batch_size, shuffle):
        num_preprocess_threads = 16
        if shuffle:
            batch_imagesA, batch_imagesB = tf.train.shuffle_batch(
                [A_img_queue, B_img_queue],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size,
                min_after_dequeue=self.min_queue_examples)
        else:
            batch_imagesA, batch_imagesB = tf.train.batch(
                [A_img_queue, B_img_queue],
                batch_size=batch_size,
                num_threads=num_preprocess_threads,
                capacity=self.min_queue_examples + 3 * batch_size)
            
        return  batch_imagesA, batch_imagesB  




def load_test_image(img_A_path, img_B_path, image_max = 3072, image_min = -1024):
    img_A = np.load(img_A_path).astype(np.float)
    img_B = np.load(img_B_path).astype(np.float)

    if image_max == None:
        image_max = np.max(img_A.reshape(-1))

    img_A = (img_A - image_min )/(image_max - image_min)
    img_B = (img_B - image_min )/(image_max - image_min)
       
    img_A = np.expand_dims(img_A, axis = 2)
    img_B = np.expand_dims(img_B, axis = 2)

    #img_AB = np.concatenate((img_A, img_B), axis=2)
    
    return img_A, img_B




def get_random_patch(image1, image2, patch_size = [64,64], overap = False):
    h, w = patch_size 
    if overap:
        size =  [int(image1.shape[0] - h/2), int(image1.shape[1] - w/2)]
    else:    
        size = [int(image1.shape[0] / h), int(image1.shape[1] / w)]
        
    sltd_idx = np.random.choice(range(size[0] * size[1]))
    i = sltd_idx % size[0]
    j = sltd_idx // size[1]
    
    return image1[j*h:j*h+h, i*w:i*w+w, :], image2[j*h:j*h+h, i*w:i*w+w, :]



def merge(images, whole_size = 512):
    h, w = images.shape[1], images.shape[2]
    size =  [int(whole_size / h), int(whole_size / w)]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def splitimg(whole_image, patch_size = [64, 64]):
    h, w = patch_size 
    size = [int(whole_image.shape[0] / h), int(whole_image.shape[1] / w)]

    patches = []
    for idx in range(size[0] * size [1]):
        i = idx % size[0]
        j = idx // size[1]
        patch = whole_image[j*h:j*h+h, i*w:i*w+w, :]
        patches.append(patch  )
    return np.array(patches)
   


#save mk img
def save_image(LDCT, NDCT, output_, save_dir = '.',  max_ = 1, min_= 0): 
    f, axes  = plt.subplots(2, 3, figsize=(30, 20))
    
    axes[0,0].imshow(LDCT,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[0,1].imshow(NDCT,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[0,2].imshow(output_,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    
    axes[1,0].imshow(NDCT.astype(np.float32) - LDCT.astype(np.float32),  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[1,1].imshow(NDCT - output_,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    axes[1,2].imshow(output_ - LDCT,  cmap=plt.cm.gray, vmax = max_, vmin = min_)
    
    axes[0,0].title.set_text('LDCT image')
    axes[0,1].title.set_text('NDCT image')
    axes[0,2].title.set_text('output image')
    
    axes[1,0].title.set_text('NDCT - LDCT  image')
    axes[1,1].title.set_text('NDCT - outupt image')
    axes[1,2].title.set_text('output - LDCT  image')
    if save_dir != '.':
        f.savefig(save_dir)
        plt.close()   
