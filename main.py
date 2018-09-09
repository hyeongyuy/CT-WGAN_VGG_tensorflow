import argparse
import os
os.chdir(os.getcwd())
from wgan_vgg_model import wganVgg
os.chdir(os.getcwd() + '/..')
print('pwd : {}'.format(os.getcwd()))
import tensorflow as tf


parser = argparse.ArgumentParser(description = '')
#set load directory
parser.add_argument('--zi_path', dest = 'zi_path', default = '/data/train_input', help = 'LDCT image directory')
parser.add_argument('--xi_path', dest = 'xi_path', default = '/data/train_target', help = 'NDCT image directory')
parser.add_argument('--test_zi_path', dest = 'test_zi_path', default = '/data/test_input', help = 'test LDCT image directory')
parser.add_argument('--test_xi_path', dest = 'test_xi_path', default = '/data/test_target', help = 'test NDCT image directory')
parser.add_argument('--pretrained_vgg', dest = 'pretrained_vgg', default = './pretrained_vgg', help = 'test NDCT image directory')

#set save directory
parser.add_argument('--checkpoint_dir', dest = 'checkpoint_dir', default = 'checkpoint', help = 'check point dir')
parser.add_argument('--test_image_save_dir', dest = 'test_image_save_dir', default = './test/output_image', help = 'test iamge save dir')
parser.add_argument('--test_npy_save_dir', dest = 'test_npy_save_dir', default = './test/output_npy', help = 'test numpy file save dir')

#set train, test phase
parser.add_argument('--phase', dest = 'phase', default = 'train', help = 'train or test')
parser.add_argument('--continue_train', dest = 'continue_train', type = bool, default = True, help = 'load trained model')

#set model parameters
parser.add_argument('--lambda_', dest = 'lambda_', type = int, default = 10, help = 'Gradient penalty term weight')
parser.add_argument('--lambda_1', dest = 'lambda_1', type = float, default = 0.1, help = 'Perceptual loss weight (in WGAN_VGG network)')
#parser.add_argument('--lambda_2', dest = 'lambda_2', type = float, default = 0.1, help = 'MSE loss weight(in WGAN_VGG network)')
parser.add_argument('--beta1', dest = 'beta1', type = float, default = 0.5, help = 'Adam optimizer parameter')
parser.add_argument('--beta2', dest = 'beta2', type = float, default = 0.9, help = 'Adam optimizer parameter')
parser.add_argument('--alpha', dest = 'alpha', type = float, default = 1e-5, help = 'learning rate')
parser.add_argument('--num_iter', dest = 'num_iter', type = float, default = 100000, help = 'iterations')
parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 128, help = 'batch size')
parser.add_argument('--d_iters', dest = 'd_iters', type = int, default = 4, help = 'discriminator iteration') 

#set image parameters
parser.add_argument('--patch_size', dest = 'patch_size', type = int, default = 64, help = 'patch_size, h = w')
parser.add_argument('--whole_size', dest = 'whole_size', type = int, default = 512, help = 'whole_size, h = w')
parser.add_argument('--img_channel', dest = 'img_channel', type = int, default = 1, help = 'whole_size, 1')
parser.add_argument('--img_vmax', dest = 'img_vmax', type = int, default = 3072, help = 'image max value, 3072')
parser.add_argument('--img_vmin', dest = 'img_vmin', type = int, default = -1024, help = 'image max value -1024')

#set frequency
parser.add_argument('--print_freq', dest = 'print_freq', type = int, default = 100, help = 'print frequency')
parser.add_argument('--save_freq', dest = 'save_freq', type = int, default = 1000, help = 'checking frequency')


args = parser.parse_args()

gpu_options = tf.GPUOptions(allow_growth = True)


sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
model = wganVgg(sess, args)
model.train(args) if args.phase = = 'train' else model.test(args)
