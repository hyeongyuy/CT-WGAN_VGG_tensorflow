# WGAN_VGG_tensorflow
Low Dose CT Image Denoising Using a Generative Adversarial Network with Wasserstein Distance and Perceptual Loss

## run..
train)<br>
$python main.py --zi_path=/data/train_input --xi_path=/data/train_target --pretrained_vgg=./pretrained_vgg<br>
test)<br>
$python main.py  --test_zi_path=/data/test_input --test_xi_path=/data/test_target --pretrained_vgg=./pretrained_vgg --phase=test

## source
vgg : https://github.com/machrisaa/tensorflow-vgg<br>
WGAN : https://github.com/jiamings/wgan
