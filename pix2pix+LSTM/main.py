import argparse
import numpy as np
import pandas as pd
from models import *
from datasets import *
import datetime as dt
import sys
import os
import matplotlib.pyplot as plt
import imageio, cv2

from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="sample", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate in Generator (UNet)")
parser.add_argument("--sample_epoch_interval", type=int, default=100, help="epoch interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

if opt.batch_size != 1:
    print('Error: batch_size must be equal to 1')
    sys.exit()

os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

filenames = glob('data/{}/{}/{}/*.jpg'.format(opt.dataset_name, 'hig_reso', '00'))
str_len = len('data/{}/{}/{}/'.format(opt.dataset_name, 'hig_reso', '00'))
img_list = [filename[str_len:] for filename in filenames]
img_list = list(sorted(img_list))
filename = 'data/{}/{}/{}/{}'.format(opt.dataset_name, 'hig_reso', '00', img_list[0])
image = imageio.imread(filename, as_gray=False, pilmode='RGB').astype(np.float)
height, width, _ = image.shape

data_loader = DataLoader(dataset_name=opt.dataset_name, height=height, width=width)
img_list = data_loader.get_img_list()
frames = len(img_list)
height, width, in_channels = data_loader.get_img_shape(img_list[0])

patch_height = height // 2 ** 4
patch_width = width // 2 ** 4
disc_patch = (patch_height, patch_width, 1)

out_channels = 1
dropout_rate = opt.dropout_rate
optimizer = Adam(lr=opt.lr, beta_1=opt.b1, beta_2=opt.b2)

# Discriminator
# pix2pixのpytorch実装では、
# MSELoss（patch毎の真贋判定なので、binary cross entropyではない）と
# L1Loss（本物の画像と偽物の画像のピクセル毎の比較）の併用
# Keras実装には下記サイトも参考にする（L1Lossのあたりとか）
# https://github.com/tommyfms2/pix2pix-keras-byt/blob/master/pix2pix.py
# pytorch実装の作者のKeras実装
# https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py
discriminator = discriminator(name='discriminator')
discriminator(Input((frames, height, width, out_channels * 2)))
# Discriminatorへの入力は、生成された画像もしくは変換後の真の画像と変換前の画像を結合したもの
discriminator.summary()
discriminator.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

# Generator
generator = generator(frames, out_channels, dropout_rate, name='generator')
generator(Input((frames, height, width, in_channels)))
generator.summary()

input_A = Input((frames, height, width, in_channels)) # Results of Low resolution
input_B = Input((frames, height, width, in_channels)) # Results of High resolution

fake_B = generator(input_A)

# For the combined model we will only train the generator
discriminator.trainable = False

# Discriminators determines validity of translated images / condition pairs
combined_imgs = Concatenate(axis=-1)([fake_B, input_A])
validity = discriminator(combined_imgs)

combined_Model = Model(inputs=[input_A, input_B], outputs=[validity, fake_B])
combined_Model.summary()
combined_Model.compile(loss=['mse', 'mae'],
                       loss_weights=[1, 100],
                       optimizer=optimizer)

if opt.epoch != 1:
    discriminator.load_weights('saved_models/%s/discriminator_%d.h5' % (opt.dataset_name, opt.epoch-1))
    combined_Model.load_weights('saved_models/%s/combined_Model_%d.h5' % (opt.dataset_name, opt.epoch-1))

def sample_images(dirs, epoch, type='horizontal', frame_interval=1):
    plot_list = []
    for i in range(0, frames, frame_interval):
        plot_list.append(i)

    for dir in dirs:
        imgs_A, imgs_B = data_loader.load_batch(dir, img_list)
        imgs_A = imgs_A.reshape(opt.batch_size, frames, height, width, in_channels)
        imgs_B = imgs_B.reshape(opt.batch_size, frames, height, width, in_channels)
        fake_B = generator.predict(imgs_A)

        titles = ['Low Resolution', 'Generated', 'High Resolution']

        imgs_A = imgs_A.reshape(frames, height, width, in_channels)
        imgs_B = imgs_B.reshape(frames, height, width, in_channels)
        fake_B = fake_B.reshape(frames, height, width, in_channels)

        if type == 'horizontal':
            fig, axs = plt.subplots(3, len(plot_list))
            for frame in range(len(plot_list)):
                # Low Resolution
#                img_tmp = 0.5 * imgs_A[plot_list[frame], :, :, :] + 0.5
                img_tmp = 255*(0.5 * imgs_A[plot_list[frame], :, :, :] + 0.5)
                print(img_tmp.max())
                print(img_tmp.min())
                img_tmp = np.array(img_tmp, np.uint8)
                img_tmp   = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
#                print(img_tmp.shape)
#                img_tmp = img_tmp.reshape(height, width)
                axs[0, frame].imshow(img_tmp)
                axs[0, frame].axis('off')
                if frame == len(plot_list) // 2:
                    axs[0, frame].set_title(titles[0])
                # Generated
#                img_tmp = 0.5 * fake_B[plot_list[frame], :, :, :] + 0.5
                img_tmp = 255*(0.5 * fake_B[plot_list[frame], :, :, :] + 0.5)
                print(img_tmp.max())
                print(img_tmp.min())
                img_tmp = np.array(img_tmp, np.uint8)
                img_tmp   = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
#                print(img_tmp.shape)
#                img_tmp = img_tmp.reshape(height, width)
                axs[1, frame].imshow(img_tmp)
                axs[1, frame].axis('off')
                if frame == len(plot_list) // 2:
                    axs[1, frame].set_title(titles[1])
                # High Resolution
#                img_tmp = 0.5 * imgs_B[plot_list[frame], :, :, :] + 0.5
                img_tmp = 255*(0.5 * imgs_B[plot_list[frame], :, :, :] + 0.5)
                print(img_tmp.max())
                print(img_tmp.min())
                img_tmp = np.array(img_tmp, np.uint8)
                img_tmp   = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
#                print(img_tmp.shape)
#                img_tmp = img_tmp.reshape(height, width)
                axs[2, frame].imshow(img_tmp)
                axs[2, frame].axis('off')
                if frame == len(plot_list) // 2:
                    axs[2, frame].set_title(titles[2])
        elif type == 'vertical':
            fig, axs = plt.subplots(len(plot_list), 3)
            for frame in range(len(plot_list)):
                # Low Resolution
#                img_tmp = 0.5 * imgs_A[plot_list[frame], :, :, :] + 0.5
                img_tmp = 255*(0.5 * imgs_A[plot_list[frame], :, :, :] + 0.5)
                print(img_tmp.max())
                print(img_tmp.min())
                img_tmp = np.array(img_tmp, np.uint8)
                img_tmp   = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
#                print(img_tmp.shape)
#                img_tmp = img_tmp.reshape(height, width)
                axs[frame, 0].imshow(img_tmp)
                axs[frame, 0].axis('off')
                if frame == 0:
                    axs[frame, 0].set_title(titles[0])
                # Generated
#                img_tmp = 0.5 * fake_B[plot_list[frame], :, :, :] + 0.5
                img_tmp = 255*(0.5 * fake_B[plot_list[frame], :, :, :] + 0.5)
                print(img_tmp.max())
                print(img_tmp.min())
                img_tmp = np.array(img_tmp, np.uint8)
                img_tmp   = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
#                print(img_tmp.shape)
#                img_tmp = img_tmp.reshape(height, width)
                axs[frame, 1].imshow(img_tmp)
                axs[frame, 1].axis('off')
                if frame == 0:
                    axs[frame, 1].set_title(titles[1])
                # High Resolution
#                img_tmp = 0.5 * imgs_B[plot_list[frame], :, :, :] + 0.5
                img_tmp = 255*(0.5 * imgs_B[plot_list[frame], :, :, :] + 0.5)
                print(img_tmp.max())
                print(img_tmp.min())
                img_tmp = np.array(img_tmp, np.uint8)
                img_tmp   = cv2.cvtColor(img_tmp, cv2.COLOR_GRAY2RGB)
#                print(img_tmp.shape)
#                img_tmp = img_tmp.reshape(height, width)
                axs[frame, 2].imshow(img_tmp)
                axs[frame, 2].axis('off')
                if frame == 0:
                    axs[frame, 2].set_title(titles[2])
        fig.savefig('images/%s/%s_%d.png' % (opt.dataset_name, dir, epoch))
        plt.close()

start_time = dt.datetime.now()

# Adversarial loss ground truths
valid = np.ones((opt.batch_size, frames, ) + disc_patch)
fake = np.zeros((opt.batch_size, frames, ) + disc_patch)

loss_Discriminator = []
loss_Generator = []
accuracy_Discriminator = []

train_dirs, val_dirs = data_loader.make_train_val_set(train_size=0.8)
print('train_dirs= ', train_dirs)
print('val_dirs= ', val_dirs)
for epoch in range(opt.epoch, opt.n_epochs+1):
#    train_dirs, val_dirs = data_loader.make_train_val_set(train_size=0.8)
    for batch_i, dir in enumerate(train_dirs):
        imgs_A, imgs_B = data_loader.load_batch(dir, img_list)
        imgs_A = imgs_A.reshape(opt.batch_size, frames, height, width, in_channels)
        imgs_B = imgs_B.reshape(opt.batch_size, frames, height, width, in_channels)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Condition on A and generate a translated version
        fake_B = generator.predict(imgs_A)

        # Train the discriminators (original images = real / generated = Fake)
#        combined_imgs_valid = Concatenate(axis=-1)([imgs_B, imgs_A])
#        combined_imgs_fake  = Concatenate(axis=-1)([fake_B, imgs_A])
        combined_imgs_valid = np.concatenate([imgs_B, imgs_A], axis=-1)
        combined_imgs_fake  = np.concatenate([fake_B, imgs_A], axis=-1)
        d_loss_real = discriminator.train_on_batch(combined_imgs_valid, valid)
        d_loss_fake = discriminator.train_on_batch(combined_imgs_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------

        # Train the generators
        g_loss = combined_Model.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])

        elapsed_time = dt.datetime.now() - start_time
        # Plot the progress
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, opt.n_epochs,
                                                                        batch_i+1, len(train_dirs),
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

        if epoch % opt.checkpoint_interval == 0:
            discriminator.save_weights('saved_models/%s/discriminator_%d.h5' % (opt.dataset_name, epoch))
            combined_Model.save_weights('saved_models/%s/combined_Model_%d.h5' % (opt.dataset_name, epoch))

        if epoch % opt.sample_epoch_interval == 0:
            sample_images(val_dirs, epoch, type='horizontal', frame_interval=4)

    loss_Discriminator.append(d_loss[0])
    loss_Generator.append(g_loss[0])
    accuracy_Discriminator.append(d_loss[1])

df = pd.DataFrame(data=None, columns=['loss_D', 'loss_G', 'acc_D'])
df['loss_D'] = loss_Discriminator
df['loss_G'] = loss_Generator
df['acc_D'] = accuracy_Discriminator
df.to_csv('loss_history.csv')
