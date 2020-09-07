## Mnist won't be used, instead we will use the dataset provided as images
from tensorflow.keras.datasets import mnist

from tensorflow.keras.layers import Input,Dense,Reshape,Flatten,Dropout,multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D,Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

class DCGAN():
    def __init__(self,img_shape = (254,254,3),noise_dim = 100,lr = 0.000002):
        # Input shape
        self.img_rows = img_shape[0]
        self.img_cols = img_shape[1]
        self.channels = img_shape[2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = noise_dim

        optimizer = Adam(lr, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 4 * 7 * 7, LeakyReLU(alpha=0.2), input_dim=self.latent_dim))
        model.add(Reshape((7*2, 7*2, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.add(UpSampling2D())

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=1))
        model.add(Dropout(0.6))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.2))
        model.add(LeakyReLU(alpha=0.6))
        model.add(Dropout(0.4))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.2))
        model.add(LeakyReLU(alpha=0.6))
        model.add(Dropout(0.4))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.2))
        model.add(LeakyReLU(alpha=0.6))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self,data,X, epochs, batch_size=128, save_interval=50):
        """ train(data,X,y,epochs,batch_size = 128, sample_interval = 50)
                    - Trains the GAN network given the data and the discriminators previously created
                    :param data:           Pandas DataFrame with the training (X,y) data.
                    :param X:              Name of the column that contains the original data.
                    :param batch_size:     Batch size of the data.
                    :param save_interval:  How you want an image to be saved
                """
        """In this case we will crop the images to produce just the optic disc"""
        # Load the data into memory
        X_train = []
        for File in data.iterrows():  # We iterate over the DataFrame
            I = cv2.imread(File[1][X])
            ## Extract Coordinates of Optic disc
            x = int(File[1]['x'])
            y = int(File[1]['y'])
            x_width = int(File[1]['x_width'])
            y_width = int(File[1]['y_width'])
            ## Extract patch
            I = I[x:x+x_width,y:y+y_width,:]
            #print(I.shape)
            #print(File[1][X])
            #print("----------")
            I = cv2.resize(I, (self.img_shape[0], self.img_shape[1]))
            I = cv2.normalize(I, None, alpha=0,  # Normalize image to fit from 0 to 1
                              beta=1,
                              norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_32F)
            X_train.append(I)
        X_train = np.array(X_train)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(cv2.cvtColor(gen_imgs[cnt],4))
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/OPTIC_DISCS_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
