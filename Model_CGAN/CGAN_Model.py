#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Eleuterio Juan Lillo Portero
# Final Project CGAN MODEL


# In[72]:


# Run on Google Colab. Run this cell if necessary
# from google.colab import drive
# drive.mount('/content/drive')


# In[67]:


# Import all neccesary 
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from numpy import expand_dims
from numpy import ones
from numpy import zeros
from numpy.random import rand
from numpy.random import randint

from keras.datasets.cifar10 import load_data
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils, to_categorical
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import Reshape
from keras.layers import Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import Input
from keras.layers import Embedding
from keras.layers import multiply 
from keras.layers import Concatenate
from keras.layers.noise import GaussianNoise
from keras.datasets import cifar10


# In[68]:


# Define Optimizer
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)
gan_optimizer = Adam(learning_rate=0.0004, beta_1=0.5)

# Declare class names
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']


# In[53]:


# Load cifar10 for GAN training
def load_cifar10():
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
    X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
    input_shape = (32, 32, 3)

    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # scale from [0,255] to [-1,1] and convert from unsigned ints to floats
    X_train = np.float32(X_train)
    X_train = (X_train / 255 - 0.5) * 2
    X_train = np.clip(X_train, -1, 1)

    X_test = np.float32(X_test)
    X_test = (X_test / 255 - 0.5) * 2
    X_test = np.clip(X_test, -1, 1)
    
    return (X_train, Y_train, X_test, Y_test)


# In[76]:


# Use the generator to generate n fake examples, with class labels
def generate_images(generator, latent_dim, n_batch):
    
    # generate noise for the generator and reshape
    noise = np.random.normal(0, 1, size=(n_batch, latent_dim))
    
    # create 'fake' class labels (0)
    random_labels = to_categorical(np.random.randint(0, 10, n_batch).reshape(-1, 1), num_classes=10)
    
    # predict outputs from generatos
    X_fake = generator.predict_on_batch([noise, random_labels])

    return X_fake, random_labels


# In[77]:


def plot_cgan(generator, latent_dim, n_batch, epoch):
    
    samples = 10
    z = np.random.normal(loc=0, scale=1, size=(samples, latent_dim))
    labels = to_categorical(np.arange(0, 10).reshape(-1, 1), num_classes=10)
        
    x_fake = generator.predict([z, labels])
    x_fake = np.clip(x_fake, -1, 1)
    x_fake = (x_fake + 1) * 127
    x_fake = np.round(x_fake).astype('uint8')

    for k in range(samples):
        plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
        plt.imshow(x_fake[k])
        plt.title(class_names[k])
    path='/content/drive/MyDrive/cgan/'
    filename = 'CGAN_generated_plot_e%03d.png' % (epoch+1)
    plt.tight_layout()
    plt.show()
    plt.savefig(path+filename) 
    plt.close()


# In[78]:


# Generate Discriminator and Generator losses
def plot_loss(losses):
      
    # Get generator and discriminator losses from data structure
    d_loss = losses["D"]
    g_loss = losses["G"]
    
    # Plot it
    plt.figure(figsize=(10,8))
    plt.plot(d_loss, label="Discriminator loss")
    plt.plot(g_loss, label="Generator loss")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# In[79]:


def create_generator(latent_dim, classes):
    
    # latent space dimension
    z = Input(shape=(latent_dim,))

    # classes
    labels = Input(shape=(classes,))

    # Generator network
    merged_layer = Concatenate()([z, labels])

    # FC: 2x2x512
    generator = Dense(2*2*512, activation='relu')(merged_layer)
    generator = LeakyReLU(alpha=0.2)(generator)
    generator = Reshape((2, 2, 512))(generator)

    # # Conv 1: 4x4x256
    generator = Conv2DTranspose(256, kernel_size=5, strides=2, padding='same')(generator)
    generator = LeakyReLU(alpha=0.2)(generator)

    # Conv 2: 8x8x128
    generator = Conv2DTranspose(128, kernel_size=5, strides=2, padding='same')(generator)
    generator = LeakyReLU(alpha=0.2)(generator)

    # Conv 3: 16x16x64
    generator = Conv2DTranspose(64, kernel_size=5, strides=2, padding='same')(generator)
    generator = LeakyReLU(alpha=0.2)(generator)

    # Conv 4: 32x32x3
    generator = Conv2DTranspose(3, kernel_size=5, strides=2, padding='same', activation='tanh')(generator)

    generator = Model(inputs=[z, labels], outputs=generator, name='generator')
    return generator


# In[80]:


def create_discriminator():
    
    # input image
    img_shape = (32,32,3)
    img_input = Input(shape=img_shape)
    
    # Create labels
    labels = Input(shape=(10,))

    # Conv 1: 16x16x64
    discriminator = Conv2D(64, kernel_size=5, strides=2, padding='same')(img_input)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)

    # Conv 2:
    discriminator = Conv2D(128, kernel_size=5, strides=2, padding='same')(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)

    # Conv 3: 
    discriminator = Conv2D(256, kernel_size=5, strides=2, padding='same')(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)

    # Conv 4: 
    discriminator = Conv2D(512, kernel_size=5, strides=2, padding='same')(discriminator)
    discriminator = LeakyReLU(alpha=0.2)(discriminator)

    # FC
    discriminator = Flatten()(discriminator)

    # Concatenate 
    merged_layer = Concatenate()([discriminator, labels])
    discriminator = Dense(512, activation='relu')(merged_layer)
    
    # Output
    discriminator = Dense(1, activation='sigmoid')(discriminator)

    discriminator = Model(inputs=[img_input, labels], outputs=discriminator, name='discriminator')
    
    # Optimizer
    discriminator.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['binary_accuracy'])
    
    return discriminator


# In[81]:


# Create Conditional GAN (CGAN)
def create_cgan(discriminator, generator, latent_dim):
    
    # Freeze Discriminator 
    discriminator.trainable = False

    # Create label and latent dimension
    label = Input(shape=(10,))
    noise = Input(shape=(latent_dim,))

    fake_img = generator([noise, label])
    validity = discriminator([fake_img, label])

    cgan = Model([noise, label], validity)

    cgan.compile(optimizer=gan_optimizer, loss='binary_crossentropy',
            metrics=['binary_accuracy'])
    cgan.summary()
    return cgan


# In[82]:


# Train the generator and discriminator
losses = {"D":[], "G":[]}
def train(generator, discriminator, cgan, dataset, latent_dim, epochs, n_batch, y_train):
    
    smoothness = 0.9
    bat_per_epo = int(dataset.shape[0] / n_batch)
    batches = len(dataset) // n_batch

    for epoch in range(epochs+1):
        for bat in range(batches):
            
            
            """ REAL SAMPLES"""
            # Create batch of real samples
            idx = np.random.randint(0, dataset.shape[0], n_batch)
            image_batch = dataset[idx]
            real_labels = y_train[idx]
          
            # Create validity for the Discriminator (CGAN) w/ smooth term
            valid = np.ones((n_batch, 1)) * (smoothness)
            
            """ FAKE SAMPLES """
            # Generate fake images by generator prediction
            fake_images, random_label = generate_images(generator, latent_dim, n_batch)
            fake_validity = np.zeros((n_batch, 1))

            # Train the discriminator
            discriminator.trainable = True
            d_loss_real, _ = discriminator.train_on_batch([image_batch, real_labels], valid)
            d_loss_fake, _ = discriminator.train_on_batch([fake_images, random_label], fake_validity)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
            
            # Set back to freezing discriminator
            discriminator.trainable = False
            
            """ TRAIN GENERATOR """
            # Create noise to train the generator
            noise = np.random.normal(0, 1, size=(n_batch, latent_dim))  
            
            # Labels for cgan training
            random_labels = to_categorical(np.random.randint(0, 10, n_batch).reshape(-1, 1), num_classes=10)
            validation_y = np.ones((n_batch, 1))
            
            # Train Generator through CGAN
            cgan_loss, _ = cgan.train_on_batch([noise, random_labels], validation_y)
            
            # Summarize loss for this batch
            print('Epoch: %d, batch %d/%d, discriminator loss =%.3f,            generator_loss =%.3f' % (epoch+1, bat+1, batches, d_loss, cgan_loss))
            
            # Plot for every ... TESTING
            # if (bat+1) % batches == 0:
            #     plot_cgan(generator, latent_dim, n_batch, epoch)
            
        losses["D"].append(d_loss)
        losses["G"].append(cgan_loss)
            
        # Save Generated model for every epoch
        if (epoch+1) % 1 == 0:
            # Plot
            plot_cgan(generator, latent_dim, n_batch, epoch)
            
            # Save generator and GAN model
            path='/content/drive/MyDrive/cgan/'
            filename = 'CGAN_CIFAR10_generator_model_%03d.h5' % (epoch + 1)
            generator.save(path + filename)
            
            name = 'CGAN_CIFAR10_gan_%03d.h5' % (epoch + 1)
            gan.save(path+name)
        


# In[83]:


# Size of input
latent_dim = 100
classes = 10

# Create models
discriminator = create_discriminator()
generator = create_generator(latent_dim, classes)
gan = create_cgan(discriminator, generator, latent_dim)

# Load image data
X_train, Y_train, X_test, Y_test = load_cifar10()
epochs = 100
n_batch = 128

# Train model
train(generator, discriminator, gan, X_train, latent_dim, epochs, n_batch, Y_train)


# In[84]:


plot_loss(losses)


# In[146]:


# CIFAR-10 classifier and GRAD-CAM based on generated CGAN images.
# Example of loading the CGAN generator model and generating images
from keras.models import load_model
 
# plot the generated images
def create_plot(samples, x_fake):

    # plot images
    for k in range(samples):
        plt.subplot(2, 5, k + 1, xticks=[], yticks=[])
        plt.imshow(x_fake[k])
        plt.title(class_names[k])
    plt.tight_layout()
    plt.show()
    plt.close()

# load model
# Google colab path '/content/drive/MyDrive/cgan/CGAN_CIFAR10_generator_model_101.h5'
model = load_model('/Users/lutelillo/Desktop/AAI_models/Model_CGAN/CGAN_CIFAR10_generator_model_101.h5')

# generate noise for the model prediction
samples = 10
latent_dim = 100
noise = np.random.normal(loc=0, scale=1, size=(samples, latent_dim))
labels = to_categorical(np.arange(0, 10).reshape(-1, 1), num_classes=10)
        
# generate images
x_pred = model.predict([noise, labels])
x_pred1 = (x_pred + 1) / 2.0

print('Shape of x_train is {}'.format(x_pred1.shape))

# plot the result
create_plot(samples, x_pred1)


# In[147]:


"""Example of a label prediction for the CGAN generated images"""

# Label list of Cifar-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']

# Load pre-trained Cifar10 classifier
# Accuracy of accuracy: 0.9349
classifier = load_model('/Users/lutelillo/Desktop/AAI_models/Model_classifier/cifar_10_class.h5')

# Plot the images based on the label given by the classifier
for i in range(10):
    plt.subplot(2, 5, i + 1, xticks=[], yticks=[])
    plt.imshow(x_pred1[i])
    img = np.array(x_pred1[i,:,:])
    p = img.reshape(1, 32, 32, 3)
    predicted_label = labels[classifier.predict(p).argmax()]
    plt.title(predicted_label)
plt.tight_layout()
plt.show()
plt.close()


# In[149]:


# Example showing what is the classifier looking at to make the classification on the generated images.
from matplotlib import cm
import tensorflow as tf
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore, BinaryScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
replace2linear = ReplaceToLinear()

# Rendering images with predicted label
list_score = []
for i in range(10):
    plt.subplot(2, 5, i + 1, xticks=[], yticks=[])
    img = np.array(x_pred1[i,:,:])
    p = img.reshape(1, 32, 32, 3)
    predicted_label = labels[classifier.predict(p).argmax()]
    list_score.append(classifier.predict(p).argmax())
    plt.imshow(x_pred1[i])
    plt.title(predicted_label)
plt.tight_layout()
plt.show()
plt.close()

score = CategoricalScore(list_score)

"""Create Gradcam object"""
# The output of Grad-CAM is a heatmap visualization for a given class label
# We can use this heatmap to visually verify where in the image the CNN is looking.
gradcam = Gradcam(classifier,
                  model_modifier=replace2linear,
                  clone=True)

# Create heatmap 
cam_heatmap = gradcam(score,
              x_pred1,
              penultimate_layer=-1)


# Display heat map camera
f, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 4))
for i, title in enumerate(labels):
    heatmap = np.uint8(cm.jet(cam_heatmap[i])[..., :3] * 255)
    ax[i].set_title(labels[list_score[i]], fontsize=16)
    ax[i].imshow(x_pred1[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()


# In[154]:


"""What is looking at the Grad-CAM for real samples """

# Load DATA for X_test
x_test,_,_,_ = load_cifar10()
x_real = x_test[540:550]
x_real = (x_real + 1) / 2.0


list_score = []

# Rendering images with predicted label
for i in range(10):
    plt.subplot(2, 5, i + 1, xticks=[], yticks=[])
    img = np.array(x_real[i,:,:])
    p = img.reshape(1, 32, 32, 3)
    predicted_label = labels[classifier.predict(p).argmax()]
    list_score.append(classifier.predict(p).argmax())
    plt.imshow(x_real[i])
    plt.title(predicted_label)
plt.tight_layout()
plt.show()
plt.close()

score = CategoricalScore(list_score)

# Create Gradcam object
# The output of Grad-CAM is a heatmap visualization for a given class label
# We can use this heatmap to visually verify where in the image the CNN is looking.
gradcam = Gradcam(classifier,
                  model_modifier=replace2linear,
                  clone=True)


# Create heatmap 
cam_heatmap = gradcam(score,
              x_real,
              penultimate_layer=-1)


# Display heat map camera
f, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 4))
for i, title in enumerate(labels):
    heatmap = np.uint8(cm.jet(cam_heatmap[i])[..., :3] * 255)
    ax[i].set_title(labels[list_score[i]], fontsize=16)
    ax[i].imshow(x_real[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.4)
    ax[i].axis('off')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




