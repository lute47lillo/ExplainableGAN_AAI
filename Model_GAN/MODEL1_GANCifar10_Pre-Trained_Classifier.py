#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Eleuterio Juan Lillo Portero
# Final Project 
# GAN MODEL 1 & GRAD-CAM visualization from Cifar_10 classifier


# In[9]:


# Import all needs
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


# In[10]:


# Define Optimizer
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)


# In[11]:


# Load cifar10 for GAN training
def load_cifar10():
    
    # load cifar10 dataset
    (trainX, _), (_, _) = load_data()
    
    # convert from unsigned ints to floats
    X = trainX.astype('float32')
    
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5
    
    return X


# In[5]:


# Use the generator to generate n fake examples, with class labels
def generate_images(generator, latent_dim, n_batch):
    
    # generate noise for the generator and reshape
    noise = np.random.normal(0, 1, size=(n_batch, latent_dim))
    
    # predict outputs from generatos
    X = generator.predict(noise)
    
    # create 'fake' class labels (0)
    y = np.zeros((n_batch,1))

    return X, y


# In[6]:


# Create plots of generated images (reversed grayscale), for report
def save_plot(examples, epoch, n=7):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    filename = 'TTTgenerated_plot_e%03d.png' % (epoch+1)
    plt.savefig(filename)
    plt.show()
    plt.close()


# In[7]:


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


# In[8]:


# Create the discriminator based on HW4 architechture 
def create_discriminator():

    model = Sequential()
    
    # normal
    model.add(Conv2D(64, (3,3), padding='same', input_shape=(32,32,3)))
    model.add(LeakyReLU(alpha=0.2))
    
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # downsample
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # downsample
    model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # classifier
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

d = create_discriminator()
#d.summary()


# In[9]:


# Create the generator
def create_generator(): 
    
    nodes = 256 * 4 * 4
    model = Sequential()
    model.add(Dense(nodes, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256)))
    
    # upsample to 8x8
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # upsample to 16x16
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 32x32
    
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    
    # output layer
    model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
    
    return model


g = create_generator()

# summarize the model
g.summary()


# In[10]:


# Create a GANd
def create_gan(d,g):
    d.trainable = False
    inputs = Input(shape=(100, ))
    hidden = g(inputs)
    output = d(hidden)
    gan = Model(inputs, output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    gan.summary()
    return gan


# In[11]:


# Evaluate the discriminator, plot generated images
def summarize_performance(epoch, generator, discriminator, dataset, latent_dim, n_batch):
    
    # prepare real samples
    image_batch = dataset[np.random.randint(0, dataset.shape[0], size=n_batch)]
    y_real = np.ones((n_batch, 1))
    
    # evaluate discriminator on real examples
    _, acc_real = discriminator.evaluate(image_batch, y_real, verbose=0)
    
    # prepare fake examples
    x_fake, y_fake = generate_images(generator, latent_dim, n_batch)
    
    # evaluate discriminator on fake examples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    
    # summarize discriminator performance
    print('Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    
    # save plot
    save_plot(x_fake, epoch)


# In[12]:


# Train the generator and discriminator
losses = {"D":[], "G":[]}
def train(generator, discriminator, gan, dataset, input_dim, epochs, n_batch):
    
    half_batch = int(n_batch/2)
    bat_per_epo = int(dataset.shape[0] / n_batch)

    for epoch in range(epochs):
        for bat in range(bat_per_epo):
            
            # Create a batch by drawing random index numbers from the training set
            image_batch = dataset[np.random.randint(0, dataset.shape[0], size=n_batch)]
            image_batch = image_batch.reshape(image_batch.shape[0], image_batch.shape[1], image_batch.shape[2], 3)
            y_real = np.ones((n_batch, 1))
            
            # Generate fake images by generator prediction
            fake_images, y_train = generate_images(generator, input_dim, n_batch)
            
            # Train the discriminator
            d.trainable = True
            d_loss_real, _ = discriminator.train_on_batch(image_batch, y_real)
            d_loss_fake, _ = discriminator.train_on_batch(fake_images, y_train)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 
            
            # Create noise to train the generator
            noise = np.random.normal(0, 1, size=(n_batch, 100))
            
            # Labels for fake samples
            validation_y = np.ones((n_batch,1))
            d.trainable = False
            
            # Train generator
            g_loss, _ = gan.train_on_batch(noise, validation_y)
            
            # Summarize loss for this batch
            print('Epoch: %d, batch %d/%d, discriminator loss =%.3f,            generator_loss =%.3f' % (epoch+1, bat+1, bat_per_epo, d_loss, g_loss))
            
            # Summarize performance every half and total n_batch
            if (bat+1) % bat_per_epo == 0:
                summarize_performance(bat, generator, discriminator, dataset, input_dim, n_batch)
            
        losses["D"].append(d_loss)
        losses["G"].append(g_loss)
            
        # Save Generated model for every epoch
        if (epoch+1) % 1 == 0:
            save_plot(fake_images, epoch)
            filename = 'JUP_CIFAR10_generator_model_%03d.h5' % (epoch + 1)
            generator.save(filename)


# In[13]:


# Size of input
input_dim = 100

# Create models
discriminator = create_discriminator()
generator = create_generator()
gan = create_gan(discriminator, generator)

# Load image data
dataset = load_cifar10()
epochs = 100
n_batch = 128

# Train model
train(generator, discriminator, gan, dataset, input_dim, epochs, n_batch)


# In[178]:


# Plot the loss of D and G
plot_loss(losses)


# In[31]:


# CIFAR-10 classifier and GRAD-CAM based on generated GAN images.
# Example of loading the generator model and generating images
from keras.models import load_model
 
# plot the generated images
def create_plot(examples, n):
    
    # plot images
    for i in range(n * n):
        
        # define subplot
        plt.subplot(n, n, 1 + i)
        # turn off axis
        plt.axis('off')
        # plot raw pixel data
        plt.imshow(examples[i, :, :])
        
    plt.show()
 
# load model. The path will be relative to your own machine
model = load_model('/Users/lutelillo/Desktop/AAI_models/Model_GAN/JUP_CIFAR10_generator_model_062.h5')

# generate noise for the model prediction
noise = np.random.normal(0, 1, size=(100, 100))

# generate images
x_pred = model.predict(noise)

print('Shape of x_train is {}'.format(x_pred.shape))
# scale from [-1,1] to [0,1]
X = (x_pred + 1) / 2.0

# plot the result
create_plot(X, 10)


# In[49]:


# Example of a label prediction from the pre-trained Cifar_10

# Label list of Cifar-10
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck']

# select the image from our test dataset
image_number = np.random.randint(0,100)

# display the image
plt.imshow(x_pred[image_number,:,:])

# load the image in an array
img = np.array(x_pred[image_number,:,:])

# Reshape it
p = img.reshape(1, 32, 32, 3)

# Load pre-trained Cifar10 classifier
# Has an accuracy of accuracy: 0.9349
# Path will be relative to your machine
classifier = load_model('/Users/lutelillo/Desktop/AAI_models/Model_classifier/cifar_10_class.h5')

# pass in the network for prediction and
# save the predicted label
predicted_label = labels[classifier.predict(p).argmax()]

# display the result
print("Predicted label is {}".format(predicted_label))
  


# In[54]:


# Example shows: What is the classifier looking at to make the classification on the generated images?
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
replace2linear = ReplaceToLinear()

# Batch of images generated by the GAN
X_pred = x_pred[30:40]
X_pred = np.clip(X_pred, -1, 1)

list_score = []
# Rendering images with predicted label
f, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 4))
for i, title in enumerate(labels):
    
    # Predict image on the cifar_10 classifier
    img = np.array(X_pred[i,:,:])
    p = img.reshape(1, 32, 32, 3)
    predicted_label = labels[classifier.predict(p).argmax()]
    list_score.append(classifier.predict(p).argmax())
    
    # Plot them
    ax[i].set_title(predicted_label, fontsize=16)
    ax[i].imshow(X_pred[i])
    ax[i].axis('off')
plt.tight_layout()
plt.show()

score = CategoricalScore(list_score)

# Create Gradcam object
# The output of Grad-CAM is a heatmap visualization for a given class label
# We can use this heatmap to visually verify where in the image the CNN is looking.
gradcam = Gradcam(classifier,
                  model_modifier=replace2linear,
                  clone=True)

# Create heatmap 
cam_heatmap = gradcam(score,
              X_pred,
              penultimate_layer=-1)


# Display heat map camera
f, ax = plt.subplots(nrows=1, ncols=10, figsize=(12, 4))
for i, title in enumerate(labels):
    heatmap = np.uint8(cm.jet(cam_heatmap[i])[..., :3] * 255)
    ax[i].set_title(labels[list_score[i]], fontsize=16)
    ax[i].imshow(X_pred[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    ax[i].axis('off')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




