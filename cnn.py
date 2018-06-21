import tensorflow as tf
from math import log
import numpy as np

def autoencoder(x,dim_z,noise_stddev,burst_error,O1=128,kapa=1,no_KL=1,deterministic=0):






    if deterministic==0:
        # encoding
        mu, sigma, dense_size = encoder(x, dim_z, depth=O1,determnistic=deterministic)
        #Reparametriazation
        z = mu+sigma* tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        # Add channel noise
        z = z + tf.random_normal(tf.shape(mu), 0, noise_stddev, dtype=tf.float32)+burst_error*np.random.choice([-1,1],1)

        # mean KL-Divergence
        KL_divergence = 0.5 * tf.reduce_mean(0*tf.square(mu) + tf.square(sigma) - tf.log(tf.maximum(tf.square(sigma), 1e-12)) - 1, 1)

        KL_divergence = tf.reduce_mean(KL_divergence)
        kappa = kapa




    else:
        mu, dense_size = encoder(x, dim_z, depth=O1,determnistic=deterministic)
        z = mu + tf.random_normal(tf.shape(mu), 0, noise_stddev, dtype=tf.float32)+burst_error*np.random.choice([-1,1],1)
        sigma=0*mu
        KL_divergence=tf.constant(0,dtype=tf.float32)
        kappa=1




        # decoding.3

    y2 = decoder(z,dense_size=dense_size,depth=O1)
    y=tf.nn.sigmoid(y2)


    # loss




    SE = tf.squared_difference(x, y)

    MSE = tf.reduce_mean(tf.reduce_mean(SE, axis=(1, 2, 3)))  # Mean of the MSEs of the batch * image_size


    PSNR = 10 * tf.log(1/MSE) / (log(10))  # Peak-SNR using Mean MSE

   # loss = 30000*kappa * MSE + no_KL*KL_divergence

    ce=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x,logits=y2))

    loss=ce+no_KL*KL_divergence/kappa

    return y, loss, 30000*MSE, KL_divergence, PSNR,mu,sigma





def encoder(X,dim_z,depth=128,determnistic=0):



    #1st convolutional layer
    conv1=conv(input=X,filter=128,name='conv1',seed=1,activation='relu',strides=2)

    #max pooling & batch normalization
    conv1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    #2nd convolutional layer
    conv2=conv(input=conv1,filter=depth,name='conv2',seed=2,activation='relu')

    #max pooling , batch normalization and relu
    #conv2=tf.layers.max_pooling2d(inputs=norma(conv2), pool_size=[2, 2], strides=2)

    #flatten
    flat=tf.layers.flatten(conv2)

    #vector with means
    mu = mydense(input=flat, units=dim_z, seed=3)


    #Fully connected layer
    if determnistic==1:
        average_power = tf.sqrt(tf.reduce_mean(mu ** 2, axis=1, keepdims=True))

        mu = mu / average_power

        return mu, flat._shape.dims[1].value


    else: #Variational

        sigma=mydense(input=flat,units=dim_z,seed=4)

        average_power = tf.sqrt(tf.reduce_mean(mu ** 2 + sigma ** 2, axis=1, keepdims=True))

        mu = mu / average_power
        sigma = sigma / average_power

        return mu, sigma, flat._shape.dims[1].value



# decoder
def decoder(z,dense_size,depth=128):


    #Fully-Connectd Layer
    FC=mydense(input=z,units=dense_size,seed=5)

    # Reshaping to 2D image with multiple channels. [Batch_Size,25x25x128]->[Batch_Size,25,25,128], inverse of flattening
    y = tf.reshape(FC, [-1, 25,25,depth])

    #unpool & normalize and relu
   # y=tf.image.resize_images(images=y, size=[50, 50], align_corners=False)

    #1st deconvolutional layer
    y=deconv(input=y,filter=128,name='deconv2',seed=6,activation='relu')

    # unpool & normalize
    y = tf.image.resize_images(images=norma(y), size=[50,50], align_corners=False)

    #2nd deconv and final layer
    out = deconv(input=y, filter=3,name='deconv1', seed=7, activation='sigmoid',strides=2)
    return out






def norma(X):
    return tf.nn.batch_normalization(X, mean=0, variance=1, offset=0, scale=1, variance_epsilon=1e-12)

def conv(input,filter,name,seed,activation,strides=1):
    if activation=='relu':
        with tf.variable_scope(name) as scope:
            y = tf.layers.conv2d(name=scope.name, inputs=input, filters=filter, kernel_size=[5, 5], strides=strides, padding="same",
                                 use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.04, seed=seed),
                                 activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
    else:
        with tf.variable_scope(name) as scope:
            y = tf.layers.conv2d(name=scope.name, inputs=input, filters=filter, kernel_size=[5, 5], strides=strides,
                                 padding="same",
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.04, seed=seed),
                                 reuse=tf.AUTO_REUSE,activation=tf.nn.leaky_relu)
    return y
def deconv(input,filter,name,seed,activation,strides=1):
    if activation=='relu':
        with tf.variable_scope(name) as scope:
            y = tf.layers.conv2d_transpose(name=scope.name, inputs=input, filters=filter, kernel_size=[5, 5], strides=strides, padding="same",
                                 use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=0.04, seed=seed),
                                 activation=tf.nn.relu, reuse=tf.AUTO_REUSE)
    else:
        with tf.variable_scope(name) as scope:
            y = tf.layers.conv2d_transpose(name=scope.name, inputs=input, filters=filter, kernel_size=[5, 5], strides=strides,
                                 padding="same",
                                 use_bias=True,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.04, seed=seed),
                                 reuse=tf.AUTO_REUSE,activation=tf.nn.leaky_relu)
    return y
def mydense(input,units,seed):
    dense_size = input._shape.dims[1].value
    y = tf.layers.dense(inputs=input, units=units,
                               use_bias=True,
                               kernel_initializer=tf.random_normal_initializer(stddev=1 / dense_size, seed=seed))
    return y
