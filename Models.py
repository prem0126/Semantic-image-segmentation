#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:40:20 2022

@author: premkumar
"""
import tensorflow as tf
from DataGenerator import dataloader
import numpy as np
#import os

class conv2D(tf.Module):
    
    def __init__(self, kernel_size, padding, strides, activation, name = None,
                 layer_name = None, batch_norm = True):
        super().__init__(name = name)
        
        initializer = tf.keras.initializers.GlorotNormal(seed = 0)
        self.w = tf.Variable(initializer(kernel_size), name = 'W',
                             trainable = True)
        self.b = tf.Variable(tf.zeros([kernel_size[3]]), name = 'B')
        self.padding = padding
        self.strides = strides
        self.activation = activation
        self.batch_norm = batch_norm
    
    #@tf.function
    def __call__(self, x):
        
        y = tf.nn.conv2d(x, self.w, strides = [1,self.strides,self.strides,1],
                         padding = self.padding)
        y = tf.nn.bias_add(y, self.b)
        
        if self.batch_norm:
            y = tf.keras.layers.BatchNormalization()(y)
        
        if self.activation == 'relu':
            return tf.nn.relu(y)
        elif self.activation == 'softmax':
            return tf.nn.softmax(y)

class maxpool_2D(tf.Module):
    
    def __init__(self, ksize, padding, name = None):
        super().__init__(name = name)
        
        self.ksize = ksize
        self.padding = padding
    
    #@tf.function
    def __call__(self, x):
        
        return tf.nn.max_pool(x, ksize = [1,self.ksize,self.ksize,1],
                              strides = [1,self.ksize,self.ksize,1],
                              padding = self.padding)

class transpose_conv2d(tf.Module):
    
    def __init__(self, kernel_size, padding, strides, activation = None, name = None,
                 bias = True, layer_name = None):
        super().__init__(name = name)
        
        initializer = tf.keras.initializers.GlorotNormal(seed = 0)
        self.w = tf.Variable(initializer(kernel_size), name = 'W',
                             trainable = True)
        self.bias = bias
        if self.bias:
            self.b = tf.Variable(tf.zeros([kernel_size[2]]), name = 'B',
                                 trainable = True)
        self.padding = padding
        self.strides = strides
        self.activation = activation
    
    #@tf.function
    def __call__(self, x):
        
        input_shape = tf.TensorShape(x.shape).as_list()
        output_shape = list(input_shape)
        c_axis, h_axis, w_axis = 3, 1, 2
        output_shape[c_axis] = self.w.shape.as_list()[2]
        output_shape[h_axis] = output_shape[h_axis] * self.strides
        output_shape[w_axis] = output_shape[w_axis] * self.strides
        output_shape = tf.TensorShape(output_shape)
        
        y = tf.nn.conv2d_transpose(x, self.w, output_shape, strides = [1,self.strides,self.strides,1], 
                                   padding = self.padding)
        
        if self.bias:
            y = tf.nn.bias_add(y, self.b)
        
        if self.activation == None:
            return y
        elif self.activation == 'relu':
            return tf.nn.relu(y)

class unet_conv(tf.Module):
    
    def __init__(self, in_channels, out_channels, padding = 'SAME', name = None):
        super().__init__(name = name)
        
        self.conv1 = conv2D(kernel_size = [3,3,in_channels,out_channels],
                            padding = padding, strides = 1,
                            activation = 'relu')
        
        self.conv2 = conv2D(kernel_size = [3,3,out_channels,out_channels],
                            padding = padding, strides = 1,
                            activation = 'relu')
    
    @tf.autograph.experimental.do_not_convert
    def __call__(self, x):
        
        A = self.conv1(x)
        output = self.conv2(A)
        
        return output

class unet_tconv(tf.Module):
    
    def __init__(self, in_channels, out_channels, padding = 'SAME', name = None):
        super().__init__(name = name)
        
        self.t_conv = transpose_conv2d(kernel_size = [2,2,out_channels,in_channels],
                                    padding = 'SAME', strides = 2)
        
        self.conv1 = conv2D(kernel_size = [3,3,in_channels,out_channels],
                          padding = 'SAME', strides = 1, activation = 'relu')
        
        self.conv2 = conv2D(kernel_size = [3,3,out_channels,out_channels],
                            padding = 'SAME', strides = 1, activation = 'relu')
    
    @tf.autograph.experimental.do_not_convert
    def __call__(self, x1, x2):
        
        A1 = self.t_conv(x1)
        x = tf.keras.layers.Concatenate()([A1, x2])
        A2 = self.conv1(x)
        output = self.conv2(A2)
        
        return output
    
class Unet(tf.Module):
    
    def __init__(self, name = None):
        super().__init__(name = name)
        
        self.conv1 = unet_conv(in_channels = 3, out_channels = 64)
        self.maxpool1 = maxpool_2D(ksize = 2, padding = 'VALID')
        
        self.conv2 = unet_conv(in_channels = 64, out_channels = 128)
        self.maxpool2 = maxpool_2D(ksize = 2, padding = 'VALID')
        
        self.conv3 = unet_conv(in_channels = 128, out_channels = 256)
        self.maxpool3 = maxpool_2D(ksize = 2, padding = 'VALID')
        
        self.conv4 = unet_conv(in_channels = 256, out_channels = 512)
        self.maxpool4 = maxpool_2D(ksize = 2, padding = 'VALID')
        
        self.bridge = unet_conv(in_channels = 512, out_channels = 1024)
        
        self.upconv1 = unet_tconv(in_channels = 1024, out_channels = 512)
        self.upconv2 = unet_tconv(in_channels = 512, out_channels = 256)
        self.upconv3 = unet_tconv(in_channels = 256, out_channels = 128)
        self.upconv4 = unet_tconv(in_channels = 128, out_channels = 64)
        
        self.outlayer = conv2D(kernel_size = [1,1,64,21], padding = 'SAME', strides = 1,
                               activation = 'softmax', batch_norm = False) 
        
    @tf.autograph.experimental.do_not_convert
    def __call__(self, x):
        
        A1 = self.conv1(x)
        D1 = self.maxpool1(A1)
        
        A2 = self.conv2(D1)
        D2 = self.maxpool2(A2)
        
        A3 = self.conv3(D2)
        D3 = self.maxpool3(A3)
        
        A4 = self.conv4(D3)
        D4 = self.maxpool4(A4)
        
        B = self.bridge(D4)
        
        U1 = self.upconv1(B, A4)
        U2 = self.upconv2(U1, A3)
        U3 = self.upconv3(U2, A2)
        U4 = self.upconv4(U3, A1)
        
        output = self.outlayer(U4)
        
        return output
    


# if __name__ == '__main__':
    
#     root = '/Users/premkumar/Desktop/Neural Networks/Project'
    
#     DL = dataloader(project_folder = root,
#                     image_size = (256, 256),
#                     mode = 'train')
#     ds = DL.tf_dataset(batch = 1)
    
#     unet = Unet(name = "U_Net")
    
#     for image, mask in ds.take(3):
#         pred = unet(image)
#         print(np.unique(pred))
    
# #     #print(unet.trainable_variables)
        
        
        
        
        
        
        
        
        
        
        
        