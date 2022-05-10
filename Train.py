#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 22:07:11 2022

@author: premkumar
"""

import tensorflow as tf
from DataGenerator import dataloader
from Models import Unet
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tensorflow.python.ops import math_ops

root = '/Users/premkumar/Desktop/Neural Networks/Project'
 

def dice_coef(gt_mask, pred, smooth = 1e-5):
    
    
    axis = [1,2] #avoidong batch and channel axis
    gt_mask = gt_mask[:,:,:,1:]
    pred = pred[:,:,:,1:]
    #pred = tf.clip_by_value()
    
    numerator = tf.math.reduce_sum(gt_mask * pred, axis = axis)
    numerator = tf.math.reduce_mean(numerator, axis = -1)
    numerator = tf.math.reduce_mean(numerator)
    
    denominator1 = tf.math.reduce_sum(gt_mask, axis = axis)
    denominator1 = tf.math.reduce_mean(denominator1, axis = -1)
    denominator1 = tf.math.reduce_mean(denominator1)
    
    denominator2 = tf.math.reduce_sum(pred, axis = axis)
    denominator2 = tf.math.reduce_mean(denominator2, axis = -1)
    denominator2 = tf.math.reduce_mean(denominator2)
    
    dice_coef = (2 * numerator)/(denominator1 + denominator2 + smooth)
    dice_coef = dice_coef
    
    return dice_coef

def mean_iou(gt_mask, pred, smooth = 1e-7):
    
    axis = [1,2] #avoiding batch and channel axis
    epsilon = 1e-7
    pred = tf.clip_by_value(pred, epsilon, 1. - epsilon)
    
    true_p = tf.math.reduce_sum(gt_mask * pred, axis = axis)
    false_n = tf.math.reduce_sum(gt_mask * (1 - pred), axis = axis)
    false_p = tf.math.reduce_sum((1 - gt_mask) * pred, axis = axis)
    
    iou_class = (true_p + smooth)/(true_p + false_p + false_n + smooth)
    mean_iou = tf.math.reduce_mean(iou_class, axis = -1)
    
    batch_mean_iou = tf.math.reduce_mean(mean_iou)
    
    return batch_mean_iou

def tversky_loss(gt_mask, pred, delta = 0.7, gamma = 0.75):
    
    smooth = 1e-5
    epsilon = 1e-7
    axis = [1,2] # avoiding batch and channel axis
    gt_mask = gt_mask[:,:,:,1:]
    pred = pred[:,:,:,1:]
    pred = tf.clip_by_value(pred, epsilon, 1. - epsilon) # to avoid division by zero
    
    true_p = tf.math.reduce_sum(gt_mask * pred, axis = axis)
    false_n = tf.math.reduce_sum(gt_mask * (1 - pred), axis = axis)
    false_p = tf.math.reduce_sum((1 - gt_mask) * pred, axis = axis)
    
    tversky_class = (true_p + smooth)/(true_p + delta*false_n + (1 - delta)*false_p + smooth)
    tversky = tf.math.reduce_mean((1 - tversky_class), axis = -1)
    #focal_tversky = tf.math.reduce_mean(tf.math.pow((1 - dice_class), gamma), axis = -1)
    
    tversky_loss = tf.math.reduce_mean(tversky)
    
    return tversky_loss
       

def loss(gt_mask, pred):
    
    alpha = 0.75
    beta = 0.25
    t_loss = tversky_loss(gt_mask, pred)
    cce_loss = tf.keras.losses.categorical_crossentropy(gt_mask, pred, 
                                                        from_logits=False)
    cce_loss = tf.reduce_mean(cce_loss)
    
    loss = alpha * t_loss + beta * cce_loss
    
    return loss
    

def colormap_mask(mask):
    
    '''voc_classes = ["background","aeroplane","bicycle","bird","boat","bottle","bus","car",
                   "cat","chair","cow","diningtable","dog","horse","motorbike","person",
                   "potted plant","sheep","sofa","train","tv/monitor"]'''
    
    voc_colormap = [[0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0],[0, 0, 128],
                    [128, 0, 128],[0, 128, 128],[128, 128, 128],[64, 0, 0],
                    [192, 0, 0],[64, 128, 0],[192, 128, 0],[64, 0, 128],[192, 0, 128],
                    [64, 128, 128],[192, 128, 128],[0, 64, 0],[128, 64, 0],[0, 192, 0],
                    [128, 192, 0],[0, 64, 128]]
    
    col_mask = np.ones([mask.shape[0], mask.shape[1], 3]) * 255
    
    for l, rgb in enumerate(voc_colormap):
        col_mask[(mask == l)] = rgb
    
    col_mask = col_mask.astype(np.uint8)
    
    return col_mask

def stitch_images(image, pred, gt_mask):
        
    mask = tf.argmax(pred, axis = -1)
    mask = np.squeeze(mask, 0)
    col_mask = colormap_mask(mask)
    
    sep_line =(np.ones((image.shape[1], 10, 3)) * 255).astype(np.uint8)
    image = np.squeeze(image, 0) * 255
    image = image.astype(np.uint8)
    gt_mask = np.squeeze(tf.argmax(gt_mask, axis = -1), 0)
    gt_mask = colormap_mask(gt_mask)
    f_image = np.concatenate([image, sep_line, gt_mask, sep_line, col_mask], axis = 1)
    
    return f_image

def generate_mask(model):
    
    DL = dataloader(project_folder = root,
                    image_size = (256, 256))
    train_set, val_set = DL.tf_dataset(batch = 1)
    i = 0
    for image, gt_mask in train_set:
        pred = model(image)
        f_image = stitch_images(image, pred, gt_mask)
        plt.imsave(os.path.join(root, 'Test2', 'Train', '{}.jpg'.format(i)), f_image)
        i = i + 1
        
    j = 0
    for image, gt_mask in val_set:
        pred = model(image)
        f_image = stitch_images(image, pred, gt_mask)
        plt.imsave(os.path.join(root, 'Test2', 'Val', '{}.jpg'.format(j)), f_image)
        j = j + 1
        
        
def generate_graphs(train_loss, val_loss, val_dice, val_iou):
    
    fig, ax = plt.subplots(2, figsize = (20,20))
    #plot loss curve
    ax[0].plot(list(range(len(train_loss))), train_loss, label = 'Train Loss')
    ax[0].plot(list(range(len(val_loss))), val_loss, label = 'Validation Loss')
    ax[0].set_title('Loss Curves')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    #plot metrics
    ax[1].plot(list(range(len(val_dice))), val_dice, label = 'Dice Co-efficient')
    ax[1].plot(list(range(len(val_iou))), val_iou, label = 'Mean-IoU')
    ax[1].set_title('Metrics')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Metrics')
    ax[1].legend()
    
    plt.savefig('/Users/premkumar/Desktop/Neural Networks/Project/Graphs/' + datetime.now().strftime("%H:%M:%S") + '.jpg')

@tf.autograph.experimental.do_not_convert
def train(model, epochs, base_lr):
    
    save_dir = '/Users/premkumar/Desktop/Neural Networks/Project/Model/'
    
    DL = dataloader(project_folder = root,
                    image_size = (256, 256))
    
    train_set, val_set = DL.tf_dataset(batch = 5)
    
    num_tr_batches = len(train_set)
    num_val_batches = len(val_set)
    print(num_tr_batches, num_val_batches)
    
    if model == 'Unet':
        model = Unet(name = "Unet")
    
    
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #decay_epochs = 20
    # learning_rate = tf.compat.v1.train.exponential_decay(base_lr, global_step,
                                                          # num_tr_batches * decay_epochs,
                                                          # decay_rate = 0.95,
                                                          # staircase=False,
                                                          # name='exp_decay')
    
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = base_lr)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.9, nesterov=False, name='SGD')
    
    epoch_tloss = []
    epoch_vloss = []
    epoch_vdice = []
    epoch_viou = []
    for epoch in range(epochs):
        print('Epoch : {}, started at {}'.format(epoch, datetime.now().strftime("%H:%M:%S")))
        running_tloss = 0
        for images, gt_masks in tqdm(train_set):
            with tf.GradientTape() as tape:
                pred = model(images)
                curr_loss = loss(gt_masks, pred)
                running_tloss += curr_loss
                        
            grads = tape.gradient(curr_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
          
        epoch_tloss.append(running_tloss/num_tr_batches)
        print('\nTrain Loss : {}'.format(running_tloss/num_tr_batches))
        
        running_vloss = 0
        running_vdice = 0
        running_viou = 0
        for images, gt_masks in val_set:
            pred = model(images)
            curr_loss = loss(gt_masks, pred)
            running_vloss += curr_loss
            curr_dice = dice_coef(gt_masks, pred)
            running_vdice += curr_dice
            curr_iou = mean_iou(gt_masks, pred)
            running_viou += curr_iou
        
        vloss = running_vloss/num_val_batches
        vdice = running_vdice/num_val_batches
        viou = running_viou/num_val_batches
        epoch_vloss.append(vloss)
        epoch_vdice.append(vdice)
        epoch_viou.append(viou)
        print('\nVal Loss : {} Dice_coeff : {} Mean_IoU : {}'.format(vloss, vdice, viou))
    
    generate_mask(model)
    
    tf.saved_model.save(model, save_dir)
    generate_graphs(epoch_tloss, epoch_vloss, epoch_vdice, epoch_viou)

if __name__ == '__main__':
    
    with tf.device("/gpu:0"):    
        train(model = 'Unet', base_lr = 1e-4, epochs = 100)
            
        
            
            
            
            
            
            
    