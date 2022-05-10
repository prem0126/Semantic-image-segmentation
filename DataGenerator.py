#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 19:19:26 2022

@author: premkumar
"""

import os
import tensorflow as tf
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import random


root = '/Users/premkumar/Desktop/Neural Networks/Project'
#labels_root = '/Users/premkumar/Desktop/Neural Networks/Project/labels.json'


class dataloader:
    
    def __init__(self, project_folder, image_size):
        
        self.project_folder = project_folder
        self.image_size = image_size
    
        
        with open(os.path.join(root, 'Image_ids.txt')) as image_ids:
            image_ids = image_ids.readlines()

        self.image_ids = [i.strip('\n') for i in image_ids]

        # with open(os.path.join(root, 'val.txt')) as val_ids:
        #     val_ids = val_ids.readlines()

        # self.val_ids = [i.strip('\n') for i in val_ids]
        
        self.voc_classes = ["background","aeroplane","bicycle","bird","boat",
                            "bottle","bus","car","cat","chair","cow","diningtable",
                            "dog","horse","motorbike","person","potted plant",
                            "sheep","sofa","train","tv/monitor"]
        
        self.voc_colormap = [[0, 0, 0],[128, 0, 0],[0, 128, 0],[128, 128, 0],
                             [0, 0, 128],[128, 0, 128],[0, 128, 128],[128, 128, 128],
                             [64, 0, 0],[192, 0, 0],[64, 128, 0],[192, 128, 0],
                             [64, 0, 128],[192, 0, 128],[64, 128, 128],[192, 128, 128],
                             [0, 64, 0],[128, 64, 0],[0, 192, 0],[128, 192, 0],
                             [0, 64, 128]]
            
    def load_data(self):
        
        
        image_paths = sorted([os.path.join(self.project_folder, 'Images', id+'.jpg') for id in self.image_ids])
        mask_paths = sorted([os.path.join(self.project_folder, 'Masks', id+'.jpg') for id in self.image_ids])
        
        # elif self.mode == 'val':
        #     image_paths = sorted([os.path.join(self.project_folder, 'Images', id+'.jpg') for id in self.val_ids])
        #     mask_paths = sorted([os.path.join(self.project_folder, 'Masks', id+'.jpg') for id in self.val_ids])

        return image_paths, mask_paths
    
    def get_image(self, image_path):
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = image/255.0
        image = image.astype(np.float32)
        
        return image
    
    def get_mask(self, mask_path):
        
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, self.image_size, interpolation = cv2.INTER_NEAREST)

        mask = mask.astype(np.float32)
        
        return mask
    
    def parse_function(self, image_path, mask_path):
        
        def parse(image_path, mask_path):
            image_path = os.path.join(image_path.decode())
            image = self.get_image(image_path)
            
            mask_path = os.path.join(mask_path.decode())
            mask = self.get_mask(mask_path)
            
            return image, mask
        
        image, mask = tf.numpy_function(parse, [image_path, mask_path], [tf.float32, tf.float32])
        
        return image, mask
    
    def horizontal_flip(self, image, mask):
        
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        
        return image, mask
    
    def corrupt_color(self, image, mask):
        
        options = ['brightness', 'saturation']
        #selection = random.choice(options)
        selection = 'saturation'
        
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        #value = np.random.choice(np.array([-40, -30, 30, 40]))
        
        if selection == 'brightness':
            value = random.uniform(0.5, 2)
            v = v * value
            v[v > 255] = 255
            
            final_hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            
            return image, mask
        
        elif selection == 'saturation':
            value = random.uniform(-0.5, 0.5)
            if value >= 0:
                lim = 255 - value
                s[s > lim] = 255
                s[s <= lim] += value
            else:
                lim = np.absolute(value)
                s[s < lim] = 0
                s[s >= lim] -= np.absolute(value)
            
            final_hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
            
            return image, mask
        
            
    
    def add_noise(self, image, mask, mean = 0, sd = 0.12):
        
        noise = np.random.normal(mean, sd, image.shape).astype(np.float32)
        image = cv2.add(image, noise)
        
        return image, mask
    
    def augment(self, image, mask):
        
        #apply = np.random.choice([True, False], p = [0.25, 0.75])
        apply = True
        
        if apply:
            def augment(image, mask):
                options = [self.corrupt_color, self.horizontal_flip,
                           self.add_noise]
                #augment_func = random.choice(options)
                image1 , mask1 = self.corrupt_color(image, mask)
                
                fig = plt.figure(figsize=(10, 7))
                fig.add_subplot(2, 2, 1)
                plt.imshow(image)
                fig.add_subplot(2, 2, 2)
                plt.imshow(mask)
                
                fig.add_subplot(2, 2, 3)
                plt.imshow(image1)
                fig.add_subplot(2, 2, 4)
                plt.imshow(mask1)
                plt.imsave('colour corrupt mask.jpg', mask.astype(np.uint8), dpi = 100)
                plt.imsave('colour corrupt image.jpg', image.astype(np.uint8)*255, dpi = 100)
                plt.imsave('colour corrupt image1.jpg', image1.astype(np.uint8)*255, dpi = 100)
                plt.imsave('colour corrupt mask1.jpg', mask1.astype(np.uint8), dpi = 100)
                
                
                
                return image, mask
            
            return tf.numpy_function(augment, [image, mask], [tf.float32, tf.float32])
        
        else:
            return image, mask
    
    def one_hot_encode(self, image, mask):
        
        def one_hot_encode(image, mask):
            height, width = mask.shape[:2]
            segmentation_mask = np.zeros((height, width, len(self.voc_colormap)), dtype=np.float32)
            
            for label_index, label in enumerate(self.voc_colormap):
                segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(np.float32)
            
            segmentation_mask = segmentation_mask.astype(np.float32)
            
            return image, segmentation_mask
        
        return tf.numpy_function(one_hot_encode, [image, mask], [tf.float32, tf.float32])
    
    def tf_dataset(self, batch):
        
        image_paths, mask_paths = self.load_data()
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        #dataset = dataset.shuffle(len(image_paths))
        dataset = dataset.map(self.parse_function)
        
        total_images = len(dataset)
        train_set = dataset.take(int(total_images * 0.8))
        val_set = dataset.skip(int(total_images * 0.8))
        
        train_set = train_set.take(1).map(self.augment) #Data Augmentation
        # train_set = train_set.map(self.one_hot_encode)
        # val_set = val_set.map(self.one_hot_encode)
        
        # train_set = train_set.batch(batch)
        # val_set = val_set.batch(batch)
        # train_set.prefetch(2)
        # val_set.prefetch(2)

        return train_set, val_set
        
                
if __name__ == '__main__':
    
    DL = dataloader(project_folder = root,
                    image_size = (256, 256))
    
    ds = DL.tf_dataset(batch = 1)

    
    for image,  mask in ds:
        image = np.squeeze(image, axis = 0)
        
        mask = tf.math.argmax(mask, axis = 3).numpy()
        mask = np.squeeze(mask, axis = 0)
        plt.imshow(mask)
        
        break
    
        
        
        
        
        
        
        
        
        
        
