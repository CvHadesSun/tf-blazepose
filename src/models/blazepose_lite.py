'''
Author: cvhadessun
Date: 2021-11-04 10:30:01
LastEditTime: 2021-11-04 15:03:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /tf-blazepose/src/models/blazepose_lite.py
'''

import tensorflow as tf
from layers import BlazeBlock
from tensorflow.keras.models import Model

class BlazePose():
    def __init__(self,num_keypoints: int):
        # super(BlazePose, self).__init__()
        self.num_keypoints = num_keypoints
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=(2, 2), padding='same', activation='relu'
        )
         
        # separable convolution (MobileNet)
        self.conv2 = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding='same', activation=None),
            tf.keras.layers.Conv2D(filters=16, kernel_size=1, activation=None)
        ])
        #  ---------- Heatmap branch ----------
        self.conv3 = BlazeBlock(block_num = 2, channel = 32)    # input res: 128
        self.conv4 = BlazeBlock(block_num = 3, channel = 64)    # input res: 64
        self.conv5 = BlazeBlock(block_num = 4, channel = 128)   # input res: 32
        self.conv6 = BlazeBlock(block_num = 5, channel = 192)   # input res: 16

        self.conv7a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.conv7b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation="relu")
        ])

        self.conv8a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv8b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation="relu")
        ])

        self.conv9a = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        self.conv9b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=32, kernel_size=1, activation="relu")
        ])


        # the output layer for heatmap and offset
        
        
        # ---------- Regression branch ----------
        #  shape = (1, 64, 64, 32)
        self.conv12a = BlazeBlock(block_num = 3, channel = 64)    # input res: 64
        self.conv12b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=64, kernel_size=1, activation="relu")
        ])

        self.conv13a = BlazeBlock(block_num = 4, channel = 128)   # input res: 32
        self.conv13b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=128, kernel_size=1, activation="relu")
        ])

        self.conv14a = BlazeBlock(block_num = 5, channel = 192)   # input res: 16
        self.conv14b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=192, kernel_size=1, activation="relu")
        ])

        self.conv15 = tf.keras.models.Sequential([
            BlazeBlock(block_num = 5, channel = 192, channel_padding = 0),
            BlazeBlock(block_num = 5, channel = 192, channel_padding = 0)
        ])

        self.seg_branch_a = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu")
        ])
        self.seg_branch_b = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu"),
            tf.keras.layers.UpSampling2D(size=(2, 2), interpolation="bilinear")
        ])
        self.seg_branch_c = tf.keras.models.Sequential([
            tf.keras.layers.DepthwiseConv2D(kernel_size=3, padding="same", activation=None),
            tf.keras.layers.Conv2D(filters=8, kernel_size=1, activation="relu")
        ])

        self.poseflag_head = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=1, kernel_size=2, name = 'conv_poseflag', activation='sigmoid'),
            tf.keras.layers.Reshape((1, 1))
        ])
        self.seg_head = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='same', name = 'output_segmentation')    
        self.ld_3d_head = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=self.num_keypoints*5, kernel_size=2, name='convld_3d'),
            tf.keras.layers.Reshape((1, self.num_keypoints*5))
        ])
        self.heatmap_head = tf.keras.layers.Conv2D(filters=self.num_keypoints, kernel_size=3, padding='same', name ='output_heatmap')
        self.world_3d_head = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=self.num_keypoints*3, kernel_size=2, name = 'convworld_3d'),
            tf.keras.layers.Reshape((1, self.num_keypoints*3))
        ])


    # def call(self, x):
    def build_model(self, model_type):
        # shape = (1, 256, 256, 3)
        input_x = tf.keras.layers.Input(shape=(256, 256, 3))
        x = self.conv1(input_x)
        # shape = (1, 128, 128, 16)
        x = x + self.conv2(x)   # <-- skip connection
        y0 = tf.keras.activations.relu(x)
        # #   --> I don't know why the relu layer is put after skip connection?
        # x = x + self.conv2_2(x)
        # y0 = tf.keras.activations.relu(x)

        # shape = (1, 128, 128, 16)
        y1 = self.conv3(y0)
        # shape = (1, 64, 64, 32)
        y2 = self.conv4(y1)
        # shape = (1, 32, 32, 64)
        y3 = self.conv5(y2)
        # shape = (1, 16, 16, 128)
        y4 = self.conv6(y3)
        # shape = (1, 8, 8, 192)

        x = self.conv7a(y4) + self.conv7b(y3)
        # shape = (1, 16, 16, 32)
        x = self.conv8a(x) + self.conv8b(y2)
        # shape = (1, 32, 32, 32)
        y = self.conv9a(x) + self.conv9b(y1)
        # shape = (1, 64, 64, 32)
        heatmap = self.heatmap_head(y)
        seg_branch = self.seg_branch_c(self.seg_branch_a(y0) + self.seg_branch_b(y))
        output_segmentation = self.seg_head(seg_branch)
        
        if model_type == "TWO_HEAD": # Stop gradient for regression on 2-head model
            y = tf.keras.backend.stop_gradient(y)
            y2 = tf.keras.backend.stop_gradient(y2)
            y3 = tf.keras.backend.stop_gradient(y3)
            y4 = tf.keras.backend.stop_gradient(y4)
            
        # ---------- regression branch ----------
        x = self.conv12a(y) + self.conv12b(y2)
        # shape = (1, 32, 32, 64)
        x = self.conv13a(x) + self.conv13b(y3)
        # shape = (1, 16, 16, 128)
        x = self.conv14a(x) + self.conv14b(y4)
        # shape = (1, 8, 8, 192)
        x = self.conv15(x)
        # shape = (1, 2, 2, 192)
        ld_3d = self.ld_3d_head(x)
        # world_3d = self.world_3d_head(x) # 
        output_poseflag = self.poseflag_head(x)
        heatmap = self.heatmap_head(y)

        
        # return  ld_3d, world_3d, output_poseflag, heatmap
        if model_type == "TWO_HEAD":
            return Model(inputs=input_x, outputs=[ld_3d, output_poseflag,heatmap,output_segmentation])
        elif model_type == "HEATMAP":
            return Model(inputs=input_x, outputs=[heatmap,output_segmentation])
        elif model_type == "REGRESSION":
            return Model(inputs=input_x, outputs=[ld_3d, output_poseflag])
        else:
            raise ValueError("Wrong model type.")