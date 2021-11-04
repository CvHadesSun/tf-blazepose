import json
import math
import os
import random

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence

from ..utils.heatmap import gen_gt_heatmap
from ..utils.keypoints import normalize_landmark
from ..utils.pre_processing import square_crop_with_keypoints
from ..utils.visualizer import visualize_keypoints
from .augmentation import augment_img
from .augmentation_utils import random_occlusion
from ..utils.affined_trans import get_rotation,get_transform_matrix,adjust_bbox,affine_joints
from ..utils.segm import affine_segmentation
from pycocotools.coco import COCO # for coco format dataset.

class DataSequence(Sequence):

    def __init__(self, 
                image_folder, 
                label_file, 
                batch_size=8, 
                input_size=(256, 256), 
                output_heatmap=True, 
                output_segmentation=False,
                output_poseflag = False,
                heatmap_size=(64, 64), 
                heatmap_sigma=4, 
                n_points=16, 
                shuffle=True, 
                augment=False, 
                random_flip=False, 
                random_rotate=False, 
                random_scale_on_crop=False, 
                clip_landmark=False, 
                symmetry_point_ids=None):

        self.batch_size = batch_size
        self.input_size = input_size
        self.output_heatmap = output_heatmap
        self.output_poseflag = output_poseflag
        self.output_segmentation = output_segmentation
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.image_folder = image_folder
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.random_scale_on_crop = random_scale_on_crop
        self.augment = augment
        self.n_points = n_points
        self.symmetry_point_ids = symmetry_point_ids
        self.clip_landmark = clip_landmark # Clip value of landmark to range [0, 1]


        # if os.path.exists(label_file):
        self.anno = COCO(label_file)
        self.indexes = list(self.anno.anns.keys())
        if shuffle:
            random.shuffle(self.indexes)

        # 
        self.debug=False

    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return math.ceil(len(self.indexes) / float(self.batch_size))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """
        batch_data = self.indexes[idx *
                               self.batch_size: (1 + idx) * self.batch_size]

        batch_image = [] # input
        # output 
        batch_landmark_2d = [] # landmark2d
        batch_landmark_3d = [] # landmark3d
        batch_heatmap = [] # heatmap 
        batch_segms = []  # segmentation
        batch_posflag = [] # poseflag.

        for id in batch_data:

            # Load and augment data
            image, landmark, heatmap,segmentation,pose_flag = self.load_data_v3(self.image_folder, id)

            batch_image.append(image)
            batch_landmark_2d.append(landmark)
            if self.output_heatmap:
                batch_heatmap.append(heatmap)
            if self.output_poseflag:
                batch_posflag.append(pose_flag)
            if self.output_segmentation:
                batch_segms.append(segmentation)

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark_2d)
        if self.output_heatmap:
            batch_heatmap = np.array(batch_heatmap)

        batch_image = DataSequence.preprocess_images(batch_image)
        batch_landmark = self.preprocess_landmarks(batch_landmark)

        # Prevent values from going outside [0, 1]
        # Only applied for sigmoid output
        if self.clip_landmark:
            batch_landmark[batch_landmark < 0] = 0
            batch_landmark[batch_landmark > 1] = 1

        if self.output_heatmap:
            return batch_image, [batch_landmark, batch_heatmap]
        elif self.output_poseflag:
            return batch_image, [batch_landmark, batch_heatmap,batch_segms,batch_posflag]
        else:
            return batch_image, batch_landmark

    @staticmethod
    def preprocess_images(images):
        # Convert color to RGB
        for i in range(images.shape[0]):
            images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float)
        images = np.array(images, dtype=np.float32)
        images = images / 255.0
        images -= mean
        return images
    @staticmethod
    def preprocess_images_v2(images,value=[0,1]):
        # Convert color to RGB
        # for i in range(images.shape[0]):
        #     images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        # mean = np.array([0.5, 0.5, 0.5], dtype=np.float)
        images = np.array(images, dtype=np.float32)
        images = images / 255. * (value[1] - value[0]) + value[0]
        return images
    def preprocess_landmarks(self, landmarks):

        first_dim = landmarks.shape[0]
        try:
            landmarks = landmarks.reshape((-1, 5))
        except:
            landmarks = landmarks.reshape((-1, 3))
        landmarks = normalize_landmark(landmarks, self.input_size)
        landmarks = landmarks.reshape((first_dim, -1))
        return landmarks

    def load_data(self, img_folder, aid):
        # momo_style = True # debug

        ann = self.anno.anns[aid]  # get one annotation.
        # Load image
        img_name = self.anno.imgs[ann['image_id']]['file_name']
        path = os.path.join(img_folder, img_name)
        image = cv2.imread(path)

        joints = []
        segmentation = []

        # Load landmark and apply square cropping for image
        if 'keypoints' in ann:
            joints = ann['keypoints']  
        if 'segmentation' in ann:
            segmentation = ann['segmentation']
        bbox = ann['bbox']   # bbox
        landmark = np.array(joints).reshape([-1,3]) #  coco format.

        # Convert all (-1, -1) to (0, 0)
        for i in range(landmark.shape[0]):
            if landmark[i][0] == -1 and landmark[i][1] == -1:
                landmark[i, :] = [0, 0]

        # Generate visibility mask
        # visible = inside image + not occluded by simulated rectangle
        # (see BlazePose paper for more detail)
        visibility = np.ones((landmark.shape[0]), dtype=int)
        vis_index = np.where(landmark[:,-1]>0)
        landmark[vis_index,-1] = 1
        visibility = landmark[:,-1]
    
        for i in range(len(visibility)):
            if 0 > landmark[i][0] or landmark[i][0] >= image.shape[1] \
                or 0 > landmark[i][1] or landmark[i][1] >= image.shape[0]:
                visibility[i] = 0

        # if momo_style:
        # recostruct the landmarks.
        np_landmark = np.zeros([landmark.shape[0],5])
        np_landmark[:,:2] = landmark[:,:2]  # x,y
        np_landmark[:,2] = landmark[:,0]  #z=x
        np_landmark[:,3] = visibility
        np_landmark[:,4] = visibility
        # 
        # scale = [1.,1.]
        bbox= self._compute_bbox(np_landmark[-2],np_landmark[-1]) # square box
        rotation = get_rotation(np_landmark[-2, :2], np_landmark[-1, :2])
        ad_bbox = adjust_bbox(bbox, rotation)
        trans = get_transform_matrix(ad_bbox, rotation, [self.input_size[0], self.input_size[1]])
        image = cv2.warpPerspective(image, trans, (self.input_size[1], self.input_size[0]),
                                        flags=cv2.INTER_LINEAR)  # padding zeros
        landmark, visibility = affine_joints(np_landmark, visibility, trans, [self.input_size[0], self.input_size[1]])
        
        if len(segmentation) > 0:
            segmentation = affine_segmentation(segmentation, trans)
        
        # assign 
        bbox = ad_bbox

        # augmentation.
        # Horizontal flip
        # and update the order of landmark points
        if self.random_flip and random.choice([0, 1]):
            image = cv2.flip(image, 1)

            # Mark missing keypoints
            missing_idxs = []
            for i in range(landmark.shape[0]):
                if landmark[i, 0] == 0 and landmark[i, 1] == 0:
                    missing_idxs.append(i)

            # Flip landmark
            landmark[:, 0] = self.input_size[0] - landmark[:, 0]

            # Restore missing keypoints
            for i in missing_idxs:
                landmark[i, 0] = 0
                landmark[i, 1] = 0

            # Change the indices of landmark points and visibility
            if self.symmetry_point_ids is not None:
                for p1, p2 in self.symmetry_point_ids:
                    l = landmark[p1, :].copy()
                    landmark[p1, :] = landmark[p2, :].copy()
                    landmark[p2, :] = l

        if self.augment:
            image, landmark[:,:2] = augment_img(image, landmark[:,:2])

        # Random occlusion
        # (see BlazePose paper for more detail)
        if self.augment and random.random() < 0.2:
            # landmark = landmark.reshape(-1, 2)
            image, visibility = random_occlusion(image, landmark[:,:2], visibility=visibility,
                                                 rect_ratio=((0.2, 0.5), (0.2, 0.5)), rect_color="random")

        visibility = np.array(visibility)
        visibility = visibility.reshape((landmark.shape[0]))
        landmark[:,3] = visibility
        landmark[:,4] = visibility
        
        # Generate heatmap
        gtmap = None
        if self.output_heatmap:
            gtmap_kps = landmark[:,:2].copy()
            gtmap_kps[:, :2] = (np.array(gtmap_kps[:, :2]).astype(float)
                                * np.array(self.heatmap_size) / np.array(self.input_size)).astype(int)
            gtmap = gen_gt_heatmap(
                gtmap_kps,visibility, self.heatmap_sigma, self.heatmap_size)
            # gtmap = np.clip(np.sum(gtmap, axis=2, keepdims=True), None, 1)

        # Uncomment following lines to debug augmentation
        if self.debug:
            draw = visualize_keypoints(image, landmark, visibility, text_color=(0,0,255))
            cv2.imwrite('/home/hades/dongmi_projects/tf-blazepose/output/gt_{}.jpg'.format(aid),draw)
        # if self.output_heatmap:
        #     cv2.imwrite('/home/hades/dongmi_projects/tf-blazepose/output/gt_{}.jpg'.format(aid),gtmap.sum(axis=2)*255)
        #     cv2.imshow("gtmap", gtmap.sum(axis=2))
        # cv2.waitKey(0)

        # return image, landmark, gtmap
        score = self._scoring(visibility)
        return image,landmark,gtmap,segmentation,score

    def load_data_v2(self, img_folder, aid):
        # momo_style = True # debug
        ann = self.anno.anns[aid]  # get one annotation.
        # Load image
        img_name = self.anno.imgs[ann['image_id']]['file_name']
        path = os.path.join(img_folder, img_name)
        image = cv2.imread(path)
        # ori_img=image.copy()

        joints = []
        segmentation = []
        score=ann['pose_flag']
        # Load landmark and apply square cropping for image
        if 'keypoints' in ann:
            joints = ann['keypoints']  
        if 'segmentation' in ann:
            segmentation = ann['segmentation']
        bbox = ann['bbox']   # bbox
        landmark = np.array(joints).reshape([-1,5]) #  blazepose format.

        # Convert all (-1, -1) to (0, 0)
        for i in range(landmark.shape[0]):
            if landmark[i][0] == -1 and landmark[i][1] == -1:
                landmark[i, :] = [0, 0]

        # Generate visibility mask
        # visible = inside image + not occluded by simulated rectangle
        # (see BlazePose paper for more detail)
        visibility = landmark[:,-2] # directly to get the vis info from gt label.
    
        for i in range(len(visibility)):
            if 0 > landmark[i][0] or landmark[i][0] >= image.shape[1] \
                or 0 > landmark[i][1] or landmark[i][1] >= image.shape[0]:
                visibility[i] = 0

        landmark[:,0] *=image.shape[1]
        landmark[:,1] *=image.shape[0]

        landmark[:,-2] = visibility
        # 
        # scale = [1.,1.]
        # bbox= self._compute_bbox(landmark[33],landmark[34]) # square box
        rotation = get_rotation(landmark[33, :2], landmark[34, :2])
        ad_bbox = adjust_bbox(bbox, rotation)
        trans = get_transform_matrix(ad_bbox, rotation, [self.input_size[0], self.input_size[1]])
        image = cv2.warpPerspective(image, trans, (self.input_size[1], self.input_size[0]),
                                        flags=cv2.INTER_LINEAR)  # padding zeros
        landmark, visibility = affine_joints(landmark, visibility, trans, [self.input_size[0], self.input_size[1]])
        
        if len(segmentation) > 0:
            segmentation = affine_segmentation(segmentation, trans)
        
        # assign 
        # bbox = ad_bbox
        # augmentation.
        # Horizontal flip
        # and update the order of landmark points
        if self.random_flip and random.choice([0, 1]):
            image = cv2.flip(image, 1)

            # Mark missing keypoints
            missing_idxs = []
            for i in range(landmark.shape[0]):
                if landmark[i, 0] == 0 and landmark[i, 1] == 0:
                    missing_idxs.append(i)

            # Flip landmark
            landmark[:, 0] = self.input_size[0] - landmark[:, 0]

            # Restore missing keypoints
            for i in missing_idxs:
                landmark[i, 0] = 0
                landmark[i, 1] = 0

            # Change the indices of landmark points and visibility
            if self.symmetry_point_ids is not None:
                for p1, p2 in self.symmetry_point_ids:
                    l = landmark[p1, :].copy()
                    landmark[p1, :] = landmark[p2, :].copy()
                    landmark[p2, :] = l

        # if self.augment:
        #     image, landmark[:,:2] = augment_img(image, landmark[:,:2])

        # Random occlusion
        # (see BlazePose paper for more detail)
        if self.augment and random.random() < 0.2:
            # landmark = landmark.reshape(-1, 2)
            image, visibility = random_occlusion(image, landmark[:,:2], visibility=visibility,
                                                 rect_ratio=((0.2, 0.5), (0.2, 0.5)), rect_color="random")

        # Concatenate visibility into landmark
        visibility = np.array(visibility)
        visibility = visibility.reshape((landmark.shape[0]))
        landmark[:,3] = visibility
        
        # Generate heatmap
        gtmap = None
        if self.output_heatmap:
            gtmap_kps = landmark[:,:2].copy()
            gtmap_kps[:, :2] = (np.array(gtmap_kps[:, :2]).astype(float)
                                * np.array(self.heatmap_size) / np.array(self.input_size)).astype(int)
            gtmap = gen_gt_heatmap(
                gtmap_kps,visibility, self.heatmap_sigma, self.heatmap_size)
            # gtmap = np.clip(np.sum(gtmap, axis=2, keepdims=True), None, 1)

        # Uncomment following lines to debug augmentation
        if self.debug:
            draw = visualize_keypoints(image, landmark, visibility, text_color=(0,0,255))
            cv2.imwrite('/home/hades/dongmi_projects/tf-blazepose/output/gt_{}.jpg'.format(aid),draw)
            # cv2.namedWindow("draw", cv2.WINDOW_NORMAL)
            # cv2.imshow("draw", draw)

        # if self.output_heatmap:
        #     cv2.imshow("gtmap", gtmap.sum(axis=2))
        # cv2.waitKey(0)

        # return image, landmark, gtmap
        # score = self._scoring(visibility)

        return image,landmark,gtmap,segmentation,score
    def _scoring(self,vis):
        score = vis.astype(np.bool).sum()
        return score
    def load_data_v3(self, img_folder, aid):
        # momo_style = True # debug

        ann = self.anno.anns[aid]  # get one annotation.
        # Load image
        img_name = self.anno.imgs[ann['image_id']]['file_name']
        path = os.path.join(img_folder, img_name)
        image = cv2.imread(path)

        joints = []
        # segmentation = []

        # Load landmark and apply square cropping for image
        if 'keypoints' in ann:
            joints = ann['keypoints']  
        # if 'segmentation' in ann:
        #     segmentation = ann['segmentation']
        bbox = ann['bbox']   # bbox
        landmark = np.array(joints).reshape([-1,3]) #  coco format.

        # Convert all (-1, -1) to (0, 0)
        for i in range(landmark.shape[0]):
            if landmark[i][0] == -1 and landmark[i][1] == -1:
                landmark[i, :] = [0, 0]

        # Generate visibility mask
        # visible = inside image + not occluded by simulated rectangle
        # (see BlazePose paper for more detail)
        visibility = np.ones((landmark.shape[0]), dtype=int)
        vis_index = np.where(landmark[:,-1]>0)
        landmark[vis_index,-1] = 1
        visibility = landmark[:,-1]
    
        for i in range(len(visibility)):
            if 0 > landmark[i][0] or landmark[i][0] >= image.shape[1] \
                or 0 > landmark[i][1] or landmark[i][1] >= image.shape[0]:
                visibility[i] = 0

        # if momo_style:
        # recostruct the landmarks.
        np_landmark = np.zeros([landmark.shape[0],5])
        np_landmark[:,:2] = landmark[:,:2]  # x,y
        np_landmark[:,2] = landmark[:,0]  #z=x
        np_landmark[:,3] = visibility
        np_landmark[:,4] = visibility
        # 
        # scale = [1.,1.]
        bbox= self._compute_bbox(np_landmark[-2],np_landmark[-1]) # square box
        rotation = get_rotation(np_landmark[-2, :2], np_landmark[-1, :2])
        ad_bbox = adjust_bbox(bbox, rotation)
        trans = get_transform_matrix(ad_bbox, rotation, [self.input_size[0], self.input_size[1]])
        image = cv2.warpPerspective(image, trans, (self.input_size[1], self.input_size[0]),
                                        flags=cv2.INTER_LINEAR)  # padding zeros
        landmark, visibility = affine_joints(np_landmark, visibility, trans, [self.input_size[0], self.input_size[1]])
        
        # if len(segmentation) > 0:
        #     segmentation = affine_segmentation(segmentation, trans)
        
        # assign 
        bbox = ad_bbox

        # augmentation.
        # Horizontal flip
        # and update the order of landmark points
        if self.random_flip and random.choice([0, 1]):
            image = cv2.flip(image, 1)

            # Mark missing keypoints
            missing_idxs = []
            for i in range(landmark.shape[0]):
                if landmark[i, 0] == 0 and landmark[i, 1] == 0:
                    missing_idxs.append(i)

            # Flip landmark
            landmark[:, 0] = self.input_size[0] - landmark[:, 0]

            # Restore missing keypoints
            for i in missing_idxs:
                landmark[i, 0] = 0
                landmark[i, 1] = 0

            # Change the indices of landmark points and visibility
            if self.symmetry_point_ids is not None:
                for p1, p2 in self.symmetry_point_ids:
                    l = landmark[p1, :].copy()
                    landmark[p1, :] = landmark[p2, :].copy()
                    landmark[p2, :] = l

        if self.augment:
            image, landmark[:,:2] = augment_img(image, landmark[:,:2])

        # Random occlusion
        # (see BlazePose paper for more detail)
        if self.augment and random.random() < 0.2:
            # landmark = landmark.reshape(-1, 2)
            image, visibility = random_occlusion(image, landmark[:,:2], visibility=visibility,
                                                 rect_ratio=((0.2, 0.5), (0.2, 0.5)), rect_color="random")

        visibility = np.array(visibility)
        visibility = visibility.reshape((landmark.shape[0]))
        landmark[:,3] = visibility
        landmark[:,4] = visibility
        
        # Generate heatmap
        gtmap = None
        if self.output_heatmap:
            gtmap_kps = landmark[:,:2].copy()
            gtmap_kps[:, :2] = (np.array(gtmap_kps[:, :2]).astype(float)
                                * np.array(self.heatmap_size) / np.array(self.input_size)).astype(int)
            gtmap = gen_gt_heatmap(
                gtmap_kps,visibility, self.heatmap_sigma, self.heatmap_size)
            # gtmap = np.clip(np.sum(gtmap, axis=2, keepdims=True), None, 1)

        # Uncomment following lines to debug augmentation
        if self.debug:
            draw = visualize_keypoints(image, landmark, visibility, text_color=(0,0,255))
            cv2.imwrite('/home/hades/dongmi_projects/tf-blazepose/output/gt_{}.jpg'.format(aid),draw)
        # if self.output_heatmap:
        #     cv2.imwrite('/home/hades/dongmi_projects/tf-blazepose/output/gt_{}.jpg'.format(aid),gtmap.sum(axis=2)*255)
        #     cv2.imshow("gtmap", gtmap.sum(axis=2))
        # cv2.waitKey(0)

        return image,landmark[:,:3],gtmap,[],0
    def _compute_bbox(self, center,top_head):
        # center = joints[-2]
        # top_head = joints[-1]
        radius = math.sqrt((center[0] - top_head[0]) ** 2 + (center[1] - top_head[1]) ** 2)
        w = radius * 2
        h = radius * 2
        xmin = center[0] - w / 2
        ymin = center[1] - h / 2
        return [xmin, ymin, w, h]