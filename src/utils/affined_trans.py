'''-cvhadessun:2021.8.16-'''
import cv2
import numpy as np
import math


def NormalizeRadians(angle_):
    '''referring to: transform the angle into [-pi,pi]
    mediapipe/calculators/util/detections_to_rects_calculators.cc
    mediapipe/calculators/util/detections_to_rects_calculators.h
    '''
    return angle_ - 2 * math.pi * math.floor((angle_ - (-math.pi)) / (2 * math.pi))


def get_rotation(center, point, target_angle_=90):
    '''referring to:
     mediapipe/calculators/util/detections_to_rects_calculators.cc
    mediapipe/calculators/util/detections_to_rects_calculators.h
    '''

    target_angle_ = target_angle_ * math.pi / 180.0
    vertical_dist = point[1] - center[1]
    horizontial_dist = point[0] - center[0]
    norm_angle = NormalizeRadians(target_angle_ - math.atan2(-vertical_dist, horizontial_dist))

    return norm_angle / math.pi * 180.0


def rotate_points(points, center, rot_rad):
    """
    :param points:  N*2
    :param center:  2
    :param rot_rad: scalar
    :return: N*2
    """
    rot_rad = rot_rad * np.pi / 180.0
    rotate_mat = np.array([[np.cos(rot_rad), -np.sin(rot_rad)],
                           [np.sin(rot_rad), np.cos(rot_rad)]])
    center = center.reshape(2, 1)
    points = points.T
    points = rotate_mat.dot(points - center) + center

    return points.T


def get_transform_matrix(bbox, rotation, input_shape):
    '''
    :param bbox:
    :param input_shape:
    :return: transform matrix [3x3]

    (xmin,ymin)----------(xmax,ymin)           (0,0)----------------(input_w,0)
        ｜                    ｜      rotation    ｜                      ｜
        ｜                    ｜     -------->    ｜                      ｜
        ｜                    ｜                  ｜                      ｜
    (xmin,ymax)----------(xmax,ymax)          (0,input_h)----------(input_w,input_h)
    '''
    xmin, ymin, w, h = bbox
    xmax = xmin + w
    ymax = ymin + h
    rect_points = np.array([[xmin, ymax],
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax]])

    rect_center = np.array([xmin + w * 0.5, ymin + h * 0.5])

    rot_rect_points = rotate_points(rect_points, rect_center, rotation).astype(np.float32)

    input_h, input_w = input_shape
    dst_corners = np.array([[0.0, input_h],
                            [0.0, 0.0],
                            [input_w, 0.0],
                            [input_w, input_h]]).astype(np.float32)

    projection_matrix = cv2.getPerspectiveTransform(rot_rect_points, dst_corners)

    return projection_matrix


def adjust_bbox(bbox, rotation, scales=[1.25, 1.25]):
    '''
    :param bbox: [x,y,w,h]
    :param rotation:  degree unit
    :return: adjucted bbox.
    '''
    # -config
    square_long = True
    x_shift = 0.
    y_shift = 0.
    x_scale = scales[0]
    y_scale = scales[1]
    x, y, w, h = bbox
    rot = rotation

    # rect_center = np.array([x + w * 0.5, y + h * 0.5])
    center_x = x + w * 0.5
    center_y = y + h * 0.5
    if rot == 0.:
        new_center_x = center_x + x_shift
        new_center_y = center_y + y_shift
    else:
        x_shift = w * x_shift * math.cos(rot) - h * y_shift * math.sin(rot)
        y_shift = w * x_shift * math.sin(rot) + h * y_shift * math.cos(rot)
        new_center_x = center_x + x_shift
        new_center_y = center_y + y_shift
    if square_long:
        long_side = max(w, h)
        w = long_side
        h = long_side
    else:
        short_side = min(w, h)
        w = short_side
        h = short_side
    w = w * x_scale
    h = h * y_scale
    x = new_center_x - w * 0.5
    y = new_center_y - h * 0.5
    new_bbox = np.array([x, y, w, h])
    return new_bbox


def affine_joints(joints,valid, trans, input_shape):
    '''
    affine tranform the joints into new coordinates sys with trans.
    :param joints:  [N,5 ] [x,y,z,vis,conf]
    :param trans: trans matrix [3x3]
    :return: new joints. [N,5]
    '''

    new_joints = joints.copy()
    # new_joints[:, -1] = 1
    tmp_joints = np.ones([joints.shape[0],3],dtype=np.float32)
    tmp_joints[:,:2] = new_joints[:,:2]
    tmp_joints = np.dot(trans, tmp_joints.T).T
    new_joints[:,:2] = tmp_joints[:,:2]
    # valid
    for i in range(joints.shape[0]):
        new_joints[i, 3] = ((new_joints[i, 0] >= 0) & (new_joints[i, 0] < input_shape[1]) & \
                            (new_joints[i, 1] >= 0) & (new_joints[i, 1] < input_shape[0])) * joints[i, 3]
        new_joints[i, 4] = new_joints[i, 3]  # [x,y,z,vis,conf]
        # valid[i] *= ((new_joints[i, 0] >= 0) & (new_joints[i, 0] < input_shape[1]) & \
        #                     (new_joints[i, 1] >= 0) & (new_joints[i, 1] < input_shape[0]))
    
    return new_joints,(valid>0)
