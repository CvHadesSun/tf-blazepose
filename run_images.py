import argparse
from genericpath import exists
import importlib
import json
import cv2
import os
import numpy as np
from glob import glob
from src.utils.heatmap import find_keypoints_from_heatmap
from src.utils.visualizer import visualize_keypoints
import tensorflow as tf
import copy
from tqdm import tqdm

for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)


def postLandmarks(coordinates, input_shape):
    # return self._heatmap2Coordinates()
    np_coordinates = np.squeeze(coordinates).reshape(-1, 5)
    norm_joints = np.zeros_like(np_coordinates)
    norm_joints[:, 0] = np_coordinates[:, 0] / input_shape[0]
    norm_joints[:, 1] = np_coordinates[:, 1] / input_shape[1]
    norm_joints[:, 2] = np_coordinates[:, 2] / input_shape[0]
    # norm_joints[:, 3] = 1.0 / (1 + np.exp(-np_coordinates[:, 3]))
    # norm_joints[:, 4] = 1.0 / (1 + np.exp(-np_coordinates[:, 4]))

    return norm_joints


def draw2DJoint(image, joints):
    img = copy.copy(image)
    # w= 1080
    # h = 1920
    # print(joints)

    # img = cv2.resize(image,(w,h))
    joints[:, 0] = joints[:, 0] * image.shape[1]
    joints[:, 1] = joints[:, 1] * image.shape[0]
    # joints[:, 0] = joints[:, 0]
    # joints[:, 1] = joints[:, 1]

    for i in range(joints.shape[0]):
        x, y = joints[i]
        if x > 0 and y > 0:
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)

    # print(img.shape)
    # cv2.imwrite('./test1.jpg', img)
    # cv2.imshow('show',img)
    # cv2.waitKey()
    return img


parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--conf_file', default="configs/momo/momo_config.json",
    help='Configuration file')
parser.add_argument(
    '-m',
    '--model', default="models/output",
    help='Path to model')
parser.add_argument(
    '-confidence',
    '--confidence',
    default=0.05,
    help='Confidence for heatmap point')
parser.add_argument(
    '-images',
    '--images', default="images/DSCF0698.JPG",
    help='Path to video file')

args = parser.parse_args()

# Webcam
# if args.video == "webcam":
#     args.video = 0

confth = float(args.confidence)

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Load model
trainer = importlib.import_module("src.trainers.{}".format(config["trainer"]))

model = trainer.load_model(config, args.model)

# Dataloader
datalib = importlib.import_module("src.data_loaders.{}".format(config["data_loader"]))
DataSequence = datalib.DataSequence

imgs_queue = []
if os.path.isdir(args.images):  # image dir
    extend_png = glob(os.path.join(args.images, '*.png'))
    extend_jpg = glob(os.path.join(args.images, '*.jpg'))
    imgs_queue += extend_png
    imgs_queue += extend_jpg
else:  # single image
    imgs_queue.append(args.images)

# cap = cv2.VideoCapture(args.video)
# cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
for item in tqdm(imgs_queue):
    try:
        origin_frame = cv2.imread(item)
    except:
        continue

    img = origin_frame
    img_name = os.path.basename(item)
    input_x = DataSequence.preprocess_images_v2(np.array([img]))

    ld_3d, heatmap = model.predict(input_x)

    norm_landmarks = postLandmarks(ld_3d, [256, 256])

    vis_img = draw2DJoint(img, norm_landmarks[:33, :2])
    # cv2.imshow("tf-result", vis_img)
    # cv2.waitKey()
    cv2.imwrite(os.path.join('output', img_name), vis_img)

    # heatmap_kps = find_keypoints_from_heatmap(heatmap)[0]
    # heatmap_kps = np.array(heatmap_kps)

    # # Scale heatmap keypoint
    # heatmap_stride = np.array([config["model"]["im_width"] / config["model"]["heatmap_width"],
    #                         config["model"]["im_height"] / config["model"]["heatmap_height"]], dtype=float)
    # heatmap_kps[:, :2] = heatmap_kps[:, :2] * scale * heatmap_stride

    # # Scale regression keypoint
    # regress_kps = regress_kps.reshape((-1, 3))
    # regress_kps[:, :2] = regress_kps[:, :2] * np.array([origin_frame.shape[1], origin_frame.shape[0]])

    # # Filter heatmap keypoint by confidence
    # heatmap_kps_visibility = np.ones((len(heatmap_kps),), dtype=int)
    # for i in range(len(heatmap_kps)):
    #     if heatmap_kps[i, 2] < confth:
    #         heatmap_kps[i, :2] = [-1, -1]
    #         heatmap_kps_visibility[i] = 0

    # regress_kps_visibility = np.ones((len(regress_kps),), dtype=int)
    # for i in range(len(regress_kps)):
    #     if regress_kps[i, 2] < 0.5:
    #         regress_kps[i, :2] = [-1, -1]
    #         regress_kps_visibility[i] = 0

    # edges = [[0,1,2,3,4,5,6]]

    # draw = origin_frame.copy()
    # draw = visualize_keypoints(draw, regress_kps[:, :2], visibility=regress_kps_visibility, edges=edges, point_color=(0, 255, 0), text_color=(255, 0, 0))
    # draw = visualize_keypoints(draw, heatmap_kps[:, :2], visibility=heatmap_kps_visibility, edges=edges, point_color=(0, 255, 0), text_color=(0, 0, 255))
    # cv2.imshow('Result', draw)

    # heatmap = np.sum(heatmap[0], axis=2)
    # heatmap = cv2.resize(heatmap, None, fx=3, fy=3)
    # heatmap = heatmap * 1.5
    # cv2.imshow('Heatmap', heatmap)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
