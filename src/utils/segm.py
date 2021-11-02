import json
import numpy as np
import cv2
from pycocotools import mask as COCOmask
from skimage import measure


def showMask(img_obj):
    img = cv2.imread(img_obj['fpath'])
    img_ori = img.copy()
    gtmasks = img_obj['gtmasks']
    n = len(gtmasks)
    print(img.shape)
    for i, mobj in enumerate(gtmasks):
        if not (type(mobj['mask']) is list):
            print("Pass a RLE mask")
            continue
        else:
            pts = np.round(np.asarray(mobj['mask'][0]))
            pts = pts.reshape(pts.shape[0] // 2, 2)
            pts = np.int32(pts)
            color = np.uint8(np.random.rand(3) * 255).tolist()
            cv2.fillPoly(img, [pts], color)
    cv2.addWeighted(img, 0.5, img_ori, 0.5, 0, img)
    cv2.imshow("Mask", img)
    cv2.waitKey(0)


def get_seg(height, width, seg_ann):
    label = np.zeros((height, width, 1))
    if type(seg_ann) == list or type(seg_ann) == np.ndarray:
        for s in seg_ann:
            poly = np.array(s, np.int).reshape(len(s) // 2, 2)
            cv2.fillPoly(label, [poly], 1)
    else:
        if type(seg_ann['counts']) == list:
            rle = COCOmask.frPyObjects([seg_ann], label.shape[0], label.shape[1])
        else:
            rle = [seg_ann]
        # we set the ground truth as one-hot
        m = COCOmask.decode(rle) * 1
        label[label == 0] = m[label == 0]
    return label[:, :, 0]


def plot_contours(img, pts, color=(0, 0, 255)):
    '''xyxy mode'''
    pts = np.flip(pts)
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(img, (x, y), 1, color=color)
    return img


def polygons2binarymask(polygons, mask_shape):
    '''xyxy mode.
    polygons: list.'''

    width, height = mask_shape
    _polygons = [np.array(p) for p in polygons]
    rles = COCOmask.frPyObjects(_polygons, height, width)
    rle = COCOmask.merge(rles)
    mask = COCOmask.decode(rle)
    return mask


def binarymask2polygons(mask):
    '''return xyxy mode polygons(list)'''
    polygons = measure.find_contours(mask, 0.5)
    _polygons = []
    for p in polygons:
        size_p = p.shape[0]
        if size_p > 10:
            _p = np.flip(p)
            _polygons.append(_p.reshape(-1).tolist())

    return _polygons


def yx2xy(polygons):
    _polygons = []
    for p in polygons:
        _p = np.array(p).reshape(-1, 2)
        _polygons.append(np.flip(_p).reshape(-1).tolist())
    return _polygons


def xy2yx(polygons):
    _polygons = []
    for p in polygons:
        _p = np.array(p).reshape(-1, 2)
        _polygons.append(np.flip(_p).reshape(-1).tolist())
    return _polygons


def affine_segmentation(polygons, trans):
    # 
    xy_polygons = polygons
    # 2.affine tranform
    affined_polygons = []

    for p in xy_polygons:
        # if len(p)//2 >= 10:
        _p = np.array(p, dtype=np.float32).reshape(-1, 2)  # [x,y]
        _ones = np.ones([_p.shape[0], 1], dtype=np.float32)
        hstack_p = np.hstack([_p, _ones])
        hstack_p = np.dot(trans, hstack_p.T).T
        affined_polygons.append(hstack_p[:, :2].reshape(-1).tolist())
        # else:
        #     continue
    # yx_affined_polygons = xy2yx(affined_polygons)
    return affined_polygons
