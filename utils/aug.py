import numpy as np
import cv2
import math


def rotate_image(image, label_box_list=[], angle=90, color=(0, 0, 0), img_scale=1.0):
    """
    rotate with angle, background filled with color, default black (0, 0, 0)
    label_box = (cls_type, box)
    box = [x0, y0, x1, y1, x2, y2, x3, y3]
    """
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise),
    # then grab the sine and cosine (i.e., the rotation components of the matrix)
    # if angle < 0, counterclockwise rotation; if angle > 0, clockwise rotation
    # 1.0 - scale, to adjust the size scale (image scaling parameter), recommended 0.75
    height_ori, width_ori = image.shape[:2]
    x_center_ori, y_center_ori = (width_ori // 2, height_ori // 2)

    rotation_matrix = cv2.getRotationMatrix2D((x_center_ori, y_center_ori), angle, img_scale)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # compute the new bounding dimensions of the image
    width_new = int((height_ori * sin) + (width_ori * cos))
    height_new = int((height_ori * cos) + (width_ori * sin))

    # adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (width_new / 2) - x_center_ori
    rotation_matrix[1, 2] += (height_new / 2) - y_center_ori

    # perform the actual rotation and return the image
    # borderValue - color to fill missing background, default black, customizable
    image_new = cv2.warpAffine(image, rotation_matrix, (width_new, height_new), borderValue=color)

    # each point coordinates
    angle = angle / 180 * math.pi
    box_rot_list = cal_rotate_box(label_box_list, angle, (x_center_ori, y_center_ori),
                                  (width_new // 2, height_new // 2))
    box_new_list = []
    for cls_type, box_rot in box_rot_list:
        for index in range(len(box_rot) // 2):
            box_rot[index * 2] = int(box_rot[index * 2])
            box_rot[index * 2] = max(min(box_rot[index * 2], width_new), 0)
            box_rot[index * 2 + 1] = int(box_rot[index * 2 + 1])
            box_rot[index * 2 + 1] = max(min(box_rot[index * 2 + 1], height_new), 0)
        box_new_list.append((cls_type, box_rot))

    image_with_boxes = [image_new, box_new_list]
    return image_with_boxes


def cal_rotate_box(box_list, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    box_list_new = []
    for (cls_type, box) in box_list:
        box_new = []
        for index in range(len(box) // 2):
            box_new.extend(cal_rotate_coordinate(box[index * 2], box[index * 2 + 1], angle, ori_center, new_center))
        label_box = (cls_type, box_new)
        box_list_new.append(label_box)
    return box_list_new


def cal_rotate_coordinate(x_ori, y_ori, angle, ori_center, new_center):
    # box = [x0, y0, x1, y1, x2, y2, x3, y3]
    # image_shape - [width, height]
    x_0 = x_ori - ori_center[0]
    y_0 = ori_center[1] - y_ori
    x_new = x_0 * math.cos(angle) - y_0 * math.sin(angle) + new_center[0]
    y_new = new_center[1] - (y_0 * math.cos(angle) + x_0 * math.sin(angle))
    return (x_new, y_new)