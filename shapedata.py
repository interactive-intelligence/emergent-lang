
#import os
import numpy as np
import cv2

from collections import namedtuple
#from tqdm.notebook import tqdm


SHAPE_TYPES = [ 'square', 'circle', 'triangle' ]
SHAPE_COLORS = [ (255,0,0), (0,255,0), (0,0,255) ]


Shape = namedtuple('Shape', ['type', 'color'])


def draw_shape(im, shape, shape_size, pos, rot=0):
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot),  np.cos(rot)]])

    radius = shape_size / 2
    center = np.array(pos) + radius

    if shape.type == 'square':
        points = (np.array([
            rot_mat @ (-1, 0),
            rot_mat @ (0, 1),
            rot_mat @ (1, 0),
            rot_mat @ (0, -1),
        ]) * radius + center).round().astype(int)
        cv2.fillConvexPoly(im, points, shape.color)
    elif shape.type == 'circle':
        cv2.circle(im, tuple(center.round().astype(int)),
                   round(radius), shape.color, -1)
    elif shape.type == 'triangle':
        points = (np.array([
            rot_mat @ (1, 0),
            rot_mat @ (-0.5, 0.8660254),
            rot_mat @ (-0.5, -0.8660254),
        ]) * radius + center).round().astype(int)
        cv2.fillConvexPoly(im, points, shape.color)
    else:
        raise Exception(f'unsupported/invalid shape type "{shape.type}"')


def draw_shapes(im, shapes, shape_scale=0.2):
    im_size = im.shape[0]
    shape_size = im_size * shape_scale

    for shape in shapes:
        pos = np.random.rand(2) * (im_size-shape_size)
        rot = np.random.rand() * np.pi
        draw_shape(im, shape, shape_size, pos, rot)


def random_shape():
    return Shape(type=SHAPE_TYPES[np.random.randint(len(SHAPE_TYPES))],
                 color=SHAPE_COLORS[np.random.randint(len(SHAPE_COLORS))])


def select_shapes(max_shapes, min_shapes=1):
    count = np.random.randint(min_shapes, max_shapes+1)
    return [random_shape() for _ in range(count)]


def create_batch(batch_size, im_size, same_p=0.5, shape_scale=0.2, min_shapes=1, max_shapes=5):
    x1 = np.zeros((batch_size,im_size,im_size,3), np.uint8)
    x1_shapes = []
    x2 = np.zeros((batch_size,im_size,im_size,3), np.uint8)
    x2_shapes = []
    y = np.zeros(batch_size, np.float32)

    for i in range(batch_size):
        shapes1 = select_shapes(max_shapes, min_shapes=min_shapes)
        if np.random.rand() < same_p:
            shapes2 = shapes1
            y[i] = 1
        else:
            shapes2 = select_shapes(max_shapes)
            y[i] = 0

        draw_shapes(x1[i], shapes1, shape_scale)
        draw_shapes(x2[i], shapes2, shape_scale)
        x1_shapes.append(shapes1)
        x2_shapes.append(shapes2)

    return (x1, x1_shapes), (x2, x2_shapes), y


def to_pytorch_model_inputs(x1, x2, y):
    X = np.concatenate([x1, x2]) / 256
    X = X.transpose(0, 3, 1, 2)
    y = np.concatenate([y, y])

    return X, y

