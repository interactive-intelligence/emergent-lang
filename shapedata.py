
import random
import numpy as np
import cv2
from PIL import Image

from collections import namedtuple

from typing import List


SQUARE_POINTS = np.array([
    (-1, 0), (0, 1), (1, 0), (0, -1),
])
TRIANGLE_POINTS = np.array([
    (1, 0), (-0.5, 0.86602), (-0.5, -0.86602),
])
STAR_POINTS = np.array([
    ( 0.4, 0.0),
    ( 0.80901, 0.58778),
    ( 0.12360, 0.38042),
    (-0.30901, 0.95105),
    (-0.32360, 0.23511),
    (-1.0, 0.0),
    (-0.32360, -0.23511),
    (-0.30901, -0.95105),
    ( 0.12360, -0.38042),
    ( 0.80901, -0.58778),
])
HEXAGON_POINTS = np.array([
    (1.00000, 0.00000),
    (0.50000, 0.86603),
    (-0.50000, 0.86603),
    (-1.00000, 0.00000),
    (-0.50000, -0.86603),
    (0.50000, -0.86603),
])

POLYGONS = {
    # name: (points, convex?)
    'square': (SQUARE_POINTS, True),
    'triangle': (TRIANGLE_POINTS, True),
    'hexagon': (HEXAGON_POINTS, True),
    'star': (STAR_POINTS, False),
}



SHAPE_TYPES = [ 'square', 'circle', 'triangle' ]
SHAPE_COLORS = [ (255,0,0), (0,255,0), (0,0,255) ]


Shape = namedtuple('Shape', ['type', 'color'])


def draw_shape(im, shape, shape_size, pos, rot=0, outline=None):
    rot_mat = np.array([[np.cos(rot), -np.sin(rot)],
                        [np.sin(rot),  np.cos(rot)]])

    radius = shape_size / 2
    center = np.array(pos) + radius

    if shape.type in POLYGONS:
        points_np, convex = POLYGONS[shape.type]
        points = ((points_np @ rot_mat.T) * radius + center).round().astype(int)
        if convex:
            cv2.fillConvexPoly(im, points, shape.color)
        else:
            cv2.fillPoly(im, [points], shape.color)

        if outline is not None:
            cv2.polylines(im, [points], True, outline)
    elif shape.type == 'circle':
        center = tuple(center.round().astype(int))
        cv2.circle(im, center, round(radius), shape.color, -1)
        if outline is not None:
            cv2.circle(im, center, round(radius), outline)
    else:
        raise Exception(f'unsupported/invalid shape type "{shape.type}"')


def draw_shapes(im, shapes, shape_scale=0.2, outline=None):
    im_size = im.shape[0]
    shape_size = im_size * shape_scale

    for shape in shapes:
        pos = np.random.rand(2) * (im_size-shape_size)
        rot = np.random.rand() * np.pi
        draw_shape(im, shape, shape_size, pos, rot, outline)


def pick_random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def to_pytorch_inputs(x1, x2, y, device=None):
    import torch

    X = np.concatenate([x1, x2]) / 256
    X = X.transpose(0, 3, 1, 2)
    y = np.concatenate([y, y])

    X = torch.from_numpy(X).to(torch.float32).to(device)
    y = torch.from_numpy(y).to(torch.float32).to(device)

    return X, y


def make_grid(im, shape=(4, 3), pad_value=100, pad_width=6):
    size = shape[0] * shape[1]
    im = im[:size].reshape(*shape, *im.shape[1:])

    im = np.concatenate([
        np.zeros((*shape, pad_width, im.shape[3], im.shape[4]), np.uint8) + pad_value,
        im,
        np.zeros((*shape, pad_width, im.shape[3], im.shape[4]), np.uint8) + pad_value,
    ], axis=2)

    im = np.concatenate([
        np.zeros((*shape, im.shape[2], pad_width, im.shape[4]), np.uint8) + pad_value,
        im,
        np.zeros((*shape, im.shape[2], pad_width, im.shape[4]), np.uint8) + pad_value,
    ], axis=3)

    im = np.concatenate(im, axis=1)
    im = np.concatenate(im, axis=1)

    return im

def demo_dataset(data, shape=(4, 3), pad_value=100, pad_width=6, sep_width=3):
    (x1, x1_shapes), (x2, x2_shapes), y = data.create_batch()
    size = shape[0] * shape[1]

    x = np.concatenate([
        x1[:size],
        np.zeros((size, x1.shape[1], sep_width, 3), np.uint8) + pad_value +
        y.astype(np.uint8)[:size].reshape(-1, 1, 1, 1) * (255 - pad_value),
        x2[:size],
    ], axis=2)

    x = make_grid(x, shape=shape, pad_value=pad_value, pad_width=pad_width)
    return Image.fromarray(x)


class ShapeData():
    """Simple shape data source of image pairs with same/diff labels

    The parameters for constructor define what images generated by this data
    source look like. TODO: document me! (describe dataset, task)

    Args:
        batch_size: Number of image pairs to create at a time
        im_size: Size of the generated images
        shape_scale: Fraction of (shape size)/(image size)
        outline: Outline color for shapes
        min_shapes: Minimum number of shapes in each image
        max_shapes: Maximum number of shapes in each image
        shape_types: List of possible shape type strings, randomly sampled
            for every generated shape
        shape_colors: Defines shape colors. Either a 0-argument function
            returning a ([0-255],[0-255],[0-255]) color tuple or a list.
            If a function, it's called to pick the color for each shape.
            If a list, it's randomly sampled for each shape.
    """

    def __init__(self, batch_size:int, im_size:int,
                 shape_scale:float=0.2, outline=None,
                 min_shapes:int=1, max_shapes:int=5,
                 shape_types=SHAPE_TYPES, shape_colors=SHAPE_COLORS,):
        self.batch_size = batch_size
        self.im_size = im_size
        self.shape_scale = shape_scale
        self.outline = outline
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes

        self.shape_types = shape_types
        if type(shape_colors) is list:
            self.shape_color_f = lambda: random.choice(shape_colors)
        else:
            self.shape_color_f = shape_colors

    def create_batch(self, same_p:float=0.5, shapes:List[Shape]=None):
        """Create a batch of image pairs along with same/diff labels.

        Args:
            same_p: Probability that images in an image pair are the same
            shapes: If provided, list of shapes to draw in images in batch.
                same_p is ignored in that case, as all images are the same.
                If None, randomly picks lists of shapes per image/pair.

        Returns:
            (x1, x1_shapes), (x2, x2_shapes), y
            x# (numpy uint8 arrays of shape [N, S, S, 3]): Generated images
            x#_shapes (List[List[Shape]]): Shapes present in each image
            y (numpy float32 array of shape [N]): labels per image pair
        """
        image_shape = (self.batch_size, self.im_size, self.im_size, 3)
        x1 = np.zeros(image_shape, np.uint8)
        x2 = np.zeros(image_shape, np.uint8)
        y = np.zeros(self.batch_size, np.float32)
        x1_shapes = []
        x2_shapes = []

        for i in range(self.batch_size):
            if shapes is None:
                y[i] = (random.random() < same_p)
                shapes1, shapes2 = self.select_pair(y[i])
            else:
                y[i] = 1
                shapes1 = shapes2 = shapes

            draw_shapes(x1[i], shapes1, self.shape_scale, outline=self.outline)
            draw_shapes(x2[i], shapes2, self.shape_scale, outline=self.outline)
            x1_shapes.append(shapes1)
            x2_shapes.append(shapes2)

        return (x1, x1_shapes), (x2, x2_shapes), y

    def select_pair(self, same, max_tries=1000):
        shapes1 = self.select_shape_list()
        if same:
            return shapes1, shapes1

        for _ in range(max_tries):
            shapes2 = self.select_shape_list()
            if shapes1 != shapes2:
                return shapes1, shapes2

        raise Exception("reached max number of tries for generating pair")

    def select_shape_list(self):
        """Utility function to select a list of shapes. """
        count = random.randint(self.min_shapes, self.max_shapes)
        return [self.select_shape() for _ in range(count)]

    def select_shape(self):
        """Utility function to select a single shape. """
        shape_type = random.choice(self.shape_types)
        shape_color = self.shape_color_f()
        return Shape(type=shape_type, color=shape_color)


class AlecModeShapeData(ShapeData):
    """Alec's suggestions for changing the data

    Base behavior is only one category of shape per image. Strong Alec mode
    means only one aspect of the shape definitions can be different in a pair.
    """

    def __init__(self, *args, weak=True, strong=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.strong = strong

    def select_shape_list(self):
        count = random.randint(self.min_shapes, self.max_shapes)
        return [self.select_shape()] * count

    def select_pair(self, same, max_tries=1000):
        if not self.strong:
            return super().select_pair(same)

        count = random.randint(self.min_shapes, self.max_shapes)
        shape_type, shape_color = self.select_shape()

        shapes1 = [Shape(type=shape_type, color=shape_color)] * count
        if same:
            return shapes1, shapes1

        shapes1_def = {'count':count, 'type':shape_type, 'color':shape_color}
        shapes2_def = {'count':count, 'type':shape_type, 'color':shape_color}

        for _ in range(max_tries):
            choice = random.choice(list(shapes2_def))
            count = random.randint(self.min_shapes, self.max_shapes)
            shape_type, shape_color = self.select_shape()
            replacements = {'count':count, 'type':shape_type, 'color':shape_color}
            shapes2_def[choice] = replacements[choice]
            if shapes1_def != shapes2_def:
                shapes2 = [Shape(type=shapes2_def['type'],
                                 color=shapes2_def['color'])] * shapes2_def['count']
                return shapes1, shapes2

        raise Exception("reached max number of tries for generating pair")


class ExcludeShapeData(ShapeData):
    """Shape data which enables "holding out" a certain shape or combination

    Args:
        exclude_shapes (Set[Shape]): Shapes to exclude from generation
        exclude_lists (List[List[Shape]]): Combinations of shapes to exclude from generation
    """

    def __init__(self, *args, exclude_shapes:set=None, exclude_lists:list=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.exclude_shapes = exclude_shapes
        self.exclude_lists = {tuple(sorted(shapes)) for shapes in exclude_lists}
    

    def select_shape_list(self, max_tries=1000):
        out = super().select_shape_list()
        for _ in range(max_tries):
            if tuple(sorted(out)) in self.exclude_lists:
                out = super().select_shape_list()
            else:
                return out

        raise Exception("reached max number of tries for generating shape list")


    def select_shape(self, max_tries=1000):
        out = super().select_shape()
        for _ in range(max_tries):
            if out in self.exclude_shapes:
                out = super().select_shape()
            else:
                return out

        raise Exception("reached max number of tries for generating shape")



