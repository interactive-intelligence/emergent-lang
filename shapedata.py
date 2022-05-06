
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


#def pick_random_color():
    #return tuple(random.randint(0, 255) for _ in range(3))


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

def demo_dataset(data, shape=(4, 3), pad_value=100, pad_width=6, sep_width=3, **kwargs):
    (x1, x1_shapes), (x2, x2_shapes), y = data.create_batch(**kwargs)
    size = shape[0] * shape[1]

    x = np.concatenate([
        x1[:size],
        np.zeros((size, x1.shape[1], sep_width, 3), np.uint8) + pad_value +
        y.astype(np.uint8)[:size].reshape(-1, 1, 1, 1) * (255 - pad_value),
        x2[:size],
    ], axis=2)

    x = make_grid(x, shape=shape, pad_value=pad_value, pad_width=pad_width)
    return Image.fromarray(x)


class NewShapeData():
    def __init__(self, batch_size:int, im_size:int,
                 shape_scale:float=0.2, outline=None, alec_mode:bool=True,
                 min_shapes:int=1, max_shapes:int=3,
                 shape_types=SHAPE_TYPES, shape_colors=SHAPE_COLORS,
                 max_tries=1000):
        self.batch_size = batch_size
        self.im_size = im_size
        self.shape_scale = shape_scale
        self.outline = outline
        self.alec_mode = alec_mode
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes

        self.shape_types = shape_types
        self.shape_colors = shape_colors

        self.max_tries = max_tries

    def create_batch(self, same_p:float=0.5, shapes:List[Shape]=None, **kwargs):
        image_shape = (self.batch_size, self.im_size, self.im_size, 3)
        x1 = np.zeros(image_shape, np.uint8)
        x2 = np.zeros(image_shape, np.uint8)
        y = np.zeros(self.batch_size, np.float32)
        x1_shapes = []
        x2_shapes = []

        for i in range(self.batch_size):
            if shapes is None:
                y[i] = (random.random() < same_p)
                shapes1, shapes2 = self.select_pair(y[i], **kwargs)
            else:
                y[i] = 1
                shapes1 = shapes2 = shapes

            draw_shapes(x1[i], shapes1, self.shape_scale, outline=self.outline)
            draw_shapes(x2[i], shapes2, self.shape_scale, outline=self.outline)
            x1_shapes.append(shapes1)
            x2_shapes.append(shapes2)

        return (x1, x1_shapes), (x2, x2_shapes), y

    def select_pair(self, same, **kwargs):
        for _ in range(self.max_tries):
            if same:
                shapes1 = shapes2 = self.select_shapes_def(**kwargs)
            else:
                shapes1 = self.select_shapes_def(**kwargs)
                shapes2 = self.select_shapes_def(**kwargs)
                if shapes1 == shapes2:
                    continue

            if self.filter_pair(shapes1, shapes2, **kwargs):
                shapes1 = self.map_shapes_def(shapes1, **kwargs)
                shapes2 = self.map_shapes_def(shapes2, **kwargs)
                return shapes1, shapes2

        raise Exception("reached max number of tries for generating pair")

    def select_shapes_def(self, **kwargs):
        count = random.randint(self.min_shapes, self.max_shapes)
        return [self.select_shape(**kwargs) for _ in range(count)]

    def select_shape(self, **kwargs):
        for _ in range(self.max_tries):
            shape = Shape(type=random.choice(self.shape_types),
                         color=random.choice(self.shape_colors))
            if self.filter_shape(shape, **kwargs):
                return shape

        raise Exception("reached max number of tries for generating pair")

    def filter_shape(self, shape, **kwargs):
        return True

    def filter_pair(self, shapes1, shapes2, **kwargs):
        return self.filter_shapes_def(shapes1, **kwargs) and self.filter_shapes_def(shapes2, **kwargs)

    def filter_shapes_def(self, shapes, **kwargs):
        return True

    def map_shapes_def(self, shapes, **kwargs):
        return shapes


AlecDef = namedtuple('AlecDef', ['count', 'type', 'color'])
class NewAlecShapeData(NewShapeData):

    def select_shapes_def(self, **kwargs):
        count = random.randint(self.min_shapes, self.max_shapes)
        type = random.choice(self.shape_types)
        color = random.choice(self.shape_colors)
        return AlecDef(count, type, color)

    def map_shapes_def(self, shapes, **kwargs):
        return [Shape(shapes.type, shapes.color)] * shapes.count


class NewOODShapeData(NewAlecShapeData):
    def __init__(self, *args, exclude=AlecDef(None, 'square', (255,0,0)), **kwargs):
        super().__init__(*args, **kwargs)

        def coerce(ex_param):
            if ex_param is None:
                return None
            elif (type(ex_param) is list) or (type(ex_param) is set):
                return list(ex_param)
            else:
                return [ex_param]

        exclude = exclude if type(exclude) in {list, set} else [exclude]

        self.exclude = []
        for ex_count, ex_type, ex_color in exclude:
            self.exclude.append((coerce(ex_count), coerce(ex_type), coerce(ex_color)))


    def filter_pair(self, shapes1, shapes2, ood=False, **kwargs):
        return ood ^ super().filter_pair(shapes1, shapes2, **kwargs)

    def filter_shapes_def(self, shapes, **kwargs):
        for ex_count, ex_type, ex_color in self.exclude:
            if ((ex_count is None or shapes.count in ex_count) and
                    (ex_type is None or shapes.type in ex_type) and
                    (ex_color is None or shapes.color in ex_color)):
                return False
        return True



class NewExcludeShapeData(NewShapeData):
    """Shape data which enables "holding out" a certain shape or combination

    Args:
        exclude_shapes (Set[Shape]): Shapes to exclude from generation
        exclude_lists (List[List[Shape]]): Combinations of shapes to exclude from generation
    """

    def __init__(self, *args, exclude_shapes:set={}, exclude_lists:list=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.exclude_shapes = exclude_shapes
        self.exclude_lists = {tuple(sorted(shapes)) for shapes in exclude_lists}

    def filter_shapes_def(self, shapes, **kwargs):
        return tuple(sorted(shapes)) not in self.exclude_lists

    def filter_shape(self, shape, **kwargs):
        return shape not in self.exclude_shapes




class ShapeData():
    """Simple shape data source of image pairs with same/diff labels

    The parameters for constructor define what images generated by this data
    source look like. TODO: document me! (describe dataset, task)

    Args:
        batch_size: Number of image pairs to create at a time
        im_size: Size of the generated images
        shape_scale: Fraction of (shape size)/(image size)
        outline: Outline color for shapes
        alec_mode: Only one category of shape per image
        min_shapes: Minimum number of shapes in each image
        max_shapes: Maximum number of shapes in each image
        shape_types: Collection of possible shape type strings
        shape_colors: Collection of possible shape colors

    """

    def __init__(self, batch_size:int, im_size:int,
                 shape_scale:float=0.2, outline=None, alec_mode:bool=True,
                 min_shapes:int=1, max_shapes:int=5,
                 shape_types=SHAPE_TYPES, shape_colors=SHAPE_COLORS):
        self.batch_size = batch_size
        self.im_size = im_size
        self.shape_scale = shape_scale
        self.outline = outline
        self.alec_mode = alec_mode
        self.min_shapes = min_shapes
        self.max_shapes = max_shapes

        self.shape_types = shape_types
        self.shape_colors = shape_colors

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
        if self.alec_mode:
            return [self.select_shape()] * count

        return [self.select_shape() for _ in range(count)]

    def select_shape(self):
        """Utility function to select a single shape. """
        shape_type = random.choice(self.shape_types)
        shape_color = random.choice(self.shape_colors)
        return Shape(type=shape_type, color=shape_color)



class OODShapeData(ShapeData):
    def __init__(self, *args, exclude=(None, 'square', (255,0,0)), **kwargs):
        super().__init__(*args, **kwargs)

        ex_count, ex_type, ex_color = exclude

        def coerce(ex_param):
            if ex_param is None:
                return None
            elif (type(ex_param) is list) or (type(ex_param) is set):
                return list(ex_param)
            else:
                return [ex_param]

        self.exclude = coerce(ex_count), coerce(ex_type), coerce(ex_color)
        self.ood = False
        

    def create_batch(self, ood=False, **kwargs):
        self.ood = ood
        return super().create_batch(**kwargs)


    def select_pair(self, same, max_tries=1000):
        shapes1 = self.select_shape_list(ood=self.ood)
        if same:
            return shapes1, shapes1

        for _ in range(max_tries):
            if self.ood:
                count, type, color = self.select_alec_def()
                shapes2 = [Shape(type=type, color=color)] * count
            else:
                shapes2 = self.select_shape_list(ood=self.ood)

            if shapes1 != shapes2:
                return shapes1, shapes2

        raise Exception("reached max number of tries for generating pair")

    def select_shape_list(self, ood=False, max_tries=1000):
        if ood:
            count, type, color = self.select_ood_def()
            return [Shape(type=type, color=color)] * count

        alec_def = self.select_alec_def()
        for _ in range(max_tries):
            if self.is_excluded(*alec_def):
                alec_def = self.select_alec_def()
            else:
                count, type, color = alec_def
                return [Shape(type=type, color=color)] * count

        raise Exception("reached max number of tries for generating shape list")

    def select_ood_def(self):
        ex_count, ex_type, ex_color = self.exclude
        if ex_count is None:
            count = random.randint(self.min_shapes, self.max_shapes)
        else:
            count = random.choice(ex_count)
        type = random.choice(self.shape_types if ex_type is None else ex_type)
        color = random.choice(self.shape_colors if ex_color is None else ex_color)
        return count, type, color


    def select_alec_def(self):
        count = random.randint(self.min_shapes, self.max_shapes)
        type = random.choice(self.shape_types)
        color = random.choice(self.shape_colors)
        return count, type, color


    def is_excluded(self, count, type, color):
        ex_count, ex_type, ex_color = self.exclude
        return ((ex_count is None or count in ex_count) and
                (ex_type is None or type in ex_type) and
                (ex_color is None or color in ex_color))




class StrongAlecModeShapeData(ShapeData):
    """Alec's suggestions for changing the data

    Base behavior is only one category of shape per image. Strong Alec mode
    means only one aspect of the shape definitions can be different in a pair.

    'Not much how bout you' - Alec
    """

    def __init__(self, *args, weak=True, strong=True,
                 smart=True, dumb=False, **kwargs):
        super().__init__(*args, alec_mode=True, **kwargs)
        #self.strong = strong

    #def select_shape_list(self):
        #count = random.randint(self.min_shapes, self.max_shapes)
        #return [self.select_shape()] * count

    def select_pair(self, same, max_tries=1000):
        #if not self.strong:
            #return super().select_pair(same)

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

    def __init__(self, *args, exclude_shapes:set={}, exclude_lists:list=[], **kwargs):
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


class AlecOODShapeData(ShapeData):
    """
    An Out of Distribution (OOD) task adhering to Alec Mode. Conceptually, this is
    a merger of AlecModeShapeData and ExcludeShapeData classes used for testing
    model understanding of language. Represents 'reasonable extrapolation'.
    
    exclude_shapes: a set of tuples in form (shape_name, color). For instance,
        {('square', (255, 0, 0)), ('circle', (0, 255, 0))} excludes red squares
        and green circles from the in-distribution training dataset. The excluded
        shapes constitute the out-of-distribution dataset.
    id_object_counts: valid object counts for in-distribution training dataset.
        For instance, [1, 2, 3] specifies that there are either 1, 2, or 3 objects
        in each in-distribution scene.
    ood_object_counts: valid object counts for out-of-distribution dataset.
        For instance, [4, 5] specifies that there are either 4 or 5 objects in each
        out of distribution scene. This can be set to be a subset of id_object_counts
        if not focusing on extrapolating to new object counts.
    """
    
    def __init__(self, *args, 
                 exclude_shapes:set={}, 
                 id_object_counts:list=[],
                 ood_object_counts:list=[],
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.exclude_shapes = exclude_shapes
        self.id_object_counts = id_object_counts
        self.ood_object_counts = ood_object_counts
    
    def select_shape_list(self, color_spec:list=[]):
        '''
        Returns a list of selected shapes with in-distribution properties
        '''
        return [self.select_shape(color_spec)] * np.random.choice(self.id_object_counts)

    def select_shape(self, color_spec:list=[], max_tries=1000):
        '''
        Selects a shape with in-distribution properties
        '''
        out = super().select_shape()
        for _ in range(max_tries):
            if out in self.exclude_shapes or (color_spec!=[] and out.color not in color_spec):
                out = super().select_shape()
            else:
                return out

        raise Exception("reached max number of tries for generating shape")
        
    def select_shape_list_ood(self):
        '''
        Returns a list of selected shapes with out-of-distribution properties
        '''
        return [self.select_shape_ood()] * np.random.choice(self.ood_object_counts)
    
    def select_shape_ood(self, max_tries=1000):
        '''
        Selects a shape with out-of-distribution properties
        '''
        out = super().select_shape()
        for _ in range(max_tries):
            if out not in self.exclude_shapes:
                out = super().select_shape()
            else:
                return out

        raise Exception("reached max number of tries for generating shape")
        
    def select_pair_ood(self, same, color_spec:list=[], max_tries=1000):
        '''
        Selects a pair of objects such that at least one adheres to OOD properties
        '''
        shapes1 = self.select_shape_list_ood()
        if same:
            shapes2 = self.select_shape_list_ood()
        else:
            shapes2 = self.select_shape_list(color_spec)
        return shapes1, shapes2
        
    def create_batch_ood(self, same_p:float=0.5, color_spec:list=[]):
        """
        Identical to create_batch_ood, except draws batch using OOD properties
        Set color_spec to generate 'other' samples that have a specified set of
        colors. Useful to prevent models from gaming metrics by color separation.
        e.g. if the out of distribution shape is a red square, you can set
        color_spec=[(255,0,0)] such that, for false labels, it is only compared
        against red shapes.
        """
        image_shape = (self.batch_size, self.im_size, self.im_size, 3)
        x1 = np.zeros(image_shape, np.uint8)
        x2 = np.zeros(image_shape, np.uint8)
        y = np.zeros(self.batch_size, np.float32)
        x1_shapes = []
        x2_shapes = []

        for i in range(self.batch_size):
            y[i] = (random.random() < same_p)
            shapes1, shapes2 = self.select_pair_ood(y[i], color_spec)
            draw_shapes(x1[i], shapes1, self.shape_scale, outline=self.outline)
            draw_shapes(x2[i], shapes2, self.shape_scale, outline=self.outline)
            x1_shapes.append(shapes1)
            x2_shapes.append(shapes2)

        return (x1, x1_shapes), (x2, x2_shapes), y
