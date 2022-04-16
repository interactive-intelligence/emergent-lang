
import numpy as np
import pandas as pd
from PIL import Image

import torch 
import torchvision
from torchvision import transforms

import shapedata


def create_dataset(model, data, n=100, progress=None):
    device = next(model.parameters()).device
    shape_list = []
    enc_list = []
    img_list = []

    it = range(n)
    if progress is not None:
        it = progress(it)

    model.eval()
    for _ in it:
        (x1, x1_shapes), (x2, x2_shapes), y = data.create_batch()
        X, y = shapedata.to_pytorch_inputs(x1, x2, y, device=device)

        loss, pred, enc = model.forward(X)

        shape_list += x1_shapes+x2_shapes
        enc_list += list(map(tuple, enc.argmax(2).T.tolist()))
        img_list += [*map(Image.fromarray, x1), *map(Image.fromarray, x2)]

    return pd.DataFrame({
        'shapes': shape_list,
        'enc': enc_list,
        'images': img_list
    })


def encoding_heatmap(enc, vocab_size=4, seq_len=10):
    arr = np.zeros((vocab_size, seq_len), int)
    for encoding, count in enc.value_counts().items():
        arr[tuple(i for i in range(vocab_size)), encoding] += count
    return arr


def sample_images(im_series:pd.Series, n=64):
    images = im_series.sample(min(n, len(im_series)))
    im_t = torchvision.utils.make_grid(
        list(images.map(transforms.ToTensor())),
        padding=2, pad_value=0.5
    )
    return transforms.functional.to_pil_image(im_t)


def match_sequence(seq, target_seq):
    for a, b in zip(seq, target_seq):
        if b != '_' and a != b:
            return False
    return True


def contains_f(type=None, color=None):
    def f(shapes):
        return any(
            (shape.type == type or type is None) and
            (shape.color == color or color is None)
            for shape in shapes
        )
    return f

def extend_dataset(df, types=shapedata.SHAPE_TYPES,
                   colors={'red':(255,0,0),'green':(0,255,0),'blue': (0,0,255)}):
    df['num_shapes'] = df['shapes'].map(len)

    if types is not None:
        for type in types:
            df[f'has_{type}'] = df['shapes'].map(contains_f(type=type))
    if colors is not None:
        for color in colors:
            df[f'has_{color}'] = df['shapes'].map(contains_f(color=colors[color]))





