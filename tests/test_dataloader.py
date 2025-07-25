import os
from tempfile import TemporaryDirectory
import numpy as np
from PIL import Image
import torch

from dataloader import load_training, load_testing


def _create_dummy_dataset(root, domain, categories=2, images_per_cat=2):
    base = os.path.join(root, domain, 'images')
    for c in range(categories):
        cat_dir = os.path.join(base, f'cat{c}')
        os.makedirs(cat_dir, exist_ok=True)
        for i in range(images_per_cat):
            arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
            Image.fromarray(arr).save(os.path.join(cat_dir, f'{i}.jpg'))


def test_dataloaders_return_batches():
    with TemporaryDirectory() as tmp:
        _create_dummy_dataset(tmp, 'd1')
        train_loader = load_training(tmp, 'd1', batch_size=2)
        images, labels = next(iter(train_loader))
        assert images.shape == (2, 3, 227, 227)
        assert labels.shape == (2,)

        test_loader = load_testing(tmp, 'd1', batch_size=2)
        images, labels = next(iter(test_loader))
        assert images.shape == (2, 3, 227, 227)
        assert labels.shape == (2,)
