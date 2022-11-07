from jaqpotpy.datasets.dataset_base import ImageDataset
from typing import Iterable, Any, Optional, Callable
from PIL import Image
import os
import matplotlib.pyplot as plt
import inspect
import numpy as np
import pickle
from torch.utils.data import Dataset


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class TorchImageDataset(ImageDataset):
    """
    Reads images from a path and creates tensors ready for modelling. This is compatible with all transform
    methods provided by torchvision, as well as custom callable transformers.
    """

    def __init__(self, path=None, X: Optional[Iterable[Any]] = None, y: Optional[Iterable[Any]] = None,
                 featurizer: Callable = None, task: str = "regression", y_name: str = None,
                 images_name: str = None) -> None:
        super(ImageDataset, self).__init__(path=path, x_cols=images_name, y_cols=y_name)
        self._y = y_name
        self._x = X
        self._X = 'ImagePath'
        self.ys = y
        self._task = task
        self.featurizer: Callable = featurizer
        self.images = None
        self.data = None
        self.raw_img = None
        # self.create()

    def create(self):
        if not self.path:
            self.path = '/'
        if not self._x:
            # All files in the self.path with extentions defined in IMG_EXTENSIONS
            # will be selected as X
            self.images = [os.path.join(self.path, item) for item in os.listdir(self.path) if
                           item.endswith(IMG_EXTENSIONS)]
        else:
            self.images = [os.path.join(self.path, item) for item in self._x]

        self.raw_img = [default_loader(image) for image in self.images]

        if self.featurizer:
            self.data = tuple([(self.featurizer(image), y_i) for image, y_i in zip(self.raw_img, self.ys)])
        else:
            self.data = tuple([(image, y_i) for image, y_i in zip(self.raw_img, self.ys)])

        return self

    def show_image(self, idx, cmap=None):
        plt.imshow(self.raw_img[idx], cmap=cmap)
        plt.title('%i' % self.ys[idx])
        plt.show()

    def __get_X__(self):
        return self.images

    def __get_Y__(self):
        return self.ys

    def __get__(self):
        return self.df

    def __getitem__(self, idx):
        # print(self.df[self.X].iloc[idx].values)
        # print(type(self.df[self.X].iloc[idx].values))
        return self.data[idx]

    def __len__(self):
        return len(self.images)

    def __repr__(self) -> str:
        # args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        # args_names = [arg for arg in args_spec.args if arg != 'self']
        # args_info = ''
        # for arg_name in args_names:
        #   value = self.__dict__[arg_name]
        #   # for str
        #   if isinstance(value, str):
        #     value = "'" + value + "'"
        #   # for list
        return self.__class__.__name__