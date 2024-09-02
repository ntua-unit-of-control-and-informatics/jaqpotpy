import numpy as np
from typing import Union, Tuple
import os
import tempfile
import tarfile
import zipfile
from urllib.request import urlretrieve
from typing import Optional


def pad_array(
    x: np.ndarray, shape: Union[Tuple, int], fill: float = 0.0, both: bool = False
) -> np.ndarray:
    """Pad an array with a fill value.

    Parameters
    ----------
    x: np.ndarray
      A numpy array.
    shape: Tuple or int
      Desired shape. If int, all dimensions are padded to that size.
    fill: float, optional (default 0.0)
      The padded value.
    both: bool, optional (default False)
      If True, split the padding on both sides of each axis. If False,
      padding is applied to the end of each axis.

    Returns
    -------
    np.ndarray
      A padded numpy array

    """
    x = np.asarray(x)
    if not isinstance(shape, tuple):
        shape = tuple(shape for _ in range(x.ndim))
    pad = []
    for i in range(x.ndim):
        diff = shape[i] - x.shape[i]
        assert diff >= 0
        if both:
            a, b = divmod(diff, 2)
            b += a
            pad.append((a, b))
        else:
            pad.append((0, diff))
    pad = tuple(pad)  # type: ignore
    x = np.pad(x, pad, mode="constant", constant_values=fill)
    return x


def get_data_dir() -> str:
    """Get the DeepChem data directory.

    Returns
    -------
    str
      The default path to store DeepChem data. If you want to
      change this path, please set your own path to `DEEPCHEM_DATA_DIR`
      as an environment variable.

    """
    if "DEEPCHEM_DATA_DIR" in os.environ:
        return os.environ["DEEPCHEM_DATA_DIR"]
    return tempfile.gettempdir()


def download_url(url: str, dest_dir: str = get_data_dir(), name: Optional[str] = None):
    """Download a file to disk.

    Parameters
    ----------
    url: str
      The URL to download from
    dest_dir: str
      The directory to save the file in
    name: str
      The file name to save it as.  If omitted, it will try to extract a file name from the URL

    """
    if name is None:
        name = url
        if "?" in name:
            name = name[: name.find("?")]
        if "/" in name:
            name = name[name.rfind("/") + 1 :]
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    urlretrieve(url, os.path.join(dest_dir, name))


def untargz_file(file: str, dest_dir: str = get_data_dir(), name: Optional[str] = None):
    """Untar and unzip a .tar.gz file to disk.

    Parameters
    ----------
    file: str
      The filepath to decompress
    dest_dir: str
      The directory to save the file in
    name: str
      The file name to save it as.  If omitted, it will use the file name

    """
    if name is None:
        name = file
    tar = tarfile.open(name)
    tar.extractall(path=dest_dir)
    tar.close()


def unzip_file(file: str, dest_dir: str = get_data_dir(), name: Optional[str] = None):
    """Unzip a .zip file to disk.

    Parameters
    ----------
    file: str
      The filepath to decompress
    dest_dir: str
      The directory to save the file in
    name: str
      The directory name to unzip it to.  If omitted, it will use the file name

    """
    if name is None:
        name = file
    if dest_dir is None:
        dest_dir = os.path.join(get_data_dir, name)
    with zipfile.ZipFile(file, "r") as zip_ref:
        zip_ref.extractall(dest_dir)
