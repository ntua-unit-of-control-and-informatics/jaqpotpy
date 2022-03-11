import numpy as np
from typing import Union, Tuple


def pad_array(x: np.ndarray,
              shape: Union[Tuple, int],
              fill: float = 0.0,
              both: bool = False) -> np.ndarray:
  """
  Pad an array with a fill value.
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
  x = np.pad(x, pad, mode='constant', constant_values=fill)
  return x