"""
Dataset base classes
"""
import numpy as np
import bisect
import warnings
from typing import Any, Dict, Iterable, Optional, Tuple, Union, cast
from jaqpotpy.descriptors.base_classes import Featurizer, MolecularFeaturizer
import os
import pandas as pd
import csv
import inspect
import functools
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)


T_co = TypeVar('T_co', covariant=True)


class Dataset(Generic[T_co]):
    r"""An abstract class representing a :class:`Dataset`.
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.
    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """
    functions: Dict[str, Callable] = {}

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py

    def __getattr__(self, attribute_name):
        if attribute_name in Dataset.functions:
            function = functools.partial(Dataset.functions[attribute_name], self)
            return function
        else:
            raise AttributeError("'{0}' object has no attribute '{1}".format(self.__class__.__name__, attribute_name))

    @classmethod
    def register_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_datapipe_as_function(cls, function_name, cls_to_register, enable_df_api_tracing=False):
        if function_name in cls.functions:
            raise Exception("Unable to add DataPipe function name {} as it is already taken".format(function_name))

        def class_function(cls, enable_df_api_tracing, source_dp, *args, **kwargs):
            result_pipe = cls(source_dp, *args, **kwargs)
            # if isinstance(result_pipe, Dataset):
            #     if enable_df_api_tracing or isinstance(source_dp, DFIterDataPipe):
            #         if function_name not in UNTRACABLE_DATAFRAME_PIPES:
            #             result_pipe = result_pipe.trace_as_dataframe()

            return result_pipe

        function = functools.partial(class_function, cls_to_register, enable_df_api_tracing)
        cls.functions[function_name] = function


class IterableDataset(Dataset[T_co]):

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])


class ChainDataset(IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.
    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.
    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
    """
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)
        return total


class ConcatDataset(Dataset[T_co]):
    r"""Dataset as a concatenation of multiple datasets.
    This class is useful to assemble different existing datasets.
    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super(ConcatDataset, self).__init__()
        self.datasets = list(datasets)
        assert len(self.datasets) > 0, 'datasets should not be an empty iterable'  # type: ignore[arg-type]
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class IterDataPipe(IterableDataset[T_co]):
    functions: Dict[str, Callable] = {}
    reduce_ex_hook : Optional[Callable] = None
    getstate_hook: Optional[Callable] = None

    def __getattr__(self, attribute_name):
        if attribute_name in IterDataPipe.functions:
            function = functools.partial(IterDataPipe.functions[attribute_name], self)
            return function
        else:
            raise AttributeError("'{0}' object has no attribute '{1}".format(self.__class__.__name__, attribute_name))

    def __getstate__(self):
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(self)
        return self.__dict__

    def __reduce_ex__(self, *args, **kwargs):
        if IterDataPipe.reduce_ex_hook is not None:
            try:
                return IterDataPipe.reduce_ex_hook(self)
            except NotImplementedError:
                pass
        return super().__reduce_ex__(*args, **kwargs)

    @classmethod
    def set_getstate_hook(cls, hook_fn):
        if IterDataPipe.getstate_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing getstate_hook")
        IterDataPipe.getstate_hook = hook_fn

    @classmethod
    def set_reduce_ex_hook(cls, hook_fn):
        if IterDataPipe.reduce_ex_hook is not None and hook_fn is not None:
            raise Exception("Attempt to override existing reduce_ex_hook")
        IterDataPipe.reduce_ex_hook = hook_fn


class SmilesDataset(object):
    """
    Astract class for datasets
    """
    def __init__(self, path, smiles_col=None, x_cols=None, y_cols=None) -> None:
        self._Y = None
        self._X = None
        self.path = path
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.smiles_col = smiles_col

    @property
    def X(self) -> Iterable[Any]:
        return self._X

    @property
    def Y(self) -> Iterable[Any]:
        return self._Y

    @property
    def ids(self) -> Iterable[Any]:
        return self.ids

    @property
    def x_cols(self) -> Iterable[Any]:
        return self._x_cols

    @property
    def y_cols(self) -> Iterable[Any]:
        return self._y_cols

    # @x_cols.setter
    # def _x_cols(self, value):
    #     self._x_cols = value
    #
    # @y_cols.setter
    # def y_cols(self, value):
    #     self._y_cols = value

    def create(self):
        raise NotImplementedError("Need implementation")

    def __repr__(self) -> str:
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != 'self']
        args_info = ''
        for arg_name in args_names:
          value = self.__dict__[arg_name]
          # for str
          if isinstance(value, str):
            value = "'" + value + "'"
          # for list
        return self.__class__.__name__

    @x_cols.setter
    def x_cols(self, value):
        self._x_cols = value

    @y_cols.setter
    def y_cols(self, value):
        self._y_cols = value


class MolecularTabularDataset(SmilesDataset):
    """
    Reads CSV with smiles, experimental features and endpoints
    """
    def __init__(self, path, x_cols=Iterable[Any], y_cols=Iterable[Any], smiles_col=None, smiles=Iterable[str]
                 , y: Iterable[Any] = Iterable[Any], X: Iterable[Any] = None,
                 featurizer: MolecularFeaturizer = None) -> None:
        # super(SmilesTabularDataset, path, x_cols, y_cols).__init__(path, x_cols, y_cols)
        super(MolecularTabularDataset, self).__init__(path=path, x_cols=x_cols, y_cols=y_cols)
        self._y = y
        self._x = X
        self._df = None
        self._x_cols_all = None
        self.smiles_col = smiles_col
        self.smiles = smiles
        self._smiles_strings = None
        self.ys = y
        self.featurizer: MolecularFeaturizer = featurizer
        self.indices: [] = None
        # self.create()

    @property
    def featurizer_name(self) -> Iterable[Any]:
        return self.featurizer.__name__

    @property
    def x_colls_all(self) -> Iterable[str]:
        return self._x_cols_all

    @x_colls_all.setter
    def x_colls_all(self, value):
        self._x_cols_all = value

    @property
    def X(self) -> Iterable[str]:
        return self._x

    @X.setter
    def X(self, value):
        self._x = value

    @property
    def y(self) -> Iterable[str]:
        return self._y

    @y.setter
    def y(self, value):
        self._y = value

    @property
    def smiles_strings(self) -> Iterable[str]:
        return self._smiles_strings

    @smiles_strings.setter
    def smiles_strings(self, value):
        self._smiles_strings = value

    @property
    def df(self) -> Any:
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    def create(self):
        name, extension = os.path.splitext(self.path)
        if extension == '.csv':
            data = pd.read_csv(self.path)
            smiles = data[self.smiles_col].to_list()
            self._smiles_strings = smiles
            descriptors = self.featurizer.featurize_dataframe(smiles)
            if self.x_cols:
                xs = data[self.x_cols]
                y = data[self.y_cols]
                self.df = pd.concat([data[self.smiles_col], descriptors, xs, y], axis=1)
                self._x_cols_all = list(descriptors) + list(xs)
                self.y_cols = list(y)
                self.y = list(y)
            else:
                y = data[self.y_cols]
                self.df = pd.concat([data[self.smiles_col], descriptors, y], axis=1)
                self._x_cols_all = list(descriptors)
                self.y_cols = list(y)
                self.y = list(y)
        return self

    def __get_X__(self):
        return self.df[self.X].to_numpy()

    def __get_Y__(self):
        return self.df[self.y].to_numpy()

    @featurizer_name.setter
    def featurizer_name(self, value):
        self._featurizer_name = value

    def __repr__(self) -> str:
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != 'self']
        args_info = ''
        for arg_name in args_names:
          value = self.__dict__[arg_name]
          # for str
          if isinstance(value, str):
            value = "'" + value + "'"
          # for list
        return self.__class__.__name__


class TestDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        sample = np.array([1])
        return sample