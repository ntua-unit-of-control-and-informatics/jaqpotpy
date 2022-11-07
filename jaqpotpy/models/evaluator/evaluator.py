from abc import ABCMeta
from typing import Dict, Callable, Any, Iterable
from jaqpotpy.datasets.molecular_datasets import SmilesDataset
import numpy as np


class Evaluator:
    functions: Dict[str, Callable] = {}
    dataset: Any

    def __init__(self):
        pass

    @classmethod
    def register_scoring_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_dataset(cls, dataset: SmilesDataset):
        dataset = dataset

    def __getitem__(self):
        return self


class GenerativeEvaluator:
    functions: Dict[str, Callable] = {}
    dataset: Iterable[str]

    def __init__(self):
        pass

    @classmethod
    def register_scoring_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_dataset(cls, dataset: Iterable[str]):
        dataset = dataset

    def __getitem__(self):
        return self

    def get_reward(self, mols):
        rr = np.empty(1)
        rr.fill(1.)
        for key in self.functions.keys():
            f = self.functions.get(key)
            rr *= f(mols)
        return rr.reshape(-1, 1)


class GenerativeReward:
    functions: Dict[str, Callable] = {}
    dataset: Any

    def __init__(self):
        pass

    @classmethod
    def register_scoring_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_dataset(cls, dataset: SmilesDataset):
        dataset = dataset

    def __getitem__(self):
        return self
