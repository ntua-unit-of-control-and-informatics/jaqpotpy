from typing import Dict, Callable, Any, Iterable
from jaqpotpy.datasets.molecular_datasets import SmilesDataset


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
    eval_functions: Dict[str, Callable] = {}
    dataset: Iterable[str]

    def __init__(self):
        pass

    @classmethod
    def register_scoring_function(cls, function_name, function):
        cls.functions[function_name] = function

    @classmethod
    def register_evaluation_function(cls, function_name, function):
        cls.eval_functions[function_name] = function

    @classmethod
    def register_dataset(cls, dataset: Iterable[str]):
        cls.dataset = dataset
        # dataset = dataset

    def __getitem__(self):
        return self

    def get_reward(self, mols):
        rr = 1.
        for key in self.functions.keys():
            try:
                f = self.functions.get(key)
                rr *= f(mols)
            except TypeError as e:
                f = self.functions.get(key)
                rr *= f(mols, self.dataset)
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
