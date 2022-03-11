from abc import ABCMeta
from typing import Dict, Callable, Any
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
