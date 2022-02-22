from typing import Dict, Callable


class Preprocesses:

    classes: Dict[str, Callable] = {}
    fitted_classes:  Dict[str, Callable] = {}

    classes_y: Dict[str, Callable] = {}
    fitted_classes_y:  Dict[str, Callable] = {}

    def __init__(self):
        pass

    @classmethod
    def register_preprocess_class(cls, class_name, class_):
        cls.classes[class_name] = class_

    @classmethod
    def register_fitted_class(cls, class_name, class_):
        cls.fitted_classes[class_name] = class_

    @classmethod
    def register_preprocess_class_y(cls, class_name, class_):
        cls.classes_y[class_name] = class_

    @classmethod
    def register_fitted_class_y(cls, class_name, class_):
        cls.fitted_classes_y[class_name] = class_

    def __getitem__(self):
        return self
