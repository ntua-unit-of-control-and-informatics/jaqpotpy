from typing import Dict, Callable


class Preprocess:
    def __init__(self):
        self.classes: Dict[str, Callable] = {}
        self.fitted_classes: Dict[str, Callable] = {}

        self.classes_y: Dict[str, Callable] = {}
        self.fitted_classes_y: Dict[str, Callable] = {}

    def register_preprocess_class(self, class_name, class_):
        self.classes[class_name] = class_

    def register_fitted_class(self, class_name, class_):
        self.fitted_classes[class_name] = class_

    def register_preprocess_class_y(self, class_name, class_):
        self.classes_y[class_name] = class_

    def register_fitted_class_y(self, class_name, class_):
        self.fitted_classes_y[class_name] = class_

    def __getitem__(self):
        return self
