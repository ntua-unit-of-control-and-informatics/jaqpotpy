from enum import Enum


class ModelType(str, Enum):
    QSAR_TOOLBOX = "QSAR_TOOLBOX"
    R_BNLEARN_DISCRETE = "R_BNLEARN_DISCRETE"
    R_CARET = "R_CARET"
    R_GBM = "R_GBM"
    R_NAIVE_BAYES = "R_NAIVE_BAYES"
    R_PBPK = "R_PBPK"
    R_RF = "R_RF"
    R_RPART = "R_RPART"
    R_SVM = "R_SVM"
    R_TREE_CLASS = "R_TREE_CLASS"
    R_TREE_REGR = "R_TREE_REGR"
    SKLEARN = "SKLEARN"
    TORCHSCRIPT = "TORCHSCRIPT"
    TORCH_ONNX = "TORCH_ONNX"

    def __str__(self) -> str:
        return str(self.value)
