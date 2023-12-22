import numpy as np
from onnx import ValueInfoProto, TensorProto

from skl2onnx.algebra.type_helper import guess_initial_types
from skl2onnx.common.data_types import (
    guess_numpy_type,
    DoubleTensorType,DataType,
    BooleanTensorType,
    Complex128TensorType, Complex64TensorType,
    FloatTensorType,
    Int8TensorType, Int16TensorType, Int32TensorType, Int64TensorType,
    UInt8TensorType, UInt16TensorType, UInt32TensorType, UInt64TensorType,
    StringTensorType
)

def select_ONNX_type(t):
    if 'float' in t:
        return FloatTensorType([None, 1])
    elif t == 'bool':
        return BooleanTensorType([None, 1])
    elif t == 'complex128':
        return Complex128TensorType([None, 1])
    elif t == 'complex64':
        return Complex64TensorType([None, 1])
    elif t == 'int8':
        return Int8TensorType([None, 1])
    elif t == 'int16':
        return Int16TensorType([None, 1])
    elif t == 'int32':
        return Int32TensorType([None, 1])
    elif t == 'int64':
        return Int64TensorType([None, 1])
    elif t == 'uint8':
        return UInt8TensorType([None, 1])
    elif t == 'uint16':
        return UInt16TensorType([None, 1])
    elif t == 'uint32':
        return UInt32TensorType([None, 1])
    elif t == 'uint64':
        return UInt64TensorType([None, 1])
    elif t == 'str':
        return StringTensorType([None, 1])
    else:
        raise ValueError('"{}" is not a valid datatype'.format(t))
    # return None


def guess_ONNX_types(df, drop=None):
    if drop is None:
        df_copy = df.copy()
    else:
        df_copy = df.drop(drop, axis=1).copy()

    type_list = [(k, select_ONNX_type(str(v)))for k, v in zip(df_copy.columns, df_copy.dtypes)]
    return type_list
    

    










#
# def guess_dtype(proto_type):
#     """
#     Converts a proto type into a :epkg:`numpy` type.
#     :param proto_type: example ``onnx.TensorProto.FLOAT``
#     :return: :epkg:`numpy` dtype
#     """
#     if proto_type == TensorProto.FLOAT:  # pylint: disable=E1101
#         return np.float32
#     if proto_type == TensorProto.BOOL:  # pylint: disable=E1101
#         return np.bool_
#     if proto_type == TensorProto.DOUBLE:  # pylint: disable=E1101
#         return np.float64
#     if proto_type == TensorProto.STRING:  # pylint: disable=E1101
#         return np.str_
#     if proto_type == TensorProto.INT64:  # pylint: disable=E1101
#         return np.int64
#     if proto_type == TensorProto.INT32:  # pylint: disable=E1101
#         return np.int32
#     if proto_type == TensorProto.INT8:  # pylint: disable=E1101
#         return np.int8
#     if proto_type == TensorProto.INT16:  # pylint: disable=E1101
#         return np.int16
#     if proto_type == TensorProto.UINT64:  # pylint: disable=E1101
#         return np.uint64
#     if proto_type == TensorProto.UINT32:  # pylint: disable=E1101
#         return np.uint32
#     if proto_type == TensorProto.UINT8:  # pylint: disable=E1101
#         return np.uint8
#     if proto_type == TensorProto.UINT16:  # pylint: disable=E1101
#         return np.uint16
#     if proto_type == TensorProto.FLOAT16:  # pylint: disable=E1101
#         return np.float16
#     raise ValueError(
#         f"Unable to convert proto_type {proto_type} to numpy type.")

#
# def guess_ONNX_types(X, itype, dtype):
#     initial_types = guess_initial_types(X, itype)
#     if dtype is None:
#         if hasattr(X, 'dtypes'):  # DataFrame
#             dtype = np.float32
#         elif hasattr(X, 'dtype'):
#             dtype = X.dtype
#         elif hasattr(X, 'type'):
#             dtype = guess_numpy_type(X.type)
#         elif isinstance(initial_types[0], ValueInfoProto):
#             dtype = guess_dtype(initial_types[0].type.tensor_type.elem_type)
#         elif initial_types is not None:
#             dtype = guess_numpy_type(initial_types[0][1])
#         else:
#             raise RuntimeError(  # pragma: no cover
#                 f"dtype cannot be guessed: {type(X)}")
#         if dtype != np.float64:
#             dtype = np.float32
#     if dtype is None:
#         raise RuntimeError("dtype cannot be None")  # pragma: no cover
#     if isinstance(dtype, FloatTensorType):
#         dtype = np.float32  # pragma: no cover
#     elif isinstance(dtype, DoubleTensorType):
#         dtype = np.float64  # pragma: no cover
#     new_dtype = dtype
#     if isinstance(dtype, np.ndarray):
#         new_dtype = dtype.dtype  # pragma: no cover
#     elif isinstance(dtype, DataType):
#         new_dtype = np.float32  # pragma: no cover
#     if new_dtype not in (np.float32, np.float64, np.int64,
#                          np.int32, np.float16):
#         raise NotImplementedError(  # pragma: no cover
#             f"dtype should be real not {new_dtype} ({dtype})")
#     return initial_types, dtype, new_dtype
