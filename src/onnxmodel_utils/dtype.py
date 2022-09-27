from enum import Enum

from onnx import TensorProto


class DType(Enum):
    UNDEFINED = TensorProto.UNDEFINED
    INT8 = TensorProto.INT8
    INT16 = TensorProto.INT16
    INT32 = TensorProto.INT32
    INT64 = TensorProto.INT64
    BOOL = TensorProto.BOOL
    BFLOAT16 = TensorProto.BFLOAT16
    FLOAT16 = TensorProto.FLOAT16
    FLOAT = TensorProto.FLOAT
    DOUBLE = TensorProto.DOUBLE
    UINT8 = TensorProto.UINT8
    UINT16 = TensorProto.UINT16
    UINT32 = TensorProto.UINT32
    UINT64 = TensorProto.UINT64
    STRING = TensorProto.STRING
    COMPLEX64 = TensorProto.COMPLEX64
    COMPLEX128 = TensorProto.COMPLEX128
