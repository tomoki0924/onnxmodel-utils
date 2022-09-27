import collections
import contextlib
import copy
import json
import os
import sys
import tempfile
import time
import weakref
from typing import Any, Dict, List, Optional, TypeVar, Union
from uuid import uuid4

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from loguru import logger
from onnx import (
    AttributeProto,
    GraphProto,
    ModelProto,
    NodeProto,
    OperatorSetIdProto,
    OptionalProto,
    TensorProto,
    TypeProto,
    ValueInfoProto,
    helper,
    numpy_helper,
    version_converter,
)
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxsim import simplify
from tqdm import tqdm

AttributeType = AttributeProto.AttributeType


from .container import AutoKeyDict, WeakList
from .dtype import DType
from .transform.base import TransformCompose
from .transform.decompose_sln import DecomposeSLN
from .transform.fuse_mulmatmul import FuseMulMatMul
from .transform.merge_matmul import MergeMatMul


def value_to_dtype(value: int) -> DType:
    maps = {}
    for dtype in DType:
        maps[dtype.value] = dtype
    return maps[value]


def get_numpy_dtype(dtype: DType) -> np.dtype:
    if dtype == DType.INT8:
        return np.int8
    elif dtype == DType.INT16:
        return np.int16
    elif dtype == DType.INT32:
        return np.int32
    elif dtype == DType.INT64:
        return np.int64
    elif dtype == DType.BOOL:
        return np.bool
    elif dtype == DType.FLOAT16:
        return np.float16
    elif dtype == DType.FLOAT:
        return np.float32
    elif dtype == DType.DOUBLE:
        return np.float64
    elif dtype == DType.UINT8:
        return np.uint8
    elif dtype == DType.UINT16:
        return np.uint16
    elif dtype == DType.UINT32:
        return np.uint32
    elif dtype == DType.UINT64:
        return np.uint64
    elif dtype == DType.STRING:
        return np.object
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))


def to_numpy(value: Any, dtype: DType) -> np.ndarray:
    if dtype == DType.BFLOAT16:
        np_fp32 = value.astype(np.float32)
        little_endisan = sys.byteorder == "little"
        np_uint16_view = np_fp32.view(dtype=np.uint16)
        np_bfp16 = np_uint16_view[1::2] if little_endisan else np_uint16_view[0::2]
        return np_bfp16.copy()
    else:
        return value.astype(get_numpy_dtype(dtype))


class Schema:
    def __init__(self, schema: onnx.defs.OpSchema):
        self.name = schema.name
        self.inputs = {i.name: i.typeStr for i in schema.inputs}
        self.outputs = {i.name: i.typeStr for i in schema.outputs}
        self.type_constraints = {
            i.type_param_str: i.allowed_type_strs for i in schema.type_constraints
        }

    @staticmethod
    def str2dtype(dtype: str) -> Optional[DType]:
        if dtype.startswith("tensor("):
            dtype = dtype[7:-1]
            return DType[dtype.upper()]
        return None
        # elif dtype.startswith("optional(tensor("):
        #     dtype = dtype[16:-2]
        #     return DType[dtype.upper()]

    def infer_output_types(self, node: "Node", name_to_tensor: Dict[str, "Tensor"]):
        if node.op_type == "Constant":
            return
        inputs = [name_to_tensor[i] if i else "" for i in node.inputs]
        outputs = [name_to_tensor[o] for o in node.outputs]
        if node.op_type == "Cast":
            a = node.get_attr_by_name("to")
            if a:
                outputs[0].dtype = value_to_dtype(a.value)
            for o in outputs:
                name_to_tensor[o.name].update(o)
            return

        type_map = {}
        for i, t in zip(inputs, self.inputs):
            if not i:
                continue
            s = self.inputs[t]
            if s in type_map:
                if type_map[s] != i.dtype and i.dtype != DType.UNDEFINED:
                    raise ValueError("Inconsistent type for input %s" % i)
            elif i.dtype != DType.UNDEFINED:
                type_map[s] = i.dtype
        for i, t in zip(outputs, self.outputs):
            s = self.outputs[t]
            if s in type_map:
                i.dtype = type_map[s]

        if not self.check_types(node, name_to_tensor):
            raise ValueError("Type mismatch after inference for node %s" % node)

        for o in outputs:
            if o.dtype == DType.UNDEFINED:
                continue
            name_to_tensor[o.name].update(o)

    def check_types(self, node: "Node", name_to_tensor: Dict[str, "Tensor"]):
        if node.op_type == "Constant":
            return
        inputs = [name_to_tensor[i] if i else "" for i in node.inputs]
        outputs = [name_to_tensor[o] for o in node.outputs]
        type_map = {}
        for i, t in zip(inputs, self.inputs):
            if not i:
                continue
            s = self.inputs[t]
            if s in type_map:
                if type_map[s] != i.dtype and i.dtype != DType.UNDEFINED:
                    logger.warning(
                        f"Type mismatch for input {i}. Expected {type_map[s]}, got {i.dtype}"
                    )
                    raise ValueError("Inconsistent type for input %s" % i)
            else:
                type_map[s] = i.dtype
        for i, t in zip(outputs, self.outputs):
            s = self.outputs[t]
            if s in type_map:
                if type_map[s] != i.dtype and i.dtype != DType.UNDEFINED:
                    logger.warning(
                        f"Type mismatch for output {i}. Expected {type_map[s]}, got {i.dtype}"
                    )
                    raise ValueError("Inconsistent type for output %s" % i)
            else:
                type_map[s] = i.dtype

        for type_param_str, allowed_type_strs in self.type_constraints.items():
            if type_param_str not in type_map:
                continue
            allowed_type_strs = [self.str2dtype(t) for t in allowed_type_strs]
            if type_map[type_param_str] not in allowed_type_strs:
                logger.warning(
                    f"Type mismatch for {type_param_str}. Expected {allowed_type_strs}, got {type_map[type_param_str]}"
                )
                return False
        return True


Shape = List[Optional[Union[int, str]]]
T = TypeVar("T")


class Base:
    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_onnx(cls):
        pass

    def clone(self):
        return copy.deepcopy(self)

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo):
        raise NotImplementedError


class Tensor(Base):
    def __init__(
        self,
        name: str,
        dtype: Optional[DType] = None,
        shape: Optional[Shape] = None,
        data: Optional[np.ndarray] = None,
        is_optional: bool = False,
    ):
        if name == "" and data is not None:
            name = str(uuid4())
        self.name = name
        self._dtype = dtype
        self._shape = shape
        self._data = data
        self.is_optional = is_optional
        if self._data is not None:
            self._data = np.array(self._data)
            if self.dtype is not DType.UNDEFINED:
                self._data = to_numpy(self._data, self.dtype)

    def __copy__(self):
        obj = Tensor(self.name, self.dtype, self.shape, self.data, self.is_optional)
        return obj

    def __deepcopy__(self, memo):
        data = self._data if self._data is None else self._data.copy()
        shape = self._shape if self._shape is None else copy.deepcopy(self._shape)
        dtype = self._dtype
        name = self.name
        is_optional = self.is_optional
        obj = Tensor(name, dtype, shape, data, is_optional)
        assert obj == self
        return obj

    def __eq__(self, __o: object) -> bool:
        return all(
            [
                self.name == __o.name,
                self.dtype == __o.dtype,
                self.shape == __o.shape,
                np.array_equal(self.data, __o.data),
                self.is_optional == __o.is_optional,
            ]
        )

    def __hash__(self) -> int:
        return hash(
            str(self.name)
            + str(self.dtype)
            + str(self.shape)
            + str(self.data)
            + str(self.is_optional)
        )

    @property
    def data(self) -> Optional[np.ndarray]:
        if self._data is None:
            return None
        return self._data

    @data.setter
    def data(self, data: np.ndarray):
        self._data = to_numpy(data, self.dtype)

    @staticmethod
    def to_shape(tensor_type: TypeProto) -> Optional[Shape]:
        shape_list = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape_list.append(d.dim_value)
            elif d.HasField("dim_param"):
                shape_list.append(d.dim_param)
            else:
                return None
        return shape_list

    @property
    def shape(self) -> Optional[Shape]:
        if self._shape is None:
            return None
        for s in self._shape:
            if s is None:
                return None
        shape = copy.deepcopy(self._shape)
        return shape

    @shape.setter
    def shape(self, shape: Shape):
        self._shape = copy.deepcopy(shape)

    def clear_shape(self):
        self._shape = ["?"]

    @property
    def dtype(self) -> DType:
        if self._dtype is None:
            return DType.UNDEFINED
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: DType):
        self._dtype = dtype
        if self.data is not None:
            self._data = to_numpy(self._data, self.dtype)

    def mod_shape(self, axis, value):
        shape = self.shape
        if shape is None:
            raise ValueError("Cannot modify shape of tensor with unknown shape")
        shape[axis] = value
        self._shape = shape

    def rename_shape(self, old_name, new_name):
        shape = self.shape
        if shape is None:
            return
        for i, s in enumerate(shape):
            if s == old_name:
                shape[i] = new_name
        self._shape = shape

    def __repr__(self):
        return f"Tensor(name={self.name}, dtype={self.dtype}, shape={self.shape}, is_optional={self.is_optional})"

    def to_onnx_tensor(self) -> TensorProto:
        if self.data is None:
            raise ValueError("Tensor data is None")
        value = helper.make_tensor(
            name=self.name,
            data_type=self.dtype.value,
            dims=self.shape,
            vals=self.data.tobytes(),
            raw=True,
        )
        if self.is_optional:
            value = helper.make_optional(
                name=self.name, elem_type=onnx.OptionalProto.TENSOR, value=value
            )
        return value

    def to_onnx_type_proto(self) -> TypeProto:
        type_proto = helper.make_tensor_type_proto(
            elem_type=self.dtype.value, shape=self.shape
        )
        if self.is_optional:
            type_proto = helper.make_optional_type_proto(type_proto)
        return type_proto

    def to_onnx_tensor_value_info(self) -> ValueInfoProto:
        if self.is_optional:
            type_proto = self.to_onnx_type_proto()
            value_info = helper.make_value_info(
                self.name,
                type_proto,
            )
        else:
            value_info = helper.make_tensor_value_info(
                name=self.name,
                elem_type=self.dtype.value,
                shape=self.shape,
            )
        return value_info

    @classmethod
    def from_onnx(cls, tensor: Union[TensorProto, ValueInfoProto]) -> "Tensor":
        dtype = None
        shape = None
        is_optional = False
        if isinstance(tensor, TensorProto) or isinstance(tensor, OptionalProto):
            is_optional = isinstance(tensor, OptionalProto)
            if is_optional:
                assert tensor.HasField("tensor_value")
                tensor = tensor.tensor_value
            with contextlib.suppress(ValueError):
                dtype = DType(tensor.data_type)
            with contextlib.suppress(ValueError):
                shape = tensor.dims
            data = numpy_helper.to_array(tensor)
            if shape is None:
                shape = data.shape
            obj = cls(
                name=tensor.name,
                dtype=dtype,
                shape=shape,
                data=data,
                is_optional=is_optional,
            )
        elif isinstance(tensor, ValueInfoProto):
            is_optional = tensor.type.HasField("optional_type")
            if is_optional:
                tensor_type = tensor.type.optional_type.elem_type.tensor_type
            else:
                tensor_type = tensor.type.tensor_type
            with contextlib.suppress(ValueError):
                dtype = DType(tensor_type.elem_type)
            with contextlib.suppress(ValueError):
                shape = cls.to_shape(tensor_type)
            obj = cls(
                name=tensor.name,
                dtype=dtype,
                shape=shape,
                is_optional=is_optional,
            )
        else:
            raise ValueError("Unsupported type: {}".format(type(tensor)))
        return obj

    def add_prefix(self, prefix: str, include_shape: bool = True):
        if self.name:
            self.name = prefix + self.name
            shape = self.shape
            if shape is not None and include_shape:
                self._shape = [prefix + s if isinstance(s, str) else s for s in shape]

    def update(self, other: "Tensor"):
        if self.name != other.name:
            raise ValueError(
                "Tensor name mismatch: {} != {}".format(self.name, other.name)
            )
        if self.dtype == DType.UNDEFINED:
            self._dtype = other.dtype
        if self.shape is None and other.shape:
            self._shape = other.shape
        if self.data is None:
            self._data = other.data

    def to(self, dtype: DType):
        self._dtype = dtype
        if self.data is not None:
            self._data = to_numpy(self._data, self.dtype)

    def has_same_data(self, other: "Tensor"):
        if self.data is None or other.data is None:
            return False
        if self.dtype != other.dtype:
            return False
        if self.shape != other.shape:
            return False
        return np.array_equal(self.data, other.data)


class Attribute(Base):
    def __init__(self, name: str, value: Any, attr_type: AttributeType = None):
        self.name = name
        self.value = self.unpack(value, attr_type)
        self.attr_type = attr_type

    def __copy__(self):
        obj = Attribute(self.name, self.value, self.attr_type)
        return obj

    def __deepcopy__(self, memo):
        obj = Attribute.from_onnx(self.to_onnx())
        obj.attr_type = self.attr_type
        return obj

    @classmethod
    def from_onnx(cls, attr: AttributeProto) -> "Attribute":
        name = attr.name
        value = helper.get_attribute_value(attr)
        attr_type = attr.type
        return cls(name, value, attr_type)

    def to_onnx(self) -> AttributeProto:
        value = self.pack(self.value, self.attr_type)
        return helper.make_attribute(self.name, value)

    def __repr__(self):
        return f"Attribute(name={self.name}, value={self.value}, attr_type={self.attr_type})"

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Attribute):
            return False
        if self.name != __o.name:
            return False
        if type(self.value) != type(__o.value):
            return False
        return self.value == __o.value

    def __hash__(self) -> int:
        strings = self.to_onnx().SerializeToString()
        return hash(strings)

    def unpack(self, value, attr_type) -> Any:
        if attr_type == AttributeType.GRAPH:
            return Graph.from_onnx(value, is_subgraph=True)
        elif attr_type == AttributeType.GRAPHS:
            return [Graph.from_onnx(g, is_subgraph=True) for g in value]
        elif attr_type == AttributeType.TENSOR:
            return Tensor.from_onnx(value)
        elif attr_type == AttributeType.TENSORS:
            return [Tensor.from_onnx(t) for t in value]
        else:
            return value

    def pack(self, value, attr_type) -> Any:
        if attr_type == AttributeType.GRAPH:
            return value.to_onnx_graph(is_subgraph=True)
        elif attr_type == AttributeType.GRAPHS:
            return [g.to_onnx_graph(is_subgraph=True) for g in value]
        elif attr_type == AttributeType.TENSOR:
            return value.to_onnx_tensor()
        elif attr_type == AttributeType.TENSORS:
            return [t.to_onnx_tensor() for t in value]
        else:
            return value


class Node(Base):
    def __init__(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attrs: List[Attribute] = None,
        domain: str = "",
    ):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attrs = attrs or []
        self.domain = domain

        if self.name == "":
            self.name = op_type + "_" + str(uuid4())

    def __copy__(self):
        obj = Node(
            self.name, self.op_type, self.inputs, self.outputs, self.attrs, self.domain
        )
        return obj

    def __deepcopy__(self, memo):
        return Node.from_onnx(self.to_onnx_node())

    def __eq__(self, __o: object) -> bool:
        return all(
            [
                self.name == __o.name,
                self.op_type == __o.op_type,
                all([i == o for i, o in zip(self.inputs, __o.inputs)]),
                all([i == o for i, o in zip(self.outputs, __o.outputs)]),
                all([i == o for i, o in zip(self.attrs, __o.attrs)]),
                self.domain == __o.domain,
            ]
        )

    def __hash__(self) -> int:
        strings = self.to_onnx_node().SerializeToString()
        return hash(strings)

    def __repr__(self):
        return f"Node(name={self.name}, op_type={self.op_type}, inputs={self.inputs}, outputs={self.outputs}, attrs={self.attrs})"

    def to_onnx_node(self) -> NodeProto:
        node = helper.make_node(
            name=self.name,
            op_type=self.op_type,
            inputs=[t for t in self.inputs],
            outputs=[t for t in self.outputs],
            domain=self.domain,
        )
        node.attribute.extend([a.to_onnx() for a in self.attrs])
        return node

    @classmethod
    def from_onnx(cls, node: NodeProto):
        attrs = []
        for attr in node.attribute:
            attrs.append(Attribute.from_onnx(attr))
        inputs = []
        for i in node.input:
            inputs.append(i)
        outputs = []
        for o in node.output:
            outputs.append(o)
        return cls(
            name=node.name,
            op_type=node.op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            domain=node.domain,
        )

    def add_prefix(self, prefix: str):
        self.name = prefix + self.name
        inputs = []
        for i in self.inputs:
            if i:
                i = prefix + i
            inputs.append(i)
        outputs = []
        for o in self.outputs:
            if o:
                o = prefix + o
            outputs.append(o)
        self.inputs = inputs
        self.outputs = outputs

    def n_inputs(self):
        inputs = set(self.inputs)
        for g in self.get_graph_attrs():
            inputs.update(g.input_names)
        return sum(1 for i in inputs if i)

    def n_outputs(self):
        return len(self.outputs)

    def get_attr_by_name(self, name: str):
        for attr in self.attrs:
            if attr.name == name:
                return attr
        return None

    def get_graph_attrs(self):
        out = []
        for attr in self.attrs:
            if attr.attr_type == AttributeType.GRAPH:
                out.append(attr.value)
            elif attr.attr_type == AttributeType.GRAPHS:
                out.extend(attr.value)
        return out

    def replace_input(self, old_name: str, new_name: str):
        for i, input in enumerate(self.inputs):
            if input == old_name:
                self.inputs[i] = new_name
        for g in self.get_graph_attrs():
            for n in g.nodes:
                n.replace_input(old_name, new_name)

    def replace_output(self, old_name: str, new_name: str):
        for i, output in enumerate(self.outputs):
            if output == old_name:
                self.outputs[i] = new_name

    def replace_io(self, old_name: str, new_name: str):
        self.replace_input(old_name, new_name)
        self.replace_output(old_name, new_name)


class Graph(Base):
    def __init__(
        self,
        name: str,
        nodes: List[Node],
        input_names: List[str],
        output_names: List[str],
        tensors: List[Tensor],
        is_subgraph: bool = False,
    ):
        self.name = name
        self.input_names = input_names
        self.output_names = output_names
        self.is_subgraph = is_subgraph
        self.name_to_tensor = AutoKeyDict(lambda x: x.name)
        for t in tensors:
            self.name_to_tensor.add(t)
        self.update_nodes(nodes)

    def __copy__(self):
        obj = Graph(
            self.name, self.nodes, self.input_names, self.output_names, self.tensors
        )
        return obj

    def __deepcopy__(self, memo):
        return Graph.from_onnx(self.to_onnx_graph(self.is_subgraph), self.is_subgraph)

    def add_tensor(self, tensor: Tensor, recursive: bool = False):
        if tensor.name in self.name_to_tensor:
            count = 0
            for node in self.nodes:
                if tensor.name in node.inputs + node.outputs:
                    logger.warning(
                        f"Tensor {tensor.name} is used by node {node.name}, cannot be replaced"
                    )
                    count += 1
            if count == 0:
                self.name_to_tensor.remove(tensor.name)
            else:
                raise ValueError(f"Tensor {tensor.name} already exists")
        self.name_to_tensor.add(tensor)
        if recursive:
            for g in self.subgraphs:
                g.add_tensor(tensor, recursive)

    def add_node(self, node: Node, left: bool = False):
        if node.name in self.name_to_node:
            raise ValueError(f"Node {node.name} already exists")
        if left:
            self.nodes.appendleft(node)
        else:
            self.nodes.append(node)
        self.name_to_node.add(node)

    def update_nodes(self, nodes: List[Node]):
        self.name_to_node = AutoKeyDict(lambda x: x.name)
        for node in nodes:
            self.name_to_node.add(node)
        self.nodes = WeakList.from_list(nodes)

    def update(self):
        self.name_to_tensor.update()
        self.name_to_node.update()
        for g in self.subgraphs:
            g.update()

    def __eq__(self, __o: object) -> bool:
        return all(
            [
                self.name == __o.name,
                all([n == o for n, o in zip(self.nodes, __o.nodes)]),
                all([i == o for i, o in zip(self.input_names, __o.input_names)]),
                all([i == o for i, o in zip(self.output_names, __o.output_names)]),
                self.name_to_tensor == __o.name_to_tensor,
            ]
        )

    def __hash__(self) -> int:
        strings = self.to_onnx_graph().SerializeToString()
        return hash(strings)

    def __repr__(self):
        return f"Graph(name={self.name})"

    def to_onnx_graph(self, is_subgraph: bool = False) -> GraphProto:
        self.is_subgraph = is_subgraph
        inputs = []
        if not self.is_subgraph:
            inputs = [t.to_onnx_tensor_value_info() for t in self.inputs]
        return helper.make_graph(
            nodes=[n.to_onnx_node() for n in self.nodes],
            name=self.name,
            inputs=inputs,
            outputs=[t.to_onnx_tensor_value_info() for t in self.outputs],
            initializer=[t.to_onnx_tensor() for t in self.initializers],
            value_info=[t.to_onnx_tensor_value_info() for t in self.value_info],
        )

    @classmethod
    def from_onnx(cls, graph: GraphProto, is_subgraph: bool = False):
        name_to_tensor = {}
        for i in graph.input:
            name_to_tensor[i.name] = Tensor.from_onnx(i)
        for o in graph.output:
            name_to_tensor[o.name] = Tensor.from_onnx(o)
        for i in graph.initializer:
            name_to_tensor[i.name] = Tensor.from_onnx(i)
        for v in graph.value_info:
            if v.name not in name_to_tensor:
                name_to_tensor[v.name] = Tensor.from_onnx(v)
            else:
                name_to_tensor[v.name].update(Tensor.from_onnx(v))

        nodes = list()
        for node in graph.node:
            nodes.append(Node.from_onnx(node))
            for i in node.input:
                if i not in name_to_tensor:
                    name_to_tensor[i] = Tensor(name=i)
            for o in node.output:
                if o not in name_to_tensor:
                    name_to_tensor[o] = Tensor(name=o)

        input_names = [i.name for i in graph.input]
        output_names = [o.name for o in graph.output]
        return cls(
            name=graph.name,
            nodes=nodes,
            input_names=input_names,
            output_names=output_names,
            tensors=list(name_to_tensor.values()),
            is_subgraph=is_subgraph,
        )

    def add_prefix(self, prefix: str, include_shape: bool = True):
        self.name = prefix + self.name
        for v in self.name_to_tensor.values():
            v.add_prefix(prefix, include_shape)
        self.name_to_tensor.update()
        new_nodes = []
        for n in self.name_to_node.values():
            n.add_prefix(prefix)
            new_nodes.append(n)
        self.name_to_node.update()
        self.update_nodes(new_nodes)
        for i in range(len(self.input_names)):
            self.input_names[i] = prefix + self.input_names[i]
        for i in range(len(self.output_names)):
            self.output_names[i] = prefix + self.output_names[i]

    def topological_sort(self, parent_initializers=None):
        if len(set([n.name for n in self.nodes])) != len(self.nodes):
            raise ValueError("Graph contains nodes with duplicate names")

        name2node = {n.name: n for n in self.nodes}
        deps_count = collections.defaultdict(int)
        deps_to_nodes = collections.defaultdict(list)
        sorted_nodes = []
        for node in self.nodes:
            deps_count[node.name] = node.n_inputs()
            if deps_count[node.name] == 0:
                sorted_nodes.append(name2node[node.name])
                continue

            for i in node.inputs:
                if i and node.name not in deps_to_nodes.get(i, []):
                    deps_to_nodes[i].append(node.name)
                for g in node.get_graph_attrs():
                    for j in g.input_names:
                        if j and node.name not in deps_to_nodes.get(j, []):
                            deps_to_nodes[j].append(node.name)

        initializer_names = [i.name for i in self.initializers if i.name]
        all_initializer_names = list(
            set(initializer_names + (parent_initializers or []))
        )
        self_input_names = [i.name for i in self.inputs if i.name]
        input_names = list(set(all_initializer_names + self_input_names))
        input_names.sort()
        for input_name in input_names:
            for node_name in deps_to_nodes[input_name]:
                deps_count[node_name] -= 1
                if deps_count[node_name] == 0:
                    sorted_nodes.append(name2node[node_name])
        start = 0
        end = len(sorted_nodes)

        while start < end:
            for output in sorted_nodes[start].outputs:
                if output not in deps_to_nodes:
                    continue
                for node_name in deps_to_nodes[output]:
                    deps_count[node_name] -= 1
                    if deps_count[node_name] != 0:
                        continue
                    sorted_nodes.append(name2node[node_name])
                    end += 1
            start += 1

        if end != len(self.nodes):
            logger.error("Graph has cycles")
            logger.error(
                f"#sorted_nodes = {len(sorted_nodes)} but #nodes = {len(self.nodes)}"
            )
            logger.error(
                f"diff: {set(map(lambda x: x.name + '::' + x.op_type, self.nodes)) - set(map(lambda x: x.name + '::' + x.op_type, sorted_nodes))}"
            )
            raise RuntimeError("Graph is not a DAG")
        self.update_nodes(sorted_nodes)
        for g in self.subgraphs:
            g.topological_sort(input_names.copy())

    def input_name_to_nodes(self):
        name2nodes = {}
        for n in self.nodes:
            for i in n.inputs:
                if i:
                    if i not in name2nodes:
                        name2nodes[i] = []
                    name2nodes[i].append(n)
                for g in n.get_graph_attrs():
                    for i in g.input_names:
                        if i not in name2nodes:
                            name2nodes[i] = []
                        elif n not in name2nodes[i]:
                            name2nodes[i].append(n)
        return name2nodes

    def output_name_to_node(self):
        name2node = weakref.WeakValueDictionary()
        for n in self.nodes:
            for o in n.outputs:
                if o:
                    name2node[o] = n
        return name2node

    def get_parents(self, node: Node, output_name_to_node: Dict[str, Node]):
        parents = WeakList()
        for i in node.inputs:
            if i and i in output_name_to_node:
                parents.append(output_name_to_node[i])
        return parents

    def get_children(self, node: Node, input_name_to_nodes: Dict[str, List[Node]]):
        children = WeakList()
        for o in node.outputs:
            if o and o in input_name_to_nodes:
                children.extend(input_name_to_nodes[o])
        return children

    @property
    def subgraphs(self):
        out = []
        for node in self.nodes:
            out.extend(node.get_graph_attrs())
        return out

    def prune(self):
        output_name_to_node = self.output_name_to_node()
        seen = set()
        queue = collections.deque()
        for o in self.outputs:
            if o.name:
                queue.append(output_name_to_node[o.name])
        while queue:
            node = queue.popleft()
            if node.name in seen:
                continue
            seen.add(node.name)
            for p in self.get_parents(node, output_name_to_node):
                queue.append(p)
        active_nodes = []
        for n in self.nodes:
            if n.name in seen:
                active_nodes.append(n)
        self.update_nodes(active_nodes)

        input_name_to_nodes = self.input_name_to_nodes()
        active_inputs = []
        for i in self.inputs:
            if i.name in input_name_to_nodes:
                active_inputs.append(i.name)
        self.input_names = active_inputs

    def node_inputs(self):
        inputs = set()
        for n in self.nodes:
            for i in n.inputs:
                if i:
                    inputs.add(i)
        for g in self.subgraphs:
            inputs.update(g.node_inputs())
        return inputs

    def node_outputs(self):
        outputs = set()
        for n in self.nodes:
            for o in n.outputs:
                outputs.add(o)
        return outputs

    @property
    def initializers(self):
        inps = self.node_inputs()
        oups = self.node_outputs()
        names = inps - oups
        return [
            self.name_to_tensor[name]
            for name in names
            if name in self.name_to_tensor
            and self.name_to_tensor[name].data is not None
        ]

    @property
    def inputs(self):
        return [self.name_to_tensor[name] for name in self.input_names]

    @property
    def outputs(self):
        return [self.name_to_tensor[name] for name in self.output_names]

    @property
    def value_info(self):
        value_info = [
            v
            for v in self.name_to_tensor.values()
            if v.name and v.shape is not None and v.dtype != DType.UNDEFINED
        ]
        return value_info

    def remove_tensor(self, name, recursive=False):
        self.name_to_tensor.remove(name)
        if recursive:
            for g in self.subgraphs:
                g.remove_tensor(name, recursive)

    def remove_duplicated_initializers(self):
        maps = {}
        initializers = self.initializers
        for i in initializers:
            if i.name in maps:
                continue
            for j in initializers:
                if i.name == j.name:
                    continue
                if j.name in maps:
                    continue
                if i.has_same_data(j):
                    maps[j.name] = i.name
                    self.remove_tensor(j.name)
        for n in self.nodes:
            for i in n.inputs:
                if i in maps:
                    n.replace_input(i, maps[i])
        self.update()


class Model(Base):
    schemas = {}

    def __init__(
        self,
        graph: Graph,
        ir_version: Optional[int] = None,
        opset_imports: Optional[List[OperatorSetIdProto]] = None,
    ):
        self.graph = graph
        self.ir_version = ir_version
        self.opset_imports = opset_imports
        self._verbose = True
        self._test_shape = {}

    def add_opset(self, domain: str, version: int):
        if self.opset_imports is not None:
            for opset in self.opset_imports:
                if opset.domain == domain:
                    opset.version = version
                    return
        if self.opset_imports is None:
            self.opset_imports = []
        self.opset_imports.append(OperatorSetIdProto(domain=domain, version=version))

    @property
    def default_opset(self):
        for opset in self.opset_imports:
            if opset.domain == "":
                return opset.version

    def update_default_opset(self, opset: int):
        for opset_import in self.opset_imports:
            if opset_import.domain == "":
                opset_import.version = opset
                return
        self.opset_imports.append(OperatorSetIdProto(domain="", version=opset))

    def version_convert(self, version: int):
        converted_model = onnx.version_converter.convert_version(
            self.to_onnx_model(),
            version,
        )
        self.update_from_onnx(converted_model)

    def __copy__(self):
        obj = Model(self.graph, self.ir_version, self.opset_imports)
        obj._verbose = self._verbose
        return obj

    def __deepcopy__(self, memo):
        return Model.from_onnx(self.to_onnx_model())

    def get_schema(self, op_type: str):
        if op_type not in self.schemas:
            for s in onnx.defs.get_all_schemas_with_history():
                if s.name == op_type:
                    self.schemas[op_type] = Schema(s)
        return self.schemas.get(op_type)

    def __eq__(self, __o: object) -> bool:
        return all(
            [
                self.graph == __o.graph,
                self.ir_version == __o.ir_version,
                all([o == i for o, i in zip(self.opset_imports, __o.opset_imports)]),
            ]
        )

    def __hash__(self):
        strings = self.to_onnx_model().SerializeToString()
        return hash(strings)

    def __repr__(self):
        return f"Model(graph={self.graph}, ir_version={self.ir_version})"

    def __str__(self):
        return self.__repr__()

    def to_onnx_model(self) -> ModelProto:
        return helper.make_model(
            graph=self.graph.to_onnx_graph(),
            ir_version=self.ir_version,
            opset_imports=self.opset_imports,
        )

    @classmethod
    def from_onnx(cls, model: ModelProto):
        obj = cls(
            graph=Graph.from_onnx(model.graph),
            ir_version=model.ir_version,
        )
        for opset in model.opset_import:
            obj.add_opset(opset.domain, opset.version)
        return obj

    def update_from_onnx(self, model: ModelProto):
        self.graph = Graph.from_onnx(model.graph)
        self.ir_version = model.ir_version
        self.opset_imports = model.opset_import

    @property
    def graphs(self):
        q = []
        q.append(self.graph)
        while q:
            g = q.pop()
            yield g
            for n in g.nodes:
                for a in n.attrs:
                    if a.attr_type == AttributeProto.GRAPH:
                        q.append(a.value)
                    if a.attr_type == AttributeProto.GRAPHS:
                        for g in a.value:
                            q.append(g)

    @property
    def nodes(self):
        for g in self.graphs:
            for n in g.nodes:
                yield n

    @property
    def tensors(self):
        for g in self.graphs:
            for t in g.value_info:
                yield t

    @property
    def edges(self):
        for g in self.graphs:
            input_name_to_nodes = g.input_name_to_nodes()
            output_name_to_node = g.output_name_to_node()
            for i, n1 in output_name_to_node.items():
                ns = input_name_to_nodes.get(i, [])
                for n2 in ns:
                    yield g, i, n1, n2
        input_name_to_nodes = self.graph.input_name_to_nodes()
        for i in self.graph.inputs:
            for n in input_name_to_nodes[i.name]:
                yield self.graph, i.name, None, n
        output_name_to_node = self.graph.output_name_to_node()
        for o in self.graph.outputs:
            yield self.graph, o.name, output_name_to_node[o.name], None

    @property
    def initializers(self):
        for g in self.graphs:
            yield from g.initializers

    @property
    def name_to_tensor(self):
        name_to_tensor = weakref.WeakValueDictionary()
        for g in self.graphs:
            for t in g.name_to_tensor.values():
                name_to_tensor[t.name] = t
        return name_to_tensor

    @property
    def name_to_node(self):
        d = weakref.WeakValueDictionary()
        for n in self.nodes:
            d[n.name] = n
        return n

    @property
    def value_info(self):
        value_info = WeakList()
        for g in self.graphs:
            value_info.extend(g.value_info)
        return value_info

    def input_name_to_nodes(self):
        input_name_to_nodes = {}
        for n in self.nodes:
            for i in n.inputs:
                if i:
                    if i not in input_name_to_nodes:
                        input_name_to_nodes[i] = []
                    input_name_to_nodes[i].append(n)
        return input_name_to_nodes

    def output_name_to_node(self):
        output_name_to_node = weakref.WeakValueDictionary()
        for n in self.nodes:
            for o in n.outputs:
                output_name_to_node[o] = n
        return output_name_to_node

    def nodes_by_op_type(self, op_type):
        for n in self.nodes:
            if n.op_type == op_type:
                yield n

    def get_parents(self, node: Node, output_name_to_node: Dict[str, Node]):
        parents = WeakList()
        for i in node.inputs:
            if i and i in output_name_to_node:
                parents.append(output_name_to_node[i])
        return parents

    def get_children(self, node: Node, input_name_to_nodes: Dict[str, List[Node]]):
        children = WeakList()
        for o in node.outputs:
            if o and o in input_name_to_nodes:
                children.extend(input_name_to_nodes[o])
        return children

    def update(self):
        used_tensor_name = set()
        for node in self.nodes:
            for i in node.inputs:
                if i:
                    used_tensor_name.add(i)
            for o in node.outputs:
                if o:
                    used_tensor_name.add(o)
        for g in self.graphs:
            keys = list(g.name_to_tensor.keys())
            for k in keys:
                if k not in used_tensor_name:
                    g.name_to_tensor.remove(k)
            g.update()

    def remove_node(self, node: Node):
        for g in self.graphs:
            if node.name in g.name_to_node:
                g.name_to_node.remove(node.name)

    def rename_input(self, old_name: str, new_name: str):
        for g in self.graphs:
            g.input_names = [
                new_name if i == old_name else i for i in self.graph.input_names
            ]
        for node in self.nodes:
            node.inputs = [new_name if i == old_name else i for i in node.inputs]
        t = self.graph.name_to_tensor[old_name].clone()
        self.graph.remove_tensor(old_name, recursive=True)
        t.name = new_name
        self.graph.add_tensor(t, recursive=True)
        self.update()

    def rename_output(self, old_name: str, new_name: str):
        for g in self.graphs:
            g.output_names = [
                new_name if i == old_name else i for i in self.graph.output_names
            ]
        for node in self.nodes:
            node.outputs = [new_name if i == old_name else i for i in node.outputs]
        t = self.graph.name_to_tensor[old_name].clone()
        self.graph.remove_tensor(old_name, recursive=True)
        t.name = new_name
        self.graph.add_tensor(t, recursive=True)
        self.update()

    def get_graph_by_node(self, node: Node) -> Graph:
        for graph in self.graphs:
            for n in graph.nodes:
                if n.name == node.name:
                    return graph
        return None

    def save(self, path: str):
        try:
            onnx.checker.check_model(self.to_onnx_model(), full_check=True)
        except:
            logger.exception("Model is not valid")
        with open(path, "wb") as f:
            f.write(self.to_onnx_model().SerializeToString())

    @classmethod
    def load(cls, path: str):
        model = ModelProto()
        with open(path, "rb") as f:
            model.ParseFromString(f.read())
        return cls.from_onnx(model)

    def add_prefix(self, prefix: str, include_shape=True):
        for g in self.graphs:
            g.add_prefix(prefix, include_shape)

    def topological_sort(self):
        all_initializers = []
        self.graph.topological_sort(all_initializers)

    def prune(self):
        for g in self.graphs:
            g.prune()

    @staticmethod
    def _infer_shapes(model):
        model = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.onnx")
            onnx.save(model, path)
            onnx.shape_inference.infer_shapes_path(path, data_prop=True)
            with contextlib.suppress(Exception):
                onnx.shape_inference.infer_shapes_path(
                    path, strict_mode=True, data_prop=True
                )
            model = onnx.load(path)
        with contextlib.suppress(Exception):
            model = SymbolicShapeInference.infer_shapes(
                model, 2**31 - 1, True, False, 3
            )
        return model

    def infer_shapes(self):
        model = self.to_onnx_model()
        model = self._infer_shapes(model)
        self.update_from_onnx(model)

    def opt_by_rt(self, enable_extened=False, enable_all=False, disable_all=False):
        self.clean()
        model = self.to_onnx_model()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.onnx")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            if enable_extened:
                so.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                )
            if enable_all:
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if disable_all:
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so.optimized_model_filepath = path
            sess = ort.InferenceSession(model.SerializeToString(), so)
            for _ in range(2):
                inps = self.get_random_inputs()
                sess.run(None, inps)
            m = onnx.load(path)
        self.update_from_onnx(m)
        self.infer_shapes()
        self.topological_sort()

    def matmul_opt(self):
        trns = TransformCompose([MergeMatMul(), FuseMulMatMul()])
        trns(self)

    def simplify(self):
        self.matmul_opt()
        self.clean()
        model = self.to_onnx_model()
        model, check = simplify(model)
        assert check, "Simplification failed"
        self.update_from_onnx(model)
        self.clean()

    def infer_for_dq(self):
        self.prune()
        self.topological_sort()
        name_to_tensor = self.name_to_tensor
        for node in self.nodes:
            if node.op_type == "DynamicQuantizeLinear":
                i1 = node.inputs[0]
                o1 = node.outputs[0]
                o2 = node.outputs[1]
                o3 = node.outputs[2]
                i1 = name_to_tensor[i1]
                o1 = name_to_tensor[o1]
                o2 = name_to_tensor[o2]
                o3 = name_to_tensor[o3]
                o1.dtype = DType.UINT8
                o2.dtype = DType.FLOAT
                o3.dtype = DType.UINT8
                o1.shape = i1.shape
            elif node.op_type == "MatMulInteger":
                o1 = node.outputs[0]
                o1 = name_to_tensor[o1]
                o1.dtype = DType.INT32
                i0 = node.inputs[0]
                i1 = node.inputs[1]
                i0 = name_to_tensor[i0].clone()
                i1 = name_to_tensor[i1].clone()
                o1 = o1.clone()
                o1.name = o1.name + "_tmp"
                i0.name = i0.name + "_dq"
                i1.name = i1.name + "_dq"
                i0.to(DType.FLOAT)
                i1.to(DType.FLOAT)
                o1.to(DType.FLOAT)
                graph = self.get_graph_by_node(node)
                graph.add_tensor(i0)
                graph.add_tensor(i1)
                graph.add_tensor(o1)
                n = self.make_node(
                    name=node.name + "_float",
                    op_type="MatMul",
                    inputs=[i0.name, i1.name],
                    outputs=[o1.name],
                    attrs=node.attrs,
                )
                graph.add_node(n)
                m = self.create_model_from_nodes(n)
                m.infer_shapes()
                o1.shape = m.name_to_tensor[o1.name].shape
                self.remove_node(n)
                del n, m
                self.update()
                self.prune()
            elif node.op_type == "ConvInteger":
                o1 = node.outputs[0]
                o1 = name_to_tensor[o1]
                o1.dtype = DType.INT32
                i0 = node.inputs[0]
                i1 = node.inputs[1]
                i0 = name_to_tensor[i0].clone()
                i1 = name_to_tensor[i1].clone()
                o1 = o1.clone()
                o1.name = o1.name + "_tmp"
                i0.name = i0.name + "_dq"
                i1.name = i1.name + "_dq"
                i0.to(DType.FLOAT)
                i1.to(DType.FLOAT)
                o1.to(DType.FLOAT)
                graph = self.get_graph_by_node(node)
                graph.add_tensor(i0)
                graph.add_tensor(i1)
                graph.add_tensor(o1)
                n = self.make_node(
                    name=node.name + "_float",
                    op_type="Conv",
                    inputs=[i0.name, i1.name],
                    outputs=[o1.name],
                    attrs=node.attrs,
                )
                graph.add_node(n)
                m = self.create_model_from_nodes(n)
                m.infer_shapes()
                o1.shape = m.name_to_tensor[o1.name].shape
                self.remove_node(n)
                del n, m
                self.update()
                self.prune()
        self.update()

    def infer_dtypes(self):
        for node in self.nodes:
            if node.op_type in {
                "DynamicQuantizeLinear",
                "MatMulInteger",
                "ConvInteger",
            }:
                continue
            schema = self.get_schema(node.op_type)
            if schema is not None:
                with contextlib.suppress(KeyError):
                    schema.infer_output_types(node, self.name_to_tensor)

        self.update()

    def clean(self):
        self.topological_sort()
        self.prune()
        self.infer_shapes()
        self.topological_sort()

    def make_tensor(
        self, name: str, dtype: DType = None, shape: List[int] = None, data: Any = None
    ):
        return Tensor(name=name, dtype=dtype, shape=shape, data=data)

    def make_node(
        self,
        name: str,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        attrs: List[Attribute] = None,
        domain: str = "",
    ):
        if attrs is None:
            attrs = []
        node = Node(
            name=name,
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            domain=domain,
        )
        return node

    def make_attr(self, name: str, value: Any):
        return Attribute(name=name, value=value)

    def build_float_fake_nodes(
        self,
        fake_node_name: str,
        unfake_node_name: str,
        tensor: Tensor,
        fake_tensor: Tensor,
        unfake_tensor: Tensor,
        from_dtype: DType,
        to_dtype: DType,
    ):
        fake_tensor.to(to_dtype)
        fake_node = self.make_node(
            name=fake_node_name,
            op_type="Cast",
            inputs=[tensor.name],
            outputs=[fake_tensor.name],
            attrs=[
                Attribute(name="to", value=to_dtype.value),
            ],
        )
        unfake_node = self.make_node(
            name=unfake_node_name,
            op_type="Cast",
            inputs=[fake_tensor.name],
            outputs=[unfake_tensor.name],
            attrs=[
                Attribute(name="to", value=from_dtype.value),
            ],
        )
        return fake_node, unfake_node

    def build_quant_node(
        self,
        node_name: str,
        tensor: Tensor,
        output_tensor: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        axis: int = -1,
    ):
        assert scale.dtype == DType.FLOAT
        assert zero_point.dtype == DType.INT8 or zero_point.dtype == DType.UINT8
        output_tensor.to(zero_point.dtype)

        node = self.make_node(
            name=node_name,
            op_type="QuantizeLinear",
            inputs=[tensor.name, scale.name, zero_point.name],
            outputs=[output_tensor.name],
            attrs=[],
        )
        return node

    def build_dequant_node(
        self,
        node_name: str,
        tensor: Tensor,
        output_tensor: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        axis: int = -1,
    ):
        assert scale.dtype == DType.FLOAT
        assert zero_point.dtype == DType.INT8 or zero_point.dtype == DType.UINT8
        tensor.to(zero_point.dtype)

        node = self.make_node(
            name=node_name,
            op_type="DequantizeLinear",
            inputs=[tensor.name, scale.name, zero_point.name],
            outputs=[output_tensor.name],
            attrs=[],
        )
        return node

    def build_qdq_fake_nodes(
        self,
        fake_node_name: str,
        unfake_node_name: str,
        tensor: Tensor,
        fake_tensor: Tensor,
        unfake_tensor: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        axis: int = -1,
    ):
        assert scale.dtype == DType.FLOAT
        assert zero_point.dtype == DType.INT8 or zero_point.dtype == DType.UINT8
        fake_tensor.to(zero_point.dtype)

        fake_node = self.make_node(
            name=fake_node_name,
            op_type="QuantizeLinear",
            inputs=[tensor.name, scale.name, zero_point.name],
            outputs=[fake_tensor.name],
            attrs=[
                Attribute(name="axis", value=axis),
            ],
        )
        unfake_node = self.make_node(
            name=unfake_node_name,
            op_type="DequantizeLinear",
            inputs=[fake_tensor.name, scale.name, zero_point.name],
            outputs=[unfake_tensor.name],
            attrs=[
                Attribute(name="axis", value=axis),
            ],
        )
        return fake_node, unfake_node

    def calc_scale_zero_point(
        self,
        op_type: str,
        signed: bool = False,
        n_trials=5,
        bins=8192,
        bin_width=4e-3,
    ):
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()
        hists = collections.defaultdict(lambda: np.zeros(bins + 1, dtype=np.float64))
        bin_range = bins * bin_width
        bin_min = -bin_range / 2
        bin_max = bin_range / 2

        def float_to_bin(x):
            x = np.clip(x, bin_min, bin_max)
            return ((x - bin_min) / bin_width).astype(np.int64)

        model = self.clone()
        tensor_name_to_graph = {}
        for node in model.nodes:
            graph = model.get_graph_by_node(node)
            for o in node.outputs:
                cond1 = (
                    o in output_name_to_node
                    and output_name_to_node[o].op_type != op_type
                )
                cond2 = all(
                    n.op_type != op_type for n in input_name_to_nodes.get(o, [])
                )
                cond3 = node.op_type != op_type
                if cond1 and cond2 and cond3:
                    continue
                tensor_name_to_graph[o] = graph
                t = graph.name_to_tensor[o]
                if t.dtype != DType.FLOAT:
                    continue
                graph.output_names.append(o)
                tensor_name_to_graph[o] = self.get_graph_by_node(node)
        model.update()

        for _ in tqdm(range(n_trials), desc="calc_scale_zero_point"):
            inps = model.get_random_inputs()
            o = model.forward(**inps)
            for k, v in o.items():
                idx = float_to_bin(v)
                hists[k][idx] += 1
            for k, v in inps.items():
                idx = float_to_bin(v)
                hists[k][idx] += 1

        scales = {}
        mins = {}
        qrange = 255 if not signed else 254
        for k, v in hists.items():
            v /= v.sum()
            cdf = np.cumsum(v)
            q1 = np.argmax(cdf > 0.01) * bin_width + bin_min
            q99 = np.argmax(cdf > 0.99) * bin_width + bin_min
            if signed:
                mins[k] = max(abs(q1), abs(q99))
                scales[k] = mins[k] / qrange * 2
            else:
                mins[k] = q1
                scales[k] = (q99 - q1) / qrange + 1e-8

        for i in self.initializers:
            v = i.data
            q1 = np.quantile(v, 0.01)
            q99 = np.quantile(v, 0.99)
            mins[i.name] = q1
            scales[i.name] = (q99 - q1) / qrange + 1e-8

        scale = {}
        zero_point = {}
        qmin = 0 if not signed else -127
        for k in scales.keys():
            s = np.float32([scales[k]]).astype(np.float32)
            zp = np.int64([qmin - mins[k] / s]).flatten()
            if signed:
                zp = np.array([0]).astype(np.int8).flatten()
            else:
                zp = np.clip(zp, 0, 255).astype(np.uint8).flatten()
            graph = tensor_name_to_graph.get(k, self.graph)
            scale[k] = Tensor(
                name=k + "_qscale",
                dtype=DType.FLOAT,
                shape=[],
                data=s,
            )
            zero_point[k] = Tensor(
                name=k + "_qzero_point",
                dtype=DType.UINT8 if not signed else DType.INT8,
                shape=[],
                data=zp,
            )
            graph.add_tensor(scale[k])
            graph.add_tensor(zero_point[k])
        return scale, zero_point

    def fake(self, from_dtype: DType, to_dtype: DType, quantize: bool = False):
        fake_node_suffix = f"fake_{from_dtype.name}_to_{to_dtype.name}"
        unfake_node_suffix = f"unfake_{from_dtype.name}_to_{to_dtype.name}"
        fake_tensor_suffix = f"faked_{from_dtype.name}_to_{to_dtype.name}"
        unfaked_tensor_suffix = f"unfaked_{from_dtype.name}_to_{to_dtype.name}"
        edges = list(self.edges)
        if quantize:
            scale, zero_point = self.calc_scale_zero_point()

        for g, i, n1, n2 in tqdm(edges):
            i = g.name_to_tensor[i]
            if i.dtype != from_dtype:
                continue
            faked_i = i.clone()
            unfaked_i = i.clone()
            n1_name = n1.name if n1 is not None else "input"
            n2_name = n2.name if n2 is not None else "output"
            faked_i.name = f"{n1_name}_{n2_name}_{i.name}_{fake_tensor_suffix}"
            unfaked_i.name = f"{n1_name}_{n2_name}_{i.name}_{unfaked_tensor_suffix}"
            fake_node_name = f"fake_{n1_name}_{n2_name}_{i.name}_{fake_node_suffix}"
            unfake_node_name = (
                f"unfake_{n1_name}_{n2_name}_{i.name}_{unfake_node_suffix}"
            )
            org_name = i.name
            if quantize and not (i.name in scale and i.name in zero_point):
                continue

            if n2 is None:
                # output case
                unfaked_output = i.clone()
                unfaked_output.name = (
                    f"output_{n1_name}_{n2_name}_{i.name}_{unfake_node_suffix}"
                )
                i.name, faked_i.name = faked_i.name, i.name
                unfaked_output.name, faked_i.name = (
                    faked_i.name,
                    unfaked_output.name,
                )
                n1.outputs[n1.outputs.index(unfaked_output.name)] = i.name

                if quantize:
                    output_unfake_node = self.make_node(
                        name=f"{n1_name}_{n2_name}_{unfaked_output.name}_{unfake_node_suffix}",
                        op_type="DequantizeLinear",
                        inputs=[
                            faked_i.name,
                            scale[org_name].name,
                            zero_point[org_name].name,
                        ],
                        outputs=[unfaked_output.name],
                        attrs=[
                            Attribute(name="axis", value=-1),
                        ],
                    )
                else:
                    output_unfake_node = self.make_node(
                        name=f"{n1_name}_{n2_name}_{unfaked_output.name}_{unfake_node_suffix}",
                        op_type="Cast",
                        inputs=[faked_i.name],
                        outputs=[unfaked_output.name],
                        attrs=[Attribute("to", from_dtype.value)],
                    )

                g.add_tensor(unfaked_output)
                g.add_node(output_unfake_node)
                self.update()
            if quantize:
                fake_node, unfake_node = self.build_qdq_fake_nodes(
                    fake_node_name=fake_node_name,
                    unfake_node_name=unfake_node_name,
                    tensor=i,
                    fake_tensor=faked_i,
                    unfake_tensor=unfaked_i,
                    scale=scale[org_name],
                    zero_point=zero_point[org_name],
                )
            else:
                fake_node, unfake_node = self.build_float_fake_nodes(
                    fake_node_name=fake_node_name,
                    unfake_node_name=unfake_node_name,
                    tensor=i,
                    fake_tensor=faked_i,
                    unfake_tensor=unfaked_i,
                    from_dtype=from_dtype,
                    to_dtype=to_dtype,
                )
            if n2 is not None:
                n2.inputs[n2.inputs.index(i.name)] = unfaked_i.name
            g.add_tensor(faked_i)
            g.add_tensor(unfaked_i)
            g.add_node(fake_node)
            g.add_node(unfake_node)

        input_name_to_nodes = self.input_name_to_nodes()
        for i in self.initializers:
            if i.dtype != from_dtype:
                continue
            for n in input_name_to_nodes[i.name]:
                faked_i = i.clone()
                unfaked_i = i.clone()
                faked_i.name = f"{n.name}_{i.name}_{fake_tensor_suffix}"
                unfaked_i.name = f"{n.name}_{i.name}_{unfaked_tensor_suffix}"
                fake_node_name = f"fake_{n.name}_{i.name}_{fake_node_suffix}"
                unfake_node_name = f"unfake_{n.name}_{i.name}_{unfake_node_suffix}"
                org_name = i.name
                if quantize and not (i.name in scale and i.name in zero_point):
                    continue

                if quantize:
                    fake_node, unfake_node = self.build_qdq_fake_nodes(
                        fake_node_name=fake_node_name,
                        unfake_node_name=unfake_node_name,
                        tensor=i,
                        fake_tensor=faked_i,
                        unfake_tensor=unfaked_i,
                        scale=scale[org_name],
                        zero_point=zero_point[org_name],
                    )
                else:
                    fake_node, unfake_node = self.build_float_fake_nodes(
                        fake_node_name=fake_node_name,
                        unfake_node_name=unfake_node_name,
                        tensor=i,
                        fake_tensor=faked_i,
                        unfake_tensor=unfaked_i,
                        from_dtype=from_dtype,
                        to_dtype=to_dtype,
                    )
                n.inputs[n.inputs.index(i.name)] = unfaked_i.name
                g.add_tensor(faked_i)
                g.add_tensor(unfaked_i)
                g.add_node(fake_node)
                g.add_node(unfake_node)

        self.update()
        self.prune()
        self.topological_sort()

    def realize_float(self, from_dtype: DType, to_dtype: DType):
        nodes = list(self.nodes)
        input_name_to_nodes = self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()
        name_to_tensor = self.name_to_tensor
        for n in tqdm(nodes):
            if n.name.startswith("fake_") or n.name.startswith("unfake_"):
                continue
            flag = False
            for i in n.inputs:
                if not i:
                    continue
                i = name_to_tensor[i]
                if i.dtype == from_dtype and i.name in output_name_to_node:
                    c = output_name_to_node[i.name]
                    if not (c.name.startswith("unfake_") and c.op_type == "Cast"):
                        flag = True
            for i in n.outputs:
                i = name_to_tensor[i]
                if i.dtype == from_dtype and i.name in input_name_to_nodes:
                    for p in input_name_to_nodes[i.name]:
                        if not (p.name.startswith("fake_") and p.op_type == "Cast"):
                            flag = True
            if flag:
                continue

            # try to cast/quantize
            m = self.create_model_from_nodes(n)
            for i in m.tensors:
                if (
                    i.dtype == from_dtype
                    and not i.name.endswith("_qscale")
                    and not i.name in self.graph.output_names
                ):
                    i.to(to_dtype)
            m.update()
            schema = self.get_schema(n.op_type)
            if schema is not None:
                if not schema.check_types(list(m.nodes)[0], m.graph.name_to_tensor):
                    logger.warning(f"node {n.name} can not be casted/quantized, skip")
                    continue
            try:
                m.try_forward()
            except Exception as e:
                logger.warning(
                    f"node {n.name} can not be casted/quantized, skip. Error: {e}"
                )
                continue
            del m

            # if succeed, replace the node
            for i in n.inputs:
                if not i:
                    continue
                i = name_to_tensor[i]
                if not (
                    i.dtype == from_dtype
                    and not i.name.endswith("_qscale")
                    and not i.name in self.graph.output_names
                ):
                    continue
                p = output_name_to_node[i.name]
                assert p.op_type == "Cast"
                orgin_name = p.inputs[0]
                n.inputs[n.inputs.index(i.name)] = orgin_name
                i.name = orgin_name
                i.to(to_dtype)
            for i in n.outputs:
                i = name_to_tensor[i]
                if i.dtype != from_dtype:
                    continue
                i.to(to_dtype)
                if i.name not in input_name_to_nodes:
                    continue
                c = input_name_to_nodes[i.name][0]
                assert c.op_type == "Cast"
                if c.outputs[0] not in input_name_to_nodes:
                    continue
                for origin_n in input_name_to_nodes[c.outputs[0]]:
                    origin_n.inputs[origin_n.inputs.index(c.outputs[0])] = i.name
        self.update()
        self.clean()

    def create_model_from_nodes(self, nodes: Union[List[Node], Node]):
        if not isinstance(nodes, list):
            nodes = [nodes]
        nodes = copy.deepcopy(nodes)
        node_input_names = set()
        node_output_names = set()
        for n in nodes:
            for i in n.inputs:
                node_input_names.add(i)
            for o in n.outputs:
                node_output_names.add(o)
        initializers = [i.name for i in self.initializers if i.name in node_input_names]
        input_names = []
        output_names = []
        for n in nodes:
            for i in n.inputs:
                if (
                    i
                    and i not in node_output_names
                    and i not in initializers
                    and i not in input_names
                ):
                    input_names.append(i)
            for o in n.outputs:
                if o not in node_input_names and o not in output_names:
                    output_names.append(o)
        tensor_names = input_names + output_names + initializers
        name_to_tensor = {
            v.name: v.clone() for v in self.value_info if v.name in tensor_names
        }
        assert set(tensor_names) == set(name_to_tensor.keys())

        tensors = list(name_to_tensor.values())
        graph = Graph(
            name="_".join([n.name for n in nodes]),
            nodes=nodes,
            input_names=input_names,
            output_names=output_names,
            tensors=tensors,
        )
        m = Model(
            graph=graph, ir_version=self.ir_version, opset_imports=self.opset_imports
        )
        return m

    def build_session(
        self,
        enable_extended=False,
        enable_all=False,
        disable_all=False,
        enable_profile=False,
        execute_parallel=False,
    ):
        so = ort.SessionOptions()
        so.execution_order = ort.ExecutionOrder.DEFAULT
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if not self._verbose:
            so.log_severity_level = 3
        if enable_extended:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if enable_all:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if disable_all:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        if enable_profile:
            so.enable_profiling = True
        if execute_parallel:
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        return ort.InferenceSession(self.to_onnx_model().SerializeToString(), so)

    def build_input(self, *args, **kwargs):
        inp = {}
        for i, a in enumerate(args):
            inp[self.graph.inputs[i].name] = a
        for k, v in kwargs.items():
            inp[k] = v
        return inp

    def forward(self, *args, **kwargs):
        input_names = [i.name for i in self.graph.inputs]
        inp = {}
        tensor_dtype = {t.name: t.dtype for t in self.graph.value_info}
        for i in self.graph.inputs:
            if i.name not in tensor_dtype:
                tensor_dtype[i.name] = i.dtype
        for k, v in self.build_input(*args, **kwargs).items():
            if k in input_names:
                dtype = get_numpy_dtype(tensor_dtype[k])
                if v.dtype != dtype and self._verbose:
                    logger.warning(
                        f"Input {k} has dtype {v.dtype} but model expects {dtype}."
                    )
                    v = v.astype(dtype)
                inp[k] = v
        sess = self.build_session(disable_all=True)
        out = sess.run(None, inp)
        oup = {}
        for i, o in enumerate(out):
            oup[self.graph.outputs[i].name] = o
        return oup

    def time(self, **kwargs):
        inp = self.get_random_inputs()
        sess = self.build_session(**kwargs)
        sess.run(None, inp)
        s = time.time()
        for _ in range(100):
            sess.run(None, inp)
        e = time.time()
        avg_t = (e - s) / 100
        return avg_t

    def profile(self, **kwargs):
        inp = self.get_random_inputs()
        sess = self.build_session(enable_profile=True, **kwargs)
        for _ in range(100):
            sess.run(None, inp)
        prof_file = sess.end_profiling()
        with open(prof_file, "r") as f:
            results = json.load(f)
        os.remove(prof_file)
        kernel_times = []
        index = 0
        for result in results:
            if result["cat"] == "Session" and result["name"] == "model_run":
                index += 1
            if result.get("name", "").endswith("_kernel_time"):
                node_name = result["name"].split("_kernel_time")[0]
                dur = int(result["dur"])
                ts = int(result["ts"])
                result = result["args"]

                op_name = result["op_name"]
                graph_index = result["graph_index"]
                exec_plan_index = result["exec_plan_index"]
                activation_size = result["activation_size"]
                parameter_size = result["parameter_size"]
                output_size = result["output_size"]
                kernel_times.append(
                    dict(
                        index=index,
                        node_name=node_name,
                        op_name=op_name,
                        graph_index=graph_index,
                        exec_plan_index=exec_plan_index,
                        activation_size=activation_size,
                        parameter_size=parameter_size,
                        output_size=output_size,
                        dur=dur,
                        start_at=ts - dur,
                        end_at=ts,
                    )
                )
        kernel_times = sorted(kernel_times, key=lambda x: x["start_at"])
        df = pd.DataFrame(kernel_times)
        return df

    def set_test_shape(self, shape):
        self._test_shape = shape

    def get_random_inputs(self):
        random_inputs = {}
        for i in self.graph.inputs:
            shape = i.shape
            if not shape:
                shape = [1]
            shape = [
                s if isinstance(s, int) else self._test_shape.get(s, 1) for s in shape
            ]
            if i.dtype not in {DType.INT64, DType.INT32}:
                random_inputs[i.name] = np.random.rand(*shape).astype(
                    get_numpy_dtype(i.dtype)
                )
            else:
                random_inputs[i.name] = np.random.randint(
                    0, 2, size=shape, dtype=get_numpy_dtype(i.dtype)
                )
        return random_inputs

    def try_forward(self):
        self._verbose = False
        random_inputs = self.get_random_inputs()
        try:
            self.forward(**random_inputs)
        except Exception as e:
            self._verbose = True
            raise e
        self._verbose = True

    def dynamic_quantize(self):
        model = self.to_onnx_model()
        if self.default_opset < 11:
            model = version_converter.convert_version(model, 11)
            self.update_default_opset(11)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.onnx")
            qmodel_path = os.path.join(tmpdir, "qmodel.onnx")
            onnx.save(model, model_path)

            quantize_dynamic(
                model_input=model_path,
                model_output=qmodel_path,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
                per_channel=True,
                reduce_range=True,
                weight_type=QuantType.QUInt8,
                optimize_model=False,
            )
            qmodel = onnx.load(qmodel_path)
        self.update_from_onnx(qmodel)
        self.prune()
        self.infer_for_dq()
        self.infer_dtypes()
        self.simplify()

    def half(self):
        self.fake(DType.FLOAT, DType.FLOAT16)
        self.realize_float(DType.FLOAT, DType.FLOAT16)
        self.infer_dtypes()
        self.simplify()

    def float(self):
        self.fake(DType.FLOAT16, DType.FLOAT)
        self.realize_float(DType.FLOAT16, DType.FLOAT)
        self.infer_dtypes()
        self.simplify()

    def double(self):
        self.float()
        self.fake(DType.FLOAT, DType.DOUBLE)
        self.realize_float(DType.FLOAT, DType.DOUBLE)
        self.infer_dtypes()
        self.simplify()

    def rename_shape(
        self,
        old_name: str,
        new_name: str,
    ):
        for t in self.tensors:
            t.rename_shape(old_name, new_name)
        self.update()
        self.infer_shapes()

    def to_dynamic(
        self,
        axis: int = 0,
        shape_name: str = "batch_size",
        input_names: List[str] = None,
    ):
        if input_names is None:
            input_names = [i.name for i in self.graph.inputs]

        name_to_tensor = self.name_to_tensor
        for i in input_names:
            i = name_to_tensor[i]
            i.mod_shape(axis, shape_name)
        self.update()
        self.infer_shapes()

    def decompose_simplified_ln(self):
        trns = TransformCompose([DecomposeSLN()])
        trns(self)

    def share_initializers(self):
        all_initializers = []
        initializer_to_graph = {}
        for g in self.graphs:
            g.remove_duplicated_initializers()
            for t in g.initializers:
                all_initializers.append(t)
                initializer_to_graph[t] = g

        shared_initializers = []
        seen = set()
        for t1 in all_initializers:
            if t1 in seen:
                continue
            t = t1.clone()
            t.name = "shared_" + t.name
            t.name = t.name.replace("then_", "").replace("else_", "")
            for t2 in all_initializers:
                if t2 in seen:
                    continue
                if t1 is t2:
                    continue
                if t1.has_same_data(t2):
                    if t not in shared_initializers:
                        g1 = initializer_to_graph[t1]
                        g1.remove_tensor(t1.name, recursive=True)
                        g1.add_tensor(Tensor.from_onnx(t.to_onnx_tensor_value_info()))
                        seen.add(t1)
                        for n1 in g1.nodes:
                            n1.replace_input(t1.name, t.name)
                        shared_initializers.append(t)
                    g2 = initializer_to_graph[t2]
                    g2.remove_tensor(t2.name, recursive=True)
                    g2.add_tensor(Tensor.from_onnx(t.to_onnx_tensor_value_info()))
                    for n2 in g2.nodes:
                        n2.replace_input(t2.name, t.name)
                    seen.add(t2)
        for t in shared_initializers:
            self.graph.add_tensor(t)
