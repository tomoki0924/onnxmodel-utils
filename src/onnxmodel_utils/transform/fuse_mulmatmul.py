import weakref

import numpy as np

from .base import TransformBase


class FuseMulMatMul(TransformBase):
    op_types = ["Mul"]

    def __init__(self):
        super().__init__()

    def prepare(self, model):
        if "name_to_tensor" not in self.context:
            self.context["name_to_tensor"] = model.name_to_tensor
        if "input_name_to_nodes" not in self.context:
            self.context["input_name_to_nodes"] = model.input_name_to_nodes()

    def forward(
        self,
        mul_node,
        model,
        nodes_to_remove,
    ):
        self.prepare(model)
        graph = model.get_graph_by_node(mul_node)
        name_to_tensor = self.context["name_to_tensor"]
        input_name_to_nodes = self.context["input_name_to_nodes"]

        mul_input_name = mul_node.inputs[0]
        mul_const_name = mul_node.inputs[1]
        mul_output_name = mul_node.outputs[0]
        mul_const = name_to_tensor[mul_const_name]
        if mul_const.data is None:
            return
        if mul_output_name not in input_name_to_nodes:
            return
        matmul_node = input_name_to_nodes[mul_output_name]
        if len(matmul_node) != 1 or matmul_node[0].op_type != "MatMul":
            return
        matmul_node = matmul_node[0]

        matmul_weight_name = matmul_node.inputs[1]
        matmul_output_name = matmul_node.outputs[0]
        matmul_weight = name_to_tensor[matmul_weight_name]
        if matmul_weight.data is None:
            return

        # fuse
        mul_w = mul_const.data
        if mul_const.shape:
            mul_w = mul_w.reshape(mul_const.shape)
        matmul_w = matmul_weight.data
        if matmul_weight.shape:
            matmul_w = matmul_w.reshape(matmul_weight.shape)
        shape = matmul_weight.shape
        if len(shape) != 2:
            return
        if np.prod(mul_w.shape) == 1:
            mul_w = mul_w * np.eye(shape[0])
        elif len(mul_w.shape) == 1:
            if len(mul_w) != shape[0]:
                return
            mul_w = np.diag(mul_w)
        else:
            return

        new_weight = np.matmul(mul_w, matmul_w)
        new_weight = model.make_tensor(
            name=f"{matmul_weight_name}_fused",
            dtype=matmul_weight.dtype,
            shape=new_weight.shape,
            data=new_weight.flatten(),
        )

        new_node = model.make_node(
            name=f"{matmul_node.name}_fused",
            op_type="MatMul",
            inputs=[mul_input_name, new_weight.name],
            outputs=[matmul_output_name],
        )

        graph.add_node(new_node)
        graph.add_tensor(new_weight)
        nodes_to_remove.append(mul_node)
        nodes_to_remove.append(matmul_node)
