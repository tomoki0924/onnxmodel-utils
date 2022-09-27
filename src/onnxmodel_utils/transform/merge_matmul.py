import weakref

import numpy as np

from .base import TransformBase


class MergeMatMul(TransformBase):
    op_types = ["MatMul"]

    def __init__(self, n_max=16):
        super().__init__()
        self.done_nodes = weakref.WeakSet()
        self.n_max = n_max

    def prepare(self, model):
        if "name_to_tensor" not in self.context:
            self.context["name_to_tensor"] = model.name_to_tensor
        if "input_name_to_nodes" not in self.context:
            self.context["input_name_to_nodes"] = model.input_name_to_nodes()

    def forward(
        self,
        node,
        model,
        nodes_to_remove,
    ):
        if node in self.done_nodes:
            return
        self.prepare(model)
        graph = model.get_graph_by_node(node)
        name_to_tensor = self.context["name_to_tensor"]
        input_name_to_nodes = self.context["input_name_to_nodes"]

        input_name = node.inputs[0]
        weight_name = node.inputs[1]
        output_name = node.outputs[0]
        input_tensor = name_to_tensor[input_name]
        weight_tensor = name_to_tensor[weight_name]
        output_tensor = name_to_tensor[output_name]
        if weight_tensor.data is None:
            return

        # find bros
        if input_name not in input_name_to_nodes:
            return
        children = input_name_to_nodes[input_name]
        if node not in children:
            return
        if len(children) <= 1:
            return

        bros = [node]
        for child in children:
            if child.name == node.name:
                continue
            if child.op_type == "MatMul" and child.inputs[0] == input_name:
                w = name_to_tensor[child.inputs[1]]
                if w.data is not None:
                    bros.append(child)
        if len(bros) == 0 or len(bros) > self.n_max:
            return
        assert node in bros

        # merge bros' weights
        name = weight_name
        weights = [weight_tensor.data.copy()]
        original_outputs = [output_name]
        for bro in bros:
            w = name_to_tensor[bro.inputs[1]]
            if len(w.data.shape) != 2:
                continue
            if w.name == weight_name:
                continue
            if w.data.shape[0] != weights[-1].shape[0]:
                continue
            if w.shape[1] != weights[-1].shape[1]:
                continue
            weights.append(w.data.copy())
            name += "_" + w.name
            original_outputs.append(bro.outputs[0])
        if len(weights) <= 1:
            return

        del w
        weights = np.concatenate(weights, axis=1).copy()
        assert input_tensor.shape[-1] == weights.shape[0]

        # make merged matmul node
        merged_weight = model.make_tensor(
            name=name,
            dtype=weight_tensor.dtype,
            shape=weights.shape,
            data=weights.flatten(),
        )
        merged_output_shape = output_tensor.shape.copy()
        merged_output_shape[-1] = weights.shape[1]
        merged_output = model.make_tensor(
            name=f"{output_name}_merged",
            dtype=output_tensor.dtype,
            shape=merged_output_shape,
        )

        merged_matmul = model.make_node(
            name=f"{input_name}_{name}_matmul",
            op_type="MatMul",
            inputs=[input_name, merged_weight.name],
            outputs=[merged_output.name],
        )

        split_node = model.make_node(
            name=f"{merged_output.name}_split",
            op_type="Split",
            inputs=[merged_output.name],
            outputs=original_outputs,
            attrs=[
                model.make_attr("axis", -1),
            ],
        )

        graph = model.get_graph_by_node(node)
        for n in [merged_matmul, split_node]:
            graph.add_node(n)
        for t in [merged_weight, merged_output]:
            graph.add_tensor(t)
        for n in bros:
            nodes_to_remove.append(n)
        for bro in bros:
            self.done_nodes.add(bro)
        self.done_nodes.add(merged_matmul)
