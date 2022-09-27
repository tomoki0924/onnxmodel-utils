import numpy as np

from .base import TransformBase


class DecomposeSLN(TransformBase):
    op_types = ["SimplifiedLayerNormalization"]

    def prepare(self, model):
        if "name_to_tensor" not in self.context:
            self.context["name_to_tensor"] = model.name_to_tensor

    def forward(
        self,
        node,
        model,
        nodes_to_remove,
    ):
        self.prepare(model)
        graph = model.get_graph_by_node(node)
        name_to_tensor = self.context["name_to_tensor"]

        input_name = node.inputs[0]
        output_name = node.outputs[0]
        node_name = node.name
        input_tensor = name_to_tensor[input_name]
        dtype = input_tensor.dtype

        eps = node.get_attr_by_name("epsilon").value
        reduced_shape = input_tensor.shape.copy()
        reduced_shape[-1] = 1

        powed = model.make_tensor(
            name=f"{input_name}_{node_name}_powed",
            dtype=dtype,
            shape=input_tensor.shape,
        )
        two = model.make_tensor(
            name=f"{input_name}_{node_name}_two",
            dtype=dtype,
            shape=[],
            data=np.array([2.0]).flatten(),
        )

        pow_node = model.make_node(
            name=f"{input_name}_{node_name}_powed",
            op_type="Pow",
            inputs=[input_name, two.name],
            outputs=[powed.name],
        )

        mu = model.make_tensor(
            name=f"{input_name}_{node_name}_mu",
            dtype=dtype,
            shape=reduced_shape,
        )
        reduce_mean_node = model.make_node(
            name=f"{input_name}_{node_name}_reduce_mean",
            op_type="ReduceMean",
            inputs=[powed.name],
            outputs=[mu.name],
            attrs=[
                model.make_attr("axes", [-1]),
                model.make_attr("keepdims", 1),
            ],
        )

        mu_eps = model.make_tensor(
            name=f"{input_name}_{node_name}_mu_eps",
            dtype=dtype,
            shape=mu.shape,
        )
        eps = model.make_tensor(
            name=f"{input_name}_{node_name}_eps",
            dtype=dtype,
            shape=[],
            data=np.array(eps),
        )
        add_node = model.make_node(
            name=f"{input_name}_{node_name}_add",
            op_type="Add",
            inputs=[mu.name, eps.name],
            outputs=[mu_eps.name],
        )

        sqrt = model.make_tensor(
            name=f"{input_name}_{node_name}_sqrt",
            dtype=dtype,
            shape=mu.shape,
        )
        sqrt_node = model.make_node(
            name=f"{input_name}_{node_name}_sqrt",
            op_type="Sqrt",
            inputs=[mu_eps.name],
            outputs=[sqrt.name],
        )

        normed = model.make_tensor(
            name=f"{input_name}_{node_name}_normed",
            dtype=dtype,
            shape=input_tensor.shape,
        )

        div_node = model.make_node(
            name=f"{input_name}_{node_name}_div",
            op_type="Div",
            inputs=[input_name, sqrt.name],
            outputs=[normed.name],
        )

        gamma = name_to_tensor[node.inputs[1]]
        mul_node = model.make_node(
            name=f"{input_name}_{node_name}_mul",
            op_type="Mul",
            inputs=[normed.name, gamma.name],
            outputs=[output_name],
        )

        nodes_to_remove.append(node)
        for n in [pow_node, reduce_mean_node, add_node, sqrt_node, div_node, mul_node]:
            graph.add_node(n)
        for t in [powed, two, mu, mu_eps, eps, sqrt, normed]:
            graph.add_tensor(t)
