from .model import Attribute, AttributeType, DType, Graph, Model, Node, Tensor


def build_if_model(
    name: str,
    cond_tensor_name: str,
    then_model: Model,
    else_model: Model,
    share_initializers: bool = True,
):
    cond_tensor = Tensor(cond_tensor_name, DType.BOOL, [1])
    then_model, else_model = _prepare(then_model, else_model)
    then_inputs = then_model.graph.input_names.copy()
    else_inputs = else_model.graph.input_names.copy()
    then_outputs = then_model.graph.output_names.copy()
    else_outputs = else_model.graph.output_names.copy()
    union_input_names = list(set(then_inputs) | set(else_inputs))
    common_input_names = list(set(then_inputs) & set(else_inputs))
    union_output_names = list(set(then_outputs) | set(else_outputs))
    common_output_names = list(set(then_outputs) & set(else_outputs))
    name_to_output_tensor = {}
    for o in union_output_names:
        if o in then_outputs:
            name_to_output_tensor[o] = then_model.graph.name_to_tensor[o].clone()
        else:
            name_to_output_tensor[o] = else_model.graph.name_to_tensor[o].clone()
    for o in union_output_names:
        if o not in common_output_names:
            if o in then_outputs:
                g1 = then_model.graph
                g2 = else_model.graph
            elif o in else_outputs:
                g1 = else_model.graph
                g2 = then_model.graph
            else:
                raise ValueError("should not happen")
            t = g1.name_to_tensor[o]
            opt_t = t.clone()
            opt_t.name = f"opt_{t.name}"
            opt_t.is_optional = True
            n1 = Node(
                name=f"optional_{t.name}_1",
                op_type="Optional",
                inputs=[t.name],
                outputs=[opt_t.name],
                attrs=[
                    Attribute(
                        name="type",
                        value=t.to_onnx_type_proto(),
                    )
                ],
            )
            n2 = Node(
                name=f"optional_{t.name}_2",
                op_type="Optional",
                inputs=[],
                outputs=[opt_t.name],
                attrs=[
                    Attribute(
                        name="type",
                        value=t.to_onnx_type_proto(),
                    )
                ],
            )
            g1.add_node(n1)
            g2.add_node(n2)
            g1.add_tensor(opt_t.clone())
            g2.add_tensor(opt_t.clone())
            g1.output_names[g1.output_names.index(t.name)] = opt_t.name
            g2.output_names.append(opt_t.name)

    assert set(then_model.graph.output_names) == set(else_model.graph.output_names)
    else_model.graph.output_names = then_model.graph.output_names.copy()
    output_names = then_model.graph.output_names.copy()
    then_model.add_prefix("then_", include_shape=False)
    else_model.add_prefix("else_", include_shape=False)
    for i1 in then_inputs:
        then_model.rename_input(f"then_{i1}", i1)
    for i2 in else_inputs:
        else_model.rename_input(f"else_{i2}", i2)
    then_graph = then_model.graph
    else_graph = else_model.graph
    nodes, tensor_list = _match_inputs(
        then_graph, else_graph, common_input_names, union_input_names, then_inputs
    )

    assert all(
        t1.dtype == t2.dtype for t1, t2 in zip(then_graph.outputs, else_graph.outputs)
    )
    attrs = []
    attrs.append(
        Attribute("then_branch", then_graph.to_onnx_graph(), AttributeType.GRAPH)
    )
    attrs.append(
        Attribute("else_branch", else_graph.to_onnx_graph(), AttributeType.GRAPH)
    )
    node = Node(
        name + "_node",
        op_type="If",
        inputs=[cond_tensor.name],
        outputs=output_names,
        attrs=attrs,
    )
    nodes.append(node)
    tensor_list.append(cond_tensor)

    graph = Graph(
        name,
        nodes,
        node.inputs + union_input_names,
        node.outputs,
        tensor_list,
    )
    ir_version = max(then_model.ir_version, else_model.ir_version, 8)
    model = Model(graph, ir_version)
    for opset in then_model.opset_imports:
        model.add_opset(opset.domain, opset.version)
    for opset in else_model.opset_imports:
        model.add_opset(opset.domain, opset.version)
    if share_initializers:
        model.share_initializers()
    return model


def build_if_model_with_cache(
    name: str,
    cache_model: Model,
    cacheless_model: Model,
    cache_names,
    share_initializers: bool = True,
):
    cache_model, cacheless_model = _prepare(cache_model, cacheless_model)
    then_inputs = cache_model.graph.input_names.copy()
    else_inputs = cacheless_model.graph.input_names.copy()
    union_input_names = list(set(then_inputs) | set(else_inputs))
    common_input_names = list(set(then_inputs) & set(else_inputs))
    assert all(c not in else_inputs for c in cache_names)
    assert all(c in then_inputs for c in cache_names)
    assert cache_model.graph.output_names == cacheless_model.graph.output_names
    outputs = cache_model.graph.output_names.copy()
    cache_model.add_prefix("then_", include_shape=False)
    cacheless_model.add_prefix("else_", include_shape=False)
    for i1 in then_inputs:
        cache_model.rename_input(f"then_{i1}", i1)
    for i2 in else_inputs:
        cacheless_model.rename_input(f"else_{i2}", i2)
    then_graph = cache_model.graph
    else_graph = cacheless_model.graph
    assert all(
        t1.dtype == t2.dtype for t1, t2 in zip(then_graph.outputs, else_graph.outputs)
    )

    nodes, tensor_list = _match_inputs(
        then_graph, else_graph, common_input_names, union_input_names, then_inputs, True
    )

    prev_name = f"{cache_names[0]}_is_opt"
    for i in range(len(cache_names) - 1):
        cond_tensor = Tensor(f"cond_{i}", DType.BOOL, [])
        cond_node = Node(
            f"cond_node_{i}",
            op_type="And",
            inputs=[prev_name, f"{cache_names[i+1]}_is_opt"],
            outputs=[cond_tensor.name],
        )
        nodes.append(cond_node.clone())
        tensor_list.append(cond_tensor.clone())
        prev_name = cond_tensor.name

    attrs = []
    attrs.append(
        Attribute("then_branch", then_graph.to_onnx_graph(), AttributeType.GRAPH)
    )
    attrs.append(
        Attribute("else_branch", else_graph.to_onnx_graph(), AttributeType.GRAPH)
    )
    if_node = Node(
        name + "_node",
        op_type="If",
        inputs=[prev_name],
        outputs=outputs,
        attrs=attrs,
    )
    nodes.append(if_node)

    graph = Graph(
        name,
        nodes,
        union_input_names,
        if_node.outputs,
        tensor_list,
    )
    ir_version = max(cache_model.ir_version, cacheless_model.ir_version, 8)
    model = Model(graph, ir_version)
    for opset in cache_model.opset_imports:
        model.add_opset(opset.domain, opset.version)
    for opset in cacheless_model.opset_imports:
        model.add_opset(opset.domain, opset.version)
    if share_initializers:
        model.share_initializers()
    return model


def _prepare(model1, model2, min_opset=16):
    model1 = model1.clone()
    model2 = model2.clone()
    opset = max(model1.default_opset, model2.default_opset, min_opset)
    model1.update_default_opset(opset)
    model2.update_default_opset(opset)
    return model1, model2


def _match_inputs(
    then_graph,
    else_graph,
    common_input_names,
    union_input_names,
    then_inputs,
    make_is_opt_node=False,
):
    for i in common_input_names:
        t1 = then_graph.name_to_tensor[i]
        t2 = else_graph.name_to_tensor[i]
        if t1.dtype != t2.dtype or t1.shape != t2.shape:
            raise ValueError(
                f"Input {i} has different shape or dtype between then and else branches"
            )
        elif t1.is_optional != t2.is_optional:
            g = else_graph if t1.is_optional else then_graph
            t = (t2 if t1.is_optional else t1).clone()
            opt_t = t.clone()
            opt_t.is_optional = True
            _make_input_optional(g, t, opt_t)

    name_to_input_tensor = {
        t.name: t
        for t in map(lambda t: t.clone(), then_graph.inputs + else_graph.inputs)
    }
    nodes = []
    tensor_list = []
    for t in common_input_names:
        tensor_list.append(name_to_input_tensor[t])
    for t1, t2 in zip(then_graph.outputs, else_graph.outputs):
        t = t1.clone()
        if t1.shape != t2.shape:
            if len(t1.shape) != len(t2.shape):
                t.clear_shape()
            else:
                shape = t1.shape.copy()
                for i in range(len(shape)):
                    if t1.shape[i] != t2.shape[i]:
                        shape[i] = f"{t1.shape[1]}_{t2.shape[1]}"
                t.shape = shape
        t.name = t.name[5:]
        tensor_list.append(t)

    for c in set(union_input_names) - set(common_input_names):
        if c in then_inputs:
            g = then_graph
        else:
            g = else_graph
        t = g.name_to_tensor[c].clone()
        opt_t = t.clone()
        opt_t.is_optional = True
        is_opt_t = Tensor(f"{t.name}_is_opt", DType.BOOL, [])
        is_opt_node = Node(
            f"{opt_t.name}_is_opt_node",
            op_type="OptionalHasElement",
            inputs=[opt_t.name],
            outputs=[is_opt_t.name],
        )
        if make_is_opt_node:
            nodes.append(is_opt_node)
            tensor_list.append(opt_t.clone())
            tensor_list.append(is_opt_t)
        else:
            tensor_list.append(t.clone())
        if not t.is_optional:
            _make_input_optional(g, t, opt_t)
    return nodes, tensor_list


def _make_input_optional(g, t, opt_t):
    g.remove_tensor(t.name)
    t.name = "_" + t.name
    g.add_tensor(t)
    for n in g.input_name_to_nodes()[opt_t.name]:
        n.inputs[n.inputs.index(opt_t.name)] = t.name
    get_opt_node = Node(
        f"{opt_t.name}_get_opt_node",
        op_type="OptionalGetElement",
        inputs=[opt_t.name],
        outputs=[t.name],
    )
    g.add_node(get_opt_node, left=True)
    g.add_tensor(opt_t.clone())
