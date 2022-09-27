from tqdm import tqdm


class TransformBase:
    op_types: "List[str]"

    def __init__(self):
        self.context = {}

    def prepare(self, model):
        pass

    def clean(self):
        del self.context


class TransformCompose:
    def __init__(self, modules) -> None:
        self.modules = modules

    def quantize(self, model: "Model") -> None:
        model.update()
        for module in tqdm(self.modules):
            for op_type in module.op_types:
                nodes_to_remove = []
                for node in tqdm(model.nodes_by_op_type(op_type), desc=op_type):
                    module.forward(
                        node,
                        model,
                        nodes_to_remove,
                    )
                for node in nodes_to_remove:
                    model.remove_node(node)
                model.update()
                model.prune()

    def __call__(self, model: "Model") -> None:
        self.quantize(model)
        model.update()
        model.prune()
