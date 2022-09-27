# onnxmodel-utils
utils for working with onnx models


## Example

Simple if model
```python
from onnxmodel_utils import Model, build_if_model


model1 = Model.load('model1.onnx')
model2 = Model.load('model2.onnx')

model = build_if_model(
    "if_model",
    "cond",
    model1,
    model2,
)
model.save('if_model.onnx')


import onnxruntime

sess = onnxruntime.InferenceSession('if_model.onnx')
inps = {
    "input": np.random.randn(1, 3, 224, 224).astype(np.float32),
    "cond": np.array([True]).astype(np.bool),
}
out1 = sess.run(None, inps)

inps["cond"] = np.array([False]).astype(np.bool)
out2 = sess.run(None, inps)
```

Optional cache model

```python
from onnxmodel_utils import Model, build_if_model_with_cache


decoder = Model.load("decoder.onnx")
decoder_init = Model.load("decoder_init.onnx")

model = build_if_model_with_cache(
    name="merged_model",
    cache_model=decoder,
    cacheless_model=decoder_init,
    cache_names=["pasts", "pasts_st"],
)
model.save("merged_model.onnx")


import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession("merged_model.onnx")
inps = {
    "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64),
    "target_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.int64),
    "pasts": None,
    "pasts_st": None,
}

init_out = sess.run(None, inps)
inps["pasts"] = init_out[1]
inps["pasts_st"] = init_out[2]

out = sess.run(None, inps)
```

## Installation

```bash
pip install onnxmodel-utils
```
