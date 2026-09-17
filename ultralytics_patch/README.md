# Ultralytics patch (8.4.37)

The experiments in the paper use a small number of custom modules that are not
part of the stock `ultralytics` package. This folder contains the exact patched
files that were used on the training instance (`ultralytics==8.4.37`,
Python 3.12, PyTorch 2.8.0+cu128).

## What is added

| Module | Location | Used by |
| --- | --- | --- |
| `BiFPN_Concat2` | `nn/modules/conv.py` | `yolov8{m,s,l}-bifpn.yaml` (4-node weighted-bidirectional-fusion Neck) |
| `BiFPN_Concat3` | `nn/modules/conv.py` | available for 3-input fusion nodes |
| `CBAM`, `ECA` | `nn/modules/conv.py` | `attention_compare/yolov8m-cbam.yaml`, `yolov8m-eca.yaml` |
| `SPDConv` | `nn/modules/conv.py` | exploratory runs (not used in the final paper) |
| `parse_model` handling | `nn/tasks.py` | registers the modules above so YAML model definitions can instantiate them |

`C3TR` (lightweight Transformer comparison) is part of stock ultralytics and
needs no patch.

BiFPN fusion weights are learned as
`w_i = ReLU(w_i) / (sum(ReLU(w)) + eps)` and are saved inside the trained
`best.pt` checkpoints; the analysis in the paper reads them from
`model.model[...].conv` weight parameters.

## How to apply

```bash
pip install ultralytics==8.4.37
python ultralytics_patch/apply_patch.py
```

The script copies the three patched files over the installed package and
creates a `*.orig` backup of each original file. A different ultralytics
version triggers a warning because the files are full-file replacements taken
from 8.4.37.

To restore the stock package, copy every `*.py.orig` backup back over the
patched file (remove the `.orig` suffix), or simply
`pip install --force-reinstall ultralytics==8.4.37`.
