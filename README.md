# KD

Symbolic regression platform for partial differential equation discovery.

## Quick start

```python
import kd2

dataset = kd2.generate_burgers_data(seed=0)
m = kd2.Model().fit(dataset)
print(m.best_expr_)
```

See [`examples/01_quickstart.py`](examples/01_quickstart.py) for an
end-to-end walkthrough and [`examples/README.md`](examples/README.md)
for the full example index.

## Install

```bash
git clone https://github.com/Mao-Hao/KD.git
cd KD
uv sync
```

Python ≥ 3.11, PyTorch ≥ 2.0.

## License

[Apache-2.0](LICENSE).
