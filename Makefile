PY ?= python3
export PYTHONPATH := .

.PHONY: data eda train compare predict eval shap badcase report ext gpu zoo all verify test

data:
	$(PY) -m src.data.make_dataset --config configs/base.yaml

eda:
	$(PY) scripts/eda.py --config configs/base.yaml

train:
	$(PY) -m src.models.train --config configs/base.yaml --model lgbm

compare:
	$(PY) -m src.models.train --config configs/base.yaml --compare --seed 42

predict:
	$(PY) -m src.models.predict --config configs/base.yaml --model lgbm

eval:
	$(PY) -m src.eval.evaluate --config configs/base.yaml

shap:
	$(PY) -m src.eval.interpret --config configs/base.yaml

badcase:
	$(PY) -m src.eval.badcase --config configs/base.yaml

report:
	$(PY) -m src.eval.report --config configs/base.yaml

ext:
	$(PY) scripts/run_extensions.py --experiment all

gpu:
	$(PY) scripts/gpu_benchmark.py --scaling

zoo:
	$(PY) scripts/model_zoo_benchmark.py

verify:
	$(PY) scripts/verify.py --config configs/base.yaml

test:
	$(PY) -m pytest -q

all: data eda compare predict eval shap badcase report ext verify
	@echo "== Bike Sharing Demand pipeline complete =="
