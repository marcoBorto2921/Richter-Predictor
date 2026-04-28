CONFIG = configs/config.yaml
PYTHON = .venv/Scripts/python  # Windows; change to .venv/bin/python on Linux/macOS

.PHONY: all eda train-lgb train-xgb train-cat train-all ensemble lint clean

all: train-all ensemble

eda:
	$(PYTHON) notebooks/01_eda.py --config $(CONFIG)

train-lgb:
	$(PYTHON) src/train.py --config $(CONFIG) --model lgb

train-xgb:
	$(PYTHON) src/train.py --config $(CONFIG) --model xgb

train-cat:
	$(PYTHON) src/train.py --config $(CONFIG) --model cat

train-all: train-lgb train-cat train-xgb

ensemble:
	$(PYTHON) src/ensemble.py --config $(CONFIG)

smoke:
	$(PYTHON) src/train.py --config $(CONFIG) --model lgb --no-optuna

lint:
	.venv/Scripts/ruff check . --fix && .venv/Scripts/ruff format .
	.venv/Scripts/vulture .

clean:
	rm -rf models/*.pkl models/*.npy
