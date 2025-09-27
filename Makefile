PYTHON ?= python3
CONFIG ?= configs/experiment.yaml

.PHONY: setup train evaluate reproduce lint tests

setup:
	$(PYTHON) -m pip install -r requirements.txt

train:
	scripts/run_train.sh $(CONFIG)

evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "CHECKPOINT path required, e.g. make evaluate CHECKPOINT=outputs/checkpoints/checkpoint_150000.pt"; \
		exit 1; \
	fi
	scripts/run_eval.sh $(CONFIG) eval.report.checkpoint_path=$(CHECKPOINT)

reproduce:
	scripts/make_reproduce.sh

lint:
	$(PYTHON) -m ruff check src

tests:
	$(PYTHON) -m pytest
