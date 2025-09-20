PYTHON ?= python3
CONFIG ?= configs/experiment.yaml

.PHONY: setup train evaluate reproduce lint

setup:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) -m hirm_experiment.cli.train --config-path configs --config-name experiment

evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "CHECKPOINT path required, e.g. make evaluate CHECKPOINT=outputs/checkpoints/checkpoint_150000.pt"; \
		exit 1; \
	fi
	$(PYTHON) -m hirm_experiment.cli.evaluate --config-path configs --config-name experiment_eval evaluation.checkpoint_path=$(CHECKPOINT)

reproduce:
	SEED=0 $(PYTHON) -m hirm_experiment.cli.evaluate --config-path configs --config-name reproduce

lint:
	$(PYTHON) -m ruff check src
