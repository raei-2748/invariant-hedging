SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3
CONFIG ?= configs/experiment.yaml
.PHONY: setup train evaluate reproduce lint tests smoke phase2 report report-lite report-paper phase2_scorecard
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
smoke:
	@set -euo pipefail; \
	python3 -m src.train --config-name=train/smoke; \
	LAST_RUN=$$(ls -td runs/*/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_RUN" ]; then echo "No run directories found" >&2; exit 1; fi; \
	CHECKPOINT=$$(python3 scripts/find_latest_checkpoint.py "$$LAST_RUN"); \
	python3 -m src.eval --config-name=eval/smoke eval.report.checkpoint_path=$$CHECKPOINT
phase2:
	@echo "See experiments/phase2_plan.md for details."
.PHONY: report
report:
	$(PYTHON) scripts/aggregate.py --config configs/report/default.yaml

report-lite:
	$(PYTHON) scripts/aggregate.py --config configs/report/default.yaml --lite

.PHONY: report-paper
report-paper:
	$(PYTHON) scripts/report/generate_report.py --config configs/report/default.yaml $(ARGS)

.PHONY: phase2_scorecard
phase2_scorecard:
	@echo "[DEPRECATED] 'make phase2_scorecard' now forwards to 'make report'." >&2
	$(PYTHON) scripts/aggregate.py --config configs/report/default.yaml
