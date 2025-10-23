SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3
CONFIG ?= configs/experiment.yaml
DRY ?= 0
SMOKE ?= 0
DATA_ROOT ?= data
.PHONY: setup train evaluate reproduce lint tests smoke phase2 report report-lite phase2_scorecard paper
.PHONY: setup train evaluate reproduce lint tests smoke phase2 report report-lite report-paper phase2_scorecard
.PHONY: ci-smoke ci-unit ci-train-lite ci-report-lite
setup:
	$(PYTHON) -m pip install -r requirements.txt
train:
	tools/run_train.sh $(CONFIG)
evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "CHECKPOINT path required, e.g. make evaluate CHECKPOINT=outputs/checkpoints/checkpoint_150000.pt"; \
		exit 1; \
	fi
	tools/run_eval.sh $(CONFIG) eval.report.checkpoint_path=$(CHECKPOINT)
reproduce:
	tools/make_reproduce.sh
lint:
	$(PYTHON) -m ruff check src
tests:
	$(PYTHON) -m pytest
.PHONY: data data-mini

ci-unit:
	python -m pip install -e .[dev]
	pytest -q tests/smoke tests/unit


ci-train-lite:
	python experiments/run_train.py \
		training.max_steps=150 \
		data.loader=smoke \
		logging.wandb.enabled=false \
		outputs.dir="runs/ci_smoke"
ci-report-lite:
	python -m legacy.report_core.lite --runs "runs/ci_smoke" --no_figures

ci-smoke: ci-unit ci-train-lite ci-report-lite
data:
	DATA_DIR=$(DATA_ROOT) tools/fetch_data.sh
	DATA_DIR=$(DATA_ROOT) $(PYTHON) tools/make_data_snapshot.py --mode both

data-mini:
	DATA_DIR=$(DATA_ROOT) tools/fetch_data.sh
	DATA_DIR=$(DATA_ROOT) $(PYTHON) tools/make_data_snapshot.py --mode mini

smoke:
	@set -euo pipefail; \
	python3 experiments/run_train.py --config-name=train/smoke; \
	LAST_RUN=$$(ls -td runs/*/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_RUN" ]; then echo "No run directories found" >&2; exit 1; fi; \
	CHECKPOINT=$$(python3 tools/scripts/find_latest_checkpoint.py "$$LAST_RUN"); \
	python3 experiments/run_diagnostics.py --config-name=eval/smoke eval.report.checkpoint_path=$$CHECKPOINT
paper:
	tools/run_of_record.sh
phase2:
	@echo "See src/legacy/experiments_notes/phase2_plan.md for details."
.PHONY: report
report:
	PYTHONPATH=. $(PYTHON) tools/scripts/aggregate.py --config configs/report/default.yaml

report-lite:
	PYTHONPATH=. $(PYTHON) tools/scripts/aggregate.py --config configs/report/default.yaml --lite

report-paper:
	PYTHONPATH=. $(PYTHON) tools/scripts/aggregate.py --config configs/report/paper.yaml

.PHONY: report-paper
report-paper:
	$(PYTHON) tools/report/generate_report.py --config configs/report/default.yaml $(ARGS)

.PHONY: phase2_scorecard
phase2_scorecard:
	@echo "[DEPRECATED] 'make phase2_scorecard' now forwards to 'make report'." >&2
	$(PYTHON) tools/scripts/aggregate.py --config configs/report/default.yaml

paper:
	@set -euo pipefail; \
	CMD="tools/run_of_record.sh"; \
	if [ "$(SMOKE)" = "1" ]; then CMD="$$CMD --smoke"; fi; \
	if [ "$(DRY)" = "1" ]; then CMD="$$CMD --dry-run"; fi; \
	echo "$$CMD"; \
	$$CMD
