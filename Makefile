SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3
CONFIG ?= configs/experiment.yaml
DRY ?= 0
SMOKE ?= 0
DATA_ROOT ?= data

.PHONY: setup train evaluate reproduce lint tests clean data data-mini smoke paper report report-lite report-paper phase2 phase2_scorecard plot-ig-wg eval-crisis coverage

LAST_PAPER ?= $(shell ls -td reports/paper/* 2>/dev/null | head -1)

setup:
	$(PYTHON) -m pip install -r requirements-lock.txt

train:
	tools/run_train.sh $(CONFIG)

evaluate:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "CHECKPOINT path required, e.g. make evaluate CHECKPOINT=reports/artifacts/latest/checkpoints/epoch.pt"; \
		exit 1; \
	fi
	tools/run_eval.sh $(CONFIG) eval.report.checkpoint_path=$(CHECKPOINT)

reproduce:
	tools/make_reproduce.sh

lint:
	$(PYTHON) -m ruff check src

tests:
	$(PYTHON) -m pytest

smoke-check:
	$(PYTHON) tools/scripts/check_smoke_determinism.py

data-check:
	$(PYTHON) tools/scripts/check_data_integrity.py --data-root $(DATA_ROOT)

clean:
	rm -rf runs outputs outputs_* htmlcov .pytest_cache .coverage .coverage.* coverage.xml reports/coverage data/raw data/external
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete

data:
	DATA_DIR=$(DATA_ROOT) tools/fetch_data.sh
	DATA_DIR=$(DATA_ROOT) $(PYTHON) tools/make_data_snapshot.py --mode both

data-mini:
	DATA_DIR=$(DATA_ROOT) tools/fetch_data.sh
	DATA_DIR=$(DATA_ROOT) $(PYTHON) tools/make_data_snapshot.py --mode mini

smoke:
	@set -euo pipefail; \
	python3 experiments/run_train.py --config-name=train/smoke; \
	LAST_RUN=$$(ls -td reports/artifacts/*/ 2>/dev/null | head -1); \
	if [ -z "$$LAST_RUN" ]; then echo "No run directories found" >&2; exit 1; fi; \
	CHECKPOINT=$$(python3 tools/scripts/find_latest_checkpoint.py "$$LAST_RUN"); \
	python3 experiments/run_diagnostics.py --config-name=eval/smoke eval.report.checkpoint_path=$$CHECKPOINT

paper:
	@set -euo pipefail; \
	CMD="tools/run_of_record.sh"; \
	if [ "$(SMOKE)" = "1" ]; then CMD="$$CMD --smoke"; fi; \
	if [ "$(DRY)" = "1" ]; then CMD="$$CMD --dry-run"; fi; \
	echo "$$CMD"; \
	$$CMD

phase2:
	@echo "See src/legacy/experiments_notes/phase2_plan.md for details."

report:
	PYTHONPATH=. $(PYTHON) tools/scripts/aggregate.py --config configs/report/default.yaml

report-lite:
	PYTHONPATH=. $(PYTHON) tools/scripts/aggregate.py --config configs/report/default.yaml --lite

report-paper:
	$(PYTHON) tools/report/generate_report.py --config configs/report/paper.yaml --smoke $(ARGS)

phase2_scorecard:
	@echo "[DEPRECATED] 'make phase2_scorecard' now forwards to 'make report'." >&2
	$(PYTHON) tools/scripts/aggregate.py --config configs/report/default.yaml

plot-ig-wg:
	@set -euo pipefail; \
	RUN_DIR="$(LAST_PAPER)"; \
	if [ -z "$$RUN_DIR" ]; then \
		echo "No paper report found. Run 'make paper' followed by 'make report-paper'." >&2; \
		exit 1; \
	fi; \
	$(PYTHON) -m src.visualization.plot_invariance_vs_ig --run_dir "$$RUN_DIR" --out_dir reports/figures/ig_vs_wg --format png

eval-crisis:
	@set -euo pipefail; \
	$(PYTHON) experiments/run_diagnostics.py --config-name=eval/robustness

coverage:
	$(PYTHON) -m pytest --maxfail=1 --disable-warnings --cov=src --cov-report=term-missing --cov-report=html:reports/coverage
