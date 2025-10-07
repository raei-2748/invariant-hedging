SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
PYTHON ?= python3
CONFIG ?= configs/experiment.yaml
.PHONY: setup train evaluate reproduce lint tests smoke smoke-train smoke-eval phase2 phase2_scorecard
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
smoke-train:
	$(PYTHON) -m src.train method=erm_reg seed=0 +phase=smoke
smoke-eval:
	$(PYTHON) -m src.eval +phase=smoke
smoke: smoke-train smoke-eval
phase2:
	bash scripts/reproduce_phase2.sh
.PHONY: phase2_scorecard
phase2_scorecard:
	python scripts/make_scorecard.py --methods ERM,ERM_reg,IRM,HIRM_Head,HIRM_Head_HighLite,GroupDRO,V_REx --seeds 0..29 --split crisis --outdir runs/scorecard_export --read_only true --phase phase2 --commit_hash $$(git rev-parse --short HEAD)
	python scripts/compute_diagnostics.py --methods ERM,ERM_reg,IRM,HIRM_Head,HIRM_Head_HighLite,GroupDRO,V_REx --seeds 0..29 --train_envs low,medium --val_envs high --test_envs crisis --out runs/scorecard_export/diagnostics_all.csv --phase phase2 --commit_hash $$(git rev-parse --short HEAD)
	python scripts/plot_cvar_violin.py --diagnostics runs/scorecard_export/diagnostics_all.csv --out runs/scorecard_export/figs/fig_cvar_violin.png
	python scripts/plot_ig_vs_cvar.py --diagnostics runs/scorecard_export/diagnostics_all.csv --out runs/scorecard_export/figs/fig_ig_vs_cvar.png
	python scripts/plot_capital_frontier.py --scorecard runs/scorecard_export/scorecard.csv --out runs/scorecard_export/figs/fig_capital_frontier.png --notional 1.0
	python scripts/export_tables.py --scorecard runs/scorecard_export/scorecard.csv --out_md runs/scorecard_export/table_crisis.md --out_tex runs/scorecard_export/table_crisis.tex
