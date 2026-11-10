.PHONY: install data models backtest report docs test dashboard all clean-data

PY ?= .venv/bin/python
QL ?= .venv/bin/quantlab

install:            ## create the virtualenv and install the package
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:               ## download (pinned, checksum-verified) + reconstruct 6 seasons + unified database (~3 min)
	$(QL) fetch && $(QL) build && $(QL) database

models:             ## walk-forward features, team model, minutes, rates, opponent factors, simulator constants
	$(QL) features && $(QL) team && $(QL) minutes && $(QL) rates && $(QL) opponent && $(QL) constants

backtest:           ## walk-forward simulation backtests: 2023-24 (tuning) -> blend weight -> blend; 2024-26 (test) (~1.5 h, 4 cores)
	QUANTLAB_THREADS=1 $(QL) backtest --season 2023-24 --modes none,team
	$(QL) blendweight
	QUANTLAB_THREADS=1 $(QL) backtest --season 2023-24 --modes blend
	QUANTLAB_THREADS=1 $(QL) backtest --season 2024-25,2025-26 --modes none,team,blend

report:             ## baselines, gradient boosting, matchup ablation, team-first reconciliation, summary + docs/FINDINGS_DATA.md
	QUANTLAB_THREADS=1 $(QL) baselines && QUANTLAB_THREADS=1 $(QL) ablation && $(QL) topdown && $(QL) report

docs:               ## re-render the README results block, docs/REPORT.md, docs/headline.json and docs/report.html from the summary
	$(PY) scripts/docs/render_report.py --check && $(PY) scripts/docs/render_docs.py --check
	$(PY) scripts/docs/render_report.py && $(PY) scripts/docs/render_docs.py && $(PY) scripts/make_report_page.py

test:
	$(PY) -m pytest

dashboard:
	$(QL) dashboard

all: data models backtest report

clean-data:
	rm -rf data/raw data/staged data/processed data/features data/models data/backtest
