.PHONY: install data models backtest report test dashboard all clean-data

PY ?= .venv/bin/python
QL ?= .venv/bin/quantlab

install:            ## create the virtualenv and install the package
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

data:               ## download (pinned, checksum-verified) + reconstruct 6 seasons + unified database (~3 min)
	$(QL) fetch && $(QL) build && $(QL) database

models:             ## walk-forward features, team model, minutes, rates, opponent factors, simulator constants
	$(QL) features && $(QL) team && $(QL) minutes && $(QL) rates && $(QL) opponent && $(QL) constants

backtest:           ## walk-forward simulation backtests for 2023-24 (tuning) and 2024-26 (test) (~1 h, 4 cores)
	$(QL) backtest --modes none,team

report:             ## baselines, gradient boosting, team-first reconciliation, summary + docs/FINDINGS_DATA.md
	QUANTLAB_THREADS=1 $(QL) baselines && $(QL) topdown && $(QL) report

test:
	$(PY) -m pytest

dashboard:
	$(QL) dashboard

all: data models backtest report

clean-data:
	rm -rf data/raw data/staged data/processed data/features data/models data/backtest
