# AI Market Intelligence Workflow

A Python-based AI/ML workflow that automates market and macroeconomic data collection, feature engineering, walk-forward backtesting, and research-ready decision-support outputs.

## Overview
This project is designed as a research prioritization and signal-screening tool rather than a stand-alone investment engine. It focuses on:
- automated multi-source data ingestion
- interpretable feature engineering
- walk-forward model evaluation
- structured outputs for analyst review

## Features
- Downloads market price data with yfinance
- Pulls macroeconomic indicators from FRED
- Engineers momentum, volatility, drawdown, and relative-performance features
- Uses walk-forward backtesting for time-series evaluation
- Generates:
  - predictions.csv
  - feature_importance.csv
  - summary.json
  - research_brief.md

## Tech Stack
- Python
- pandas
- yfinance
- scikit-learn

## How to Run
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python ai_market_intel.py

## Notes
This project is intended as a prototype for AI-enabled research automation and structured decision support.
