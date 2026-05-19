# DQN Tool Selection Agent

A lightweight, interpretable, and controllable RL-based AutoML-style project
for treating machine learning pipeline construction as a sequential
decision-making problem.

The agent will eventually learn to select tools such as scalers, dimensionality
reduction, feature selection, classifiers, and evaluation actions based on
dataset meta-features and the current pipeline state.

## Current Status

This repository currently contains only the initial project skeleton. The DQN
algorithm, environment transitions, sklearn pipeline execution, evaluator, and
baselines are intentionally left as placeholders.

## Project Structure

```text
dqn_tool_selection_agent/
├── configs/
├── src/
│   ├── data/
│   ├── env/
│   ├── tools/
│   ├── evaluator/
│   ├── agent/
│   ├── baselines/
│   └── utils/
├── scripts/
├── outputs/
├── tests/
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## Basic Verification

From the project root:

```bash
python main.py
python tests/test_imports.py
```

If pytest is available, the import smoke test can also be run with:

```bash
python -m pytest tests/test_imports.py
```

## Notes

- No DQN training logic is implemented yet.
- No sklearn pipeline execution logic is implemented yet.
- No dataset loading, profiling, reward, or evaluator logic is implemented yet.
