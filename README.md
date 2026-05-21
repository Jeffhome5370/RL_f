# DQN Tool Selection Agent

A lightweight RL-based AutoML-style project that treats machine learning
pipeline construction as a sequential decision-making problem.

The MVP implements the full loop:

- Dataset loading
- Dataset profiling
- State construction
- Action-based pipeline building
- Reward calculation
- DQN training
- Baseline comparison
- Result logging and plotting

The agent learns to select preprocessing tools, feature tools, classifiers, and
an evaluation action based on dataset meta-features and the current pipeline
state.

## Environment Setup

This project is intended to run with Conda. GPU PyTorch is installed through
Conda channels, so `torch` is intentionally not listed in `requirements.txt`.

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate dqn-tool-agent
```

Verify that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Expected output should include:

```text
True
12.1
```

If `torch.cuda.is_available()` returns `False`, check that the NVIDIA driver is
installed correctly and that the installed CUDA runtime version is compatible
with your GPU driver.

## Project Structure

```text
.
|-- main.py
|-- train_dqn.py
|-- run_baselines.py
|-- config.py
|-- requirements.txt
|-- environment.yml
|
|-- data/
|   |-- dataset_manager.py
|   `-- data_profiler.py
|
|-- env/
|   `-- tool_selection_env.py
|
|-- tools/
|   |-- tool_executor.py
|   `-- pipeline_cache.py
|
|-- agents/
|   |-- q_network.py
|   |-- replay_buffer.py
|   `-- dqn_agent.py
|
|-- baselines/
|   |-- random_agent.py
|   |-- fixed_pipeline.py
|   `-- grid_search_baseline.py
|
|-- utils/
|   |-- seed.py
|   |-- metrics.py
|   `-- plot.py
|
`-- results/
    |-- logs/
    |-- figures/
    `-- tables/
```

## Run Order

Run the project from the repository root.

### 1. Test the Environment

```bash
python main.py
```

This checks that the environment can reset, sample actions, build a pipeline,
and return rewards.

### 2. Run Baselines

```bash
python run_baselines.py
```

This generates:

```text
results/tables/random_agent_results.csv
results/tables/fixed_pipeline_results.csv
results/tables/grid_search_results.csv
```

### 3. Train DQN

```bash
python train_dqn.py
```

This generates:

```text
results/logs/dqn_training_results.csv
results/logs/dqn_agent.pth
results/figures/dqn_reward_curve.png
results/figures/dqn_f1_curve.png
results/figures/dqn_invalid_curve.png
results/figures/dqn_pipeline_length_curve.png
```

## Datasets

The MVP uses built-in scikit-learn classification datasets:

- Iris
- Wine
- Breast cancer
- Digits

Each dataset is split into train, validation, and test sets. The current
environment evaluates pipelines on the validation split.

## Action Space

The current action space is defined in `config.py`:

```text
standard_scaler
minmax_scaler
pca
feature_selection
random_forest
svm
knn
evaluate
```

Invalid actions are allowed but penalized, so the DQN learns action constraints
from rewards rather than using an action mask.

## Current Limitations

- Datasets are limited to scikit-learn built-in datasets.
- Feature selection currently uses `k="all"` as a placeholder.
- Reward does not yet include a detailed computational cost term.
- DQN does not use action masking.
- Grid search uses a small candidate space for MVP comparison.

## Version Control Notes

Commit source code, configuration files, and `.gitkeep` files under `results/`.
Avoid committing local runtime caches, trained model files, and generated result
tables unless the experiment output is intentionally part of the report.
