# A Single-Loop Bilevel Deep Learning Method for Optimal Control of Obstacle Problems

Numerical experiments for the paper by Yongcun Song, Shangzhi Zeng, Jin Zhang, and Lvgang Zhang, published in *Neural Networks*, 206 (2027), 109761. [Paper](https://doi.org/10.1016/j.neunet.2026.109761)

## Getting started

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Run the following commands from the repository root.

```sh
uv sync --locked --group notebook --python 3.10.19
uv run --no-sync jupyter lab
```

Select the **Python 3 (ipykernel)** kernel and run the cells in order. Run Example 3 from the repository root with `uv run --no-sync python example3_high_dimensional_space.py`.

Training uses CUDA when available and otherwise runs on the CPU. The locked environment was tested on macOS (Apple silicon, CPU). The default training budgets can take substantial time.

## Experiments

| Example | Entry point |
| --- | --- |
| 1. Bi-active set | [Notebook](example1_bi_active_set.ipynb) |
| 1. Control constraints | [Notebook](example1_control_constraints.ipynb) |
| 1. Weighted PINN baseline | [Notebook](example1_weighted_pinn.ipynb) |
| 2. Flat free boundary | [Notebook](example2_flat_free_boundary.ipynb) |
| 3. Five-dimensional obstacle problem | [Python script](example3_high_dimensional_space.py) |
| 4. Complex domain | [Notebook](example4_complex_domain.ipynb) |
| 5. Obstacle control | [Notebook](example5_obstacle_control.ipynb) |
| 6. Non-symmetric EVI | [Notebook](example6_non_symmetric_evi.ipynb) |

Run outputs, including logs and checkpoints, are written under `outputs/`.

## Data and figures

- [Example 2 data](data/example2/data_nu_0.02_original.mat) contain the reference solution for the paper setting $\nu=0.02$.
- [Example 5 data](data/example5/example3_data_modified.mat) contain the reference solution. The notebook initializes the state network from the bundled [pretrained weights](data/example5/pretrain_y_energy_final.pth). These weights were checked with the code. Their identity with the weights in the original paper run has not been verified.
- [draw.ipynb](draw.ipynb) generates the training and forward-solve curves for Examples 1, 2, 4, 5, and 6 from the included CSV files in `data/paper_curves/`. Run all cells to save the figures under `outputs/figures/` without retraining.

The historical inputs for the weighted PINN comparison plots and Example 1 solution panels are unavailable in this release. The training notebooks can produce new results and solution plots.
