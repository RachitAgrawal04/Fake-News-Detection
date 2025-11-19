# Fake News Detection (Notebook-Only)

This repo intentionally keeps just the pieces needed to re-run the original notebook experiment. There are no extra scripts, pipelines, or presentations anymore.

## What is here?

- `FakeNewsDetectionModel.ipynb`: the full exploratory workflow.
- `train_data.tsv`, `Validation_data.tsv`, `test_data.tsv`: the raw datasets the notebook expects.
- `pyproject.toml`: dependency list so you can install the same packages.

That’s all.

## Quickstart

```bash
pip install pipx
pipx runpip uv pip install --editable .
pipx run uv pip install -e .
```

Then open the notebook (VS Code + Jupyter or `jupyter lab`). If you prefer using `pip` directly:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

## Running the notebook

1. Launch Jupyter (VS Code notebook UI or `jupyter lab`).
2. Select the `.venv` interpreter.
3. Execute cells top to bottom. The datasets live in the repo root so paths resolve as written.

## Notes

- No automatic training scripts or saved models remain.
- Keep the TSV files in place; the notebook reads them by relative path.
- Add your own experiments in new notebooks if you want, but the baseline stays minimal.
