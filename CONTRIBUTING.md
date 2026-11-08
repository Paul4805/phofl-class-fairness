# Contributing

This repository is a measurement study, not a method-development codebase. The
most useful contributions are reproducibility reports, analysis corrections,
and bug fixes. Please do not reframe the work as a new federated-learning
algorithm.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

The scripts import the pinned PHP-FL repository from
`third_party/PHP-FL-main`:

```bash
git clone https://github.com/Siyuan01/PHP-FL-main.git third_party/PHP-FL-main
git -C third_party/PHP-FL-main checkout 283c4a4c561b153d17ee0278ac35e5580b3fe28f
```

Generated CIFAR-10 data and model checkpoints are intentionally excluded. See
`DATA.md` for regeneration instructions.

## Checks before opening a pull request

```bash
python -m compileall -q .
pytest -q
```

Please keep pull requests focused. Include:

- what changed and why;
- which artifacts or tables are affected;
- whether any numerical result changes;
- the exact command used to reproduce the change, if training was involved.

## Data and artifact policy

Do not commit:

- model checkpoints (`*.pt`, `*.pth`);
- `phofl_half/data/`, `phofl_half/data_cap/`, or `phofl_half/run_store/`;
- `third_party/` clones;
- virtual environments;
- raw training logs (`*.log`).

Result CSVs, analysis tables, figures, and per-run structured logs are welcome
if they are small and support a claim in the paper.

## Scope of claims

- The participation hypothesis is closed and must not be revived.
- `class_FM` is a reporting metric, not an objective.
- The diagnostic ladder is a negative result; do not describe it as a new
  method.
- The half-scale results are one-seed and are not comparable to the paper’s
  full-scale numbers.

## Reporting issues

Use the GitHub issue templates. For security-sensitive reports, follow
`SECURITY.md` instead of opening a public issue.
