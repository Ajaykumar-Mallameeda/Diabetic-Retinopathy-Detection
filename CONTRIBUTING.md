# Contributing to Diabetic Retinopathy Detection

## Getting Started

1. Fork the repository.
2. Create a branch: `git checkout -b feature/your-feature-name`.
3. Make your changes.
4. If adding a model, include training/evaluation notebooks.
5. Push and open a Pull Request.

## Code Style

- Follow PEP 8 for Python code.
- Keep notebooks clean — clear outputs before committing.
- Add docstrings to new functions and classes.

## Dataset Notes

- Do **not** commit dataset files to the repository.
- Use `src/utils/config.py` to configure dataset paths locally.
- Document any new dataset requirements in the README.

## Pull Request Checklist

- [ ] Code runs end-to-end without errors
- [ ] New models include training and evaluation code
- [ ] Results documented and caveated (small dataset, single experiment)
- [ ] Notebook outputs are cleared before commit
- [ ] No large files or datasets included in the commit

## Reporting Issues

Open an issue with:
- Steps to reproduce
- Error message (if applicable)
- What you expected to happen
