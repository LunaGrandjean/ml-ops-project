# Working as a team of five

1. Each task has an issue, an owner and a definition of done.
2. Start from an up-to-date `main`; create a `feat/...`, `fix/...` or `docs/...` branch.
3. Make short, descriptive commits. No direct push to `main`.
4. Open a pull request and ask another member to review it.
5. Merge once CI is green. Resolve conflicts in the working branch.
6. Update the documentation and keep the run IDs of the selected experiments.

Protect `main` on GitHub: one approval and the CI checks required.
Keep TEAM.md up to date with the five members and their responsibilities.

All repository text (code, comments, docs, commit messages) is written in English.

Never commit secrets, Inside data, model weights or caches. DVC metadata
(`.dvc`, `dvc.lock`) and `uv.lock` are versioned. Notebooks are only for exploration;
move reusable code to `src/` or `scripts/`.

When changing dependencies: run `uv lock`, check the tests, and commit
`pyproject.toml` and `uv.lock` together.
