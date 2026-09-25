# Publishing checklist

## Repository

- Suggested name: `gutsignal-lab`
- Suggested description: Independent signal-to-insight prototype: a
  transparent bowel-sound event-detection baseline with participant-aware
  validation.
- Suggested topics: `machine-learning`, `signal-processing`, `digital-health`,
  `streamlit`, `responsible-ai`
- Default branch: `main`
- Visibility: public for the application portfolio, unless the application
  process specifically requests a private link.

Internal application materials and the walkthrough script are intentionally
excluded through `.gitignore`. Raw questionnaire responses, interview
recordings, contact details, API keys and Streamlit secrets must never be
committed.

## Pre-publish checks

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest -q
git status --short --ignored
```

Confirm that:

- the bundled model and three synthetic, non-human demo WAV files are present;
- no row-level participant or recording identifiers are tracked;
- `docs/DATA_AND_ATTRIBUTION.md` is visible;
- `docs/APPLICATION_DRAFT.md` is ignored;
- no `.env` or `.streamlit/secrets.toml` file is staged;
- the app starts with `uv run streamlit run streamlit_app.py`.

## Streamlit deployment

1. Create the GitHub repository and push the `main` branch.
2. In Streamlit Community Cloud, choose that repository.
3. Set the entrypoint to `streamlit_app.py`.
4. Select Python 3.12 if the deployment interface asks for a version.
5. No secrets are required.
6. After deployment, test all three curated clips, the four tabs, document
   downloads and the mobile layout.

## Submission links

The final application should use three links, in this order:

1. Live interactive demo
2. Two-minute walkthrough video
3. GitHub repository

The ZIP checkpoint is a backup, not something to send to Suna.
