# Publishing checklist for Packt's GitHub repository

## Target

- Repository: <https://github.com/PacktPublishing/LLMs-in-Finance>
- Default branch: `main`
- Notebook-suite release: `1.1.0`
- Release tag: `notebook-suite-v1.1.0`

The notebook-suite tag is intentionally distinct from the book's immutable
canonical companion tag `B32413-v1.0.0`.

## Before copying files

From the release tree:

```bash
python -m pip install -r requirements-lock.txt
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
make release
```

`make release` lints the Python sources, regenerates all fixtures, rebuilds and
executes every notebook, runs the tests, exports the README previews, seals the
manifest, creates a deterministic ZIP, extracts it into a temporary directory,
and repeats the complete build. It must end with a byte-for-byte match.

## Copy into a clean clone

Work from a fresh clone so an unrelated local tree cannot contaminate the
release:

```bash
git clone https://github.com/PacktPublishing/LLMs-in-Finance.git
cd LLMs-in-Finance
git switch main
```

Copy the contents of the verified release tree into that clone while preserving
the clone's `.git/` directory. Then inspect:

```bash
git status --short
git diff --check
git diff --stat
```

Do not copy a ZIP into the repository. The deterministic ZIP and checksum belong
on the GitHub Release only.

## Recommended repository settings

After the first push:

1. Require the `notebooks` and `codeql` checks before merging to `main`.
2. Require at least one review and dismiss approvals after new commits.
3. Enable secret scanning, push protection, dependency alerts, and private
   vulnerability reporting.
4. Upload `assets/github_social_preview.png` under **Settings → General →
   Social preview**.
5. Add repository topics such as `finance`, `llm`, `rag`,
   `model-governance`, `jupyter-notebook`, and `education`.
6. Confirm every Colab link resolves after the files are public.

## Release

Create and push the independent tag only after the protected `main` checks pass:

```bash
git tag -a notebook-suite-v1.1.0 -m "Verified notebook suite v1.1.0"
git push origin notebook-suite-v1.1.0
```

The tag workflow verifies that the tag equals the package version, reproduces
the release again, and publishes the deterministic ZIP plus its SHA-256 file.

## Final manual checks

- Open the README on GitHub and inspect the banner, demo GIF, tables, and links.
- Open notebooks 00, 04, 06, 16, and 17 in both GitHub and Colab.
- Confirm the Actions badge resolves against `main`.
- Confirm the MIT license is detected by GitHub.
- Confirm no public issue or artifact contains credentials or proprietary data.
