VERSION := $(shell python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')
ARCHIVE := dist/LLMs-in-Finance-notebook-suite-v$(VERSION).zip

.PHONY: setup setup-release setup-extensions data build lint test execute previews manifest verify reproduce package verify-archive release clean

setup:
	python -m pip install -e ".[notebooks,dev]"

setup-release:
	python -m pip install -r requirements-lock.txt
	python -m pip install -r requirements-dev.txt
	python -m pip install --no-deps -e .
	python -m pip check

setup-extensions:
	python -m pip install -e ".[notebooks,extensions]"

data:
	python scripts/generate_data.py

build:
	python scripts/build_notebooks.py

lint:
	python -m compileall -q src scripts tests notebook_sources
	ruff check src scripts tests

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

execute:
	python scripts/execute_notebooks.py --in-place

previews:
	python scripts/export_readme_previews.py

manifest:
	python scripts/write_release_manifest.py

verify:
	python scripts/verify_release.py

reproduce:
	python scripts/generate_data.py
	python scripts/build_notebooks.py
	PYTHONPATH=src python -m unittest discover -s tests -v
	python scripts/execute_notebooks.py --in-place
	python scripts/export_readme_previews.py
	python scripts/write_release_manifest.py
	python scripts/verify_release.py

package:
	python scripts/package_release.py

verify-archive:
	python scripts/verify_archive.py $(ARCHIVE)

release: lint reproduce package verify-archive

clean:
	python scripts/clean_notebook_outputs.py
