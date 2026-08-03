# v1.0.1 rebuild provenance

This release was rebuilt in an isolated directory from two trusted inputs.
No file from the reported concurrently rewritten working tree was copied,
merged, or used as a reference.

## Trusted baseline

- Archive: `Large_Language_Models_in_Finance_Notebook_Suite_v1.0.0.zip`
- SHA-256:
  `f7da74eed510ad959898e402172cb5b5b39911d07c420143b512c91b42aa3522`
- Role: fresh baseline for the complete repository

## Verified replacement bundle

- Archive: `files - 2026-07-28T102259.194.zip`
- SHA-256:
  `5ddf57d02a67b1690bac33f7449c30b3fdaac70a85d74d7f5524754663289f96`
- Shape: ten flat regular files; no links, paths, or nested archives
- Role: replacement modules under `src/finllm_lab/`

Incoming file hashes:

| File | SHA-256 |
|---|---|
| `__init__.py` | `0e28686e4a7a110df5f4755ed379af0eee06b5ec2a9fed2159133ce197721cd1` |
| `contracts.py` | `31b658005480b949d1de45a5139dcb1b9ac3a8a78b2ab2eefeb79304136b9e0b` |
| `core.py` | `4fdc3d4b898defc85a5fc011a1c83fbd96495c9af736cabe0e80f9f3abd6918e` |
| `governance.py` | `978c24c36c2f42d1bbac53dbbe68ab5abc580f9156e524b8b90f26941b46ff1f` |
| `market.py` | `f374d775851d17ddd1f3c7dfc3310b7348d6fccb1fe39bb59b7e64f48cb8a5a7` |
| `metrics.py` | `0f0dbe55ad464e44d0a67aa0c68f5b3d9efdcca3fa7d8a7c7b28a82c052f8e62` |
| `rag.py` | `32151d9d3b89438c2e61faf1e2175254258077b4b36d1bd94a534df093b3e6c6` |
| `rl.py` | `0420b10b543acce2544a112e0db930748431c7a0526ee8568e8a4fe090ce4936` |
| `temporal.py` | `d59645b0f137e2f8b4a0ae9ec6a28d37a9f5c76164e0ff16874dc7f60756f4e5` |
| `text.py` | `4ce3cd2ee8d8e6b6cbad678947851cd9edabbf3c5827decae681dd1cfcd29ae8` |

The ten files were compared byte-for-byte with the extracted replacement
bundle immediately after staging. Subsequent v1.0.1 changes are ordinary
reviewed repository changes and are hashed in `RELEASE_MANIFEST.json`.

## Excluded material

The rebuild did not use the suspect model card or any other content described
as appearing only in the concurrently modified tree. In particular, this
release derives its model-card facts from the regenerated fixture: 84 fictional
earnings calls, four fictional issuers, and periods 2021Q1 through 2026Q1.

## Reproduction and validation

The exact command sequence is documented in the root `README.md`. The release
manifest records notebook execution counts, figure counts, test discovery, and
SHA-256 hashes for the complete auditable release surface.
