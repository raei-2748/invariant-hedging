# Data access and licensing

This repository ships with a tiny synthetic options surface that mirrors the
schema expected by the training code. The asset is bundled **only** to power CI
and local smoke tests; it is not intended to approximate live markets.

## Bundled artefacts

| Dataset | Location | SHA256 |
| --- | --- | --- |
| Synthetic SPY options sample (CSV) | `tools/data_seed/spy_options_synthetic.csv` | `296298b443bed9325dbdf4b0148a663668855ac4328594aa461bc1dbbe8a0f67` |

Running `tools/fetch_data.sh` copies the dataset into `${DATA_DIR:-data}/external/`,
verifies the checksum, and materialises `${DATA_DIR:-data}/raw/spy_options_synthetic.csv`.

## Licence

The synthetic sample was procedurally generated for this project and is
released under the terms of [Creative Commons CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
You may use, modify, and redistribute the bundled sample without restriction.

Real-market datasets (e.g. OptionMetrics IvyDB, CBOE, or Yahoo Finance
downloads) remain subject to their original licences. They are **not**
redistributed in this repository; consult the upstream providers for access.
