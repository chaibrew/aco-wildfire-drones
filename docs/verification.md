# Setup verification — 11 October 2026

- Python 3.11.15 project virtual environment, editable `fire-sim` command installed; dependency versions recorded in `requirements.lock`.
- Published Largefire archive downloaded and MD5 verified: `fdb5f5130194b2d12e544d9672a1a43c`.
- `scripts/fetch_feds.py --year 2018` verified against the cached archive and wrote source provenance.
- `pytest -q`: **21 passed**.
- Real-data CLI: 2018 fireID 3885, first eight snapshots, 500 m grid, 3 drones. Active-point counts: **160, 300, 312, 325, 321, 349, 314, 346**. These are integration checks, not final experimental results.
- Dynamic synthetic, static synthetic, local-growth and distributed-growth CLI runs completed; JSON, CSV and PNG outputs generated.
- Static run: all stages after initial activation report zero changed fraction.
- Controlled static growth at stage 2: realised changed fractions **1/3 local** and **9/14 distributed**; later stages report zero further change. These configurations are not matched-magnitude comparisons.
- Repeated synthetic run: identical snapshots, routes, observation events and metrics, excluding measured planning runtime.
- Router integration example completed three stages and verified predicted and executed objectives agree.

Generated outputs are under `outputs/`; the real-data example is `outputs/feds-2018-3885/`. The sample router, synthetic priorities and compressed replay clock require pilot calibration before algorithm comparisons. ACO/VNS and paired experimental runs have not been implemented or evaluated here.
