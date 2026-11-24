# FERRETS

FERRETS (Fast Exact Random Radix Encodings of Time Series) is a symbolic
transform for time series. It draws random dictionaries of (word source,
window, dilation), keys every window of a series and of its first
difference into an exact 8-bit word (binary PAA and Fourier words, plus a
level word of the window's mean and standard deviation), and returns
l2-normalised bags of sublinear term-frequency counts, one column per word,
in an output whose size is bounded in advance. Multivariate series add
channel-wise words and cross-channel words. Classification is its main
application: a class-weighted ridge head fitted in one batch, or an averaged
least-squares SGD head that streams the design from a cache for data that do
not fit in memory. This repository holds the implementation (two aeon
estimators), the paper ([paper/ferrets.pdf](paper/ferrets.pdf)), every
result the paper reads and the scripts that produce them. The model is
described in [docs/model.md](docs/model.md).

## Install

The package alone, from the repository root:

```bash
pip install .
```

The experiments' environment is Python 3.14.6 with the pins of
`requirements.txt` (the environment every released row was measured in).
aeon 1.5.0 is the binding pin: it caps numba, scikit-learn, pandas, numpy
and scipy, and the other pins are the newest releases it permits. Install
aeon first or keep the pins as written: asking pip for a newer numba makes
it resolve the conflict with an empty `aeon-0.0.0` package.

```bash
py -3.14 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Keep the environment's path short or enable Windows' long paths: torch's
wheel holds files whose paths come close to the 260-character limit.

torch backs Hydra and MultiRocket-Hydra (the CPU build:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`),
sparse backs BORF, psutil reads the memory columns and openpyxl reads the
MONSTER results workbook.

## Usage

```python
from aeon.datasets import load_classification
from ferrets import FERRETS, FERRETSClassifier

X_train, y_train = load_classification("GunPoint", split="train")  # (n, C, m)
X_test, y_test = load_classification("GunPoint", split="test")

clf = FERRETSClassifier(n_jobs=10).fit(X_train, y_train)                # batch fit, ridge head
print((clf.predict(X_test) == y_test).mean())

# The transform alone (no labels): the weighted bag rows, a float64 CSR matrix:
rows = FERRETS(n_jobs=10).fit_transform(X_train)

# Bounded-memory streaming fit (averaged least-squares head), for large data:
clf = FERRETSClassifier(use_streaming=True, n_jobs=10).fit(X_train, y_train)

# Without the level word (each window's mean and std bins; on by default):
clf = FERRETSClassifier(level_word=False, n_jobs=10).fit(X_train, y_train)
```

`FERRETSClassifier` is an aeon classifier: it accepts `(n, m)` or
`(n, C, m)` arrays and labels in a numpy array or pandas Series. `FERRETS`
alone is the label-free transform (an aeon collection transformer); a fitted
classifier holds it as `clf.transformer_`.

## Results

Every CSV the paper reads. [results/README.md](results/README.md) gives
their columns and units, and
[results/per_dataset/README.md](results/per_dataset/README.md) the
per-dataset tables that replace the paper's longtables.

| File | Contents | Written by | Read by the paper for |
| --- | --- | --- | --- |
| `results/bench_ucr.csv` | FERRETS and BORF on the 112 UCR datasets, resamples 0-29; the streaming head with and without class weights on the ablation's 92 datasets, resamples 1-9 | `experiments/bench.py` | Figure 1, the UCR comparisons of Section 5.3 and Appendix C, the UCR cost table (Appendix B), the streaming class-weight row of the ablation |
| `results/bench_multiverse.csv` | FERRETS, MiniRocket, Hydra and BORF on the 133 Multiverse problems at the published split; FERRETS at resamples 1-2; the per-channel layout at folds 0-2 | `experiments/bench.py` | the Multiverse comparisons of Section 5.3 and Appendix C, the Multiverse cost table, the channel-layout row of the multivariate ablation, Appendix E's input-check error |
| `results/bench_monster.csv` | FERRETS (streaming head) and MiniRocket on the 29 MONSTER problems, folds 0-4; the batch head on nine problems; the streaming head without class weights | `experiments/bench.py` | the MONSTER comparisons of Section 5.3, the MONSTER cost figure of Section 5.2, Appendix A's disk and memory exclusions, the MONSTER class-weight row of the ablation |
| `results/bench_cost.csv` | the cost benchmark: 13 synthetic cells varying one axis at a time and six real anchors, 13 methods, 3 repeats | `experiments/bench.py` | the cost curves of Section 5.2 and Appendix B, the scaling table, the per-channel layout's cost |
| `results/rivals_ucr.csv` | MultiRocket-Hydra and RDST at UCR resample 10, beside FERRETS, MiniRocket, Hydra and QUANT timed in the same session | `experiments/bench.py` | Figure 1's MultiRocket-Hydra and RDST, their paired time ratios, Appendix B's drift of the anchors |
| `results/ablation_ucr.csv` | the ablation's 25 arms on 92 UCR datasets, resamples 1-9, and the four step and epoch choice arms on its 37 development-third datasets | `experiments/ablation.py` | the ablation figure of Section 5.5, the UCR ablation table (Appendix D), the cost of the ablation's alternatives (Appendix B), the development third, the streaming head's step and epochs (Appendix A) |
| `results/ablation_multiverse.csv` | the ablation's arms on 57 problems of the Multiverse core, published split | `experiments/ablation.py` | the multivariate ablation table (Appendix D) and the ablation figure, the cost of the cross-channel words |
| `results/ablation_monster.csv` | nine arms on the 18 MONSTER problems Multiverse also carries, fold 0, streaming fit | `experiments/ablation.py` | the multivariate ablation table and the ablation figure, the MONSTER cost of the level word and of the bag additions |
| `results/ablation_multiverse_resampled.csv` | BASE, the three cross-channel variants and NO_LEVEL on 73 Multiverse problems, resamples 1-2 | `experiments/ablation.py` | the resampled rows of the multivariate ablation table |
| `results/ucr_family_resample10.csv` | WEASEL 2.0, Hydra, MrSQM, TDE, BORF, cBOSS and QUANT on the 112 UCR datasets at resample 10 | `experiments/ucr_family_resample10.py` | Figure 1 and the UCR cost table (time), the UCR time ratios of Section 5.2, Appendix B's cell counts |
| `results/ucr_minirocket_resample10.csv` | MiniRocket on the 112 UCR datasets at resample 10 | `experiments/ucr_minirocket_resample10.py` | Figure 1 and the UCR cost table (time) |
| `results/ucr_memory_runs.csv` | nine methods on the 20 datasets of the memory runs at resample 10, at 10 threads and, for five of them, one thread | `experiments/ucr_memory_runs.py` | Figure 1's memory axis, the memory columns of the UCR cost table, the one-thread numbers of Section 5.2 |
| `results/uea_muse.csv` | WEASEL+MUSE with its default ridge head and with its logistic head on the 26 equal-length UEA problems | `experiments/uea_muse.py` | Appendix E |
| `results/seeds_ucr.csv` | FERRETS at seeds 1-4 on the ablation's 92 UCR datasets, resamples 1-9 | `experiments/seeds_ucr.py` | the seed paragraph of Appendix A and the seed sentence of Section 5.1 |
| `results/lsqr_exact.csv` | the batch head's iterative ridge route against the exact ridge, four routes on 25 (problem, fold) units | `experiments/lsqr_exact.py` | Appendix H's comparison of the two routes |
| `results/ties.csv` | identical windows and exact ties of the keys on the 112 UCR datasets, two synthetic series sets and two MONSTER problems | `experiments/ties.py` | the ties table (Appendix H) and the share of identical windows of Section 3 |
| `results/published_ucr.csv` | 19 published estimators on the 112 UCR datasets, 30 resamples: accuracy, balanced accuracy, F1 | `experiments/import_published.py` | every UCR comparison |
| `results/published_multiverse.csv` | the Multiverse archive's published estimators at the published split: accuracy, balanced accuracy, F1 | `experiments/import_published_multiverse.py` | every Multiverse comparison and the coverage rule |
| `results/published_monster.csv` | the MONSTER results workbook in long form, every sheet | `experiments/import_published_monster.py` | every MONSTER comparison and Appendix F's workbook check |
| `results/published_tsfm.csv` | five foundation models on the 112 UCR datasets at the published split, accuracy | `experiments/import_published_tsfm.py` | the foundation-model matrix of Section 5.4 |

## Reproducing

Every script runs from the repository root and resumes: a row already in
its CSV is skipped, so deleting rows is how a cell runs again. Each one
takes `--dry-run`, which prints the cells left to run and ends with the
line `cells to run: <N>`. On this release every dry run plans 0 cells:

```bash
python experiments/bench.py queue --target all --dry-run
python experiments/ablation.py run --benchmark ucr --dry-run
python experiments/ablation.py run --benchmark ucr --arms STREAM_EPOCHS_10,STREAM_EPOCHS_40,STREAM_ETA_HALF,STREAM_ETA_DOUBLE --datasets development --dry-run
python experiments/ablation.py run --benchmark multiverse --dry-run
python experiments/ablation.py run --benchmark monster --dry-run
python experiments/ablation.py run --benchmark multiverse --arms BASE,META_NONE,META_COARSE,META_FLAGS,NO_LEVEL --datasets resampled --folds 1,2 --csv results/ablation_multiverse_resampled.csv --dry-run
python experiments/seeds_ucr.py --dry-run
python experiments/lsqr_exact.py --dry-run
python experiments/ties.py --dry-run
python experiments/ucr_family_resample10.py --dry-run
python experiments/ucr_minirocket_resample10.py --dry-run
python experiments/ucr_memory_runs.py --dry-run
python experiments/uea_muse.py --dry-run
```

Drop `--dry-run` to run. `experiments/bench.py queue --target all` runs
every benchmark step in order and ends with `COMPLETE` or `INCOMPLETE`;
`experiments/bench.py run --benchmark B --arms A --datasets D --folds F`
runs the named cells that have no row. The importers fetch the published
results again (`python experiments/import_published.py`, and likewise for
`_multiverse`, `_monster` and `_tsfm`). The Multiverse importer reads the
archive at a commit, and the MONSTER and foundation-model importers check
the workbook's and the e-print's checksum; the UCR importer reads
timeseriesclassification.com as it is served, so a later import may differ
from the released file.

Data. The UCR and Multiverse archives are downloaded by aeon into
`$AEON_DATA`, else `~/aeon_data`; the benchmark caches the Multiverse
problems as `.npz` files under `~/aeon_data/_bench_npz` (`bench.py
prefetch`) and drops a dataset's download once no unbanked cell needs it
(`--keep-cache` keeps it, `bench.py reap` sweeps by hand). MONSTER is read
from the Hugging Face hub (`monster_loader.py`) into the hub's cache. The
benchmark refuses to start below 10 GiB free, and a cell whose data or
streaming cache would not fit a healthy disk (60 GB free) is excluded: a
`not run: disk` row.

Protocol. Every row was measured on one machine: an AMD Ryzen 5 5500 (6
cores, 12 threads), 64 GiB of DDR4 memory and an NVMe SSD, under Windows 11
Pro, the scripts launched as above. The benchmark, the ablation, the MUSE
check and the timings at resample 10 ran at 10 threads (an estimator's
`n_jobs` equals numba's thread count), the memory runs at 10 threads and,
for five methods, one; the untimed studies (`seeds_ucr.py`, `lsqr_exact.py`,
`ties.py`) ran at 6 threads. The `n_threads` column, where a file has one,
records the count; MrSQM gets the same 10 threads but has no parallel
implementation: its core uses one. `bench.py`, the memory runs, the MUSE check and the
timings at resample 10 run one subprocess per cell, and a cell that runs
past its cap is a timeout row: one hour per fold for the benchmark, 1,800 s
for the memory runs, 3,600 s for the MUSE check. The family timings at
resample 10 run uncapped unless `--cap` is given; their released rows with
an empty `n_threads` were measured in-process, without a cap. The ablation
and the studies fit in-process and cannot be capped: the ablation banks a
cell `not run` where the benchmark's FERRETS row of the same cell failed or
ran too long. Times
measured on another machine are not comparable with the released rows, and
on one machine the same code drifts between sessions (Appendix B measures
the drift of the anchors). The first run of a fresh checkout compiles the
numba kernels into their cache; warm the cache before any measured run, or
the first cells measure compilation (the memory columns included):

```bash
python -c "from aeon.datasets import load_classification as load; from ferrets import FERRETSClassifier as F; [F(use_streaming=s, n_jobs=10).fit(*load(d)) for d in ('GunPoint', 'BasicMotions') for s in (False, True)]"
```

MrSQM has no Windows wheel. It runs in a separate Linux interpreter under
WSL2: a Python 3.14 environment with the pins of `requirements.txt` except
scikit-learn 1.6.1 (mrsqm 0.0.7 needs scikit-learn below 1.7), plus mrsqm
0.0.7 and the FFTW3 library it builds against. Set `FERRETS_MRSQM_PYTHON` to
that interpreter's path inside WSL (for example `~/venv-mrsqm/bin/python`)
and `FERRETS_WSL_DISTRO` to the WSL distribution (default `Ubuntu`). Without
`FERRETS_MRSQM_PYTHON` the MrSQM cells are left unrun and the queue ends
`INCOMPLETE`. The WSL child is given the Windows side's data home
(`AEON_DATA`) and Hugging Face cache, so both sides read the same downloads.
MrSQM's rows carry `os = wsl2-linux` in the `bench_*.csv` files;
`ucr_family_resample10.csv` has no `os` column, and its MrSQM rows ran the
same way.

Skips. The feasibility gate predicts the time and the peak memory of every
gated cell (the slow methods, every MONSTER and cost cell, and the 18
Multiverse problems MONSTER also carries) from the rows already in
`results/`, and skips a cell predicted to reach the hour or 56,000 MB, or
whose method already failed on that dataset. A
skip is a `not run: <prediction>` row, never a measured failure. Deleting
such a row runs the gate again on that cell, from the rows that remain.

## Citation

```bibtex
@misc{guillaume2026ferrets,
  title = {FERRETS: A Scalable Symbolic Transform for Time Series Classification},
  author = {Antoine Guillaume},
  year = {2026},
  howpublished = {\url{https://github.com/baraline/ferrets_tsc}}
}
```

## License

The code is under the BSD 3-Clause License (`LICENSE`). The paper,
`paper/ferrets.pdf`, is the author's and is not covered by it. The
`results/published_*.csv` files reproduce results published by others
(timeseriesclassification.com, the aeon-toolkit/multiverse repository, the
MONSTER results workbook and the MantisV2 report) and remain under their
sources' terms.
