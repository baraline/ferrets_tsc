# Results

The CSVs the paper reads. The README at the repository root says which
script writes each file and what the paper reads it for. Units throughout:
times in seconds (wall clock), memory in MB = 10^6 bytes (GiB = MB /
1073.74), accuracies as fractions. An empty cell is a value not measured or
not published. Rows are keyed by the columns named under each file; every
script skips a key already present, so deleting rows is how cells run again.

## Benchmark rows: `bench_ucr.csv`, `bench_multiverse.csv`, `bench_monster.csv`, `bench_cost.csv`, `rivals_ucr.csv`

Written by `experiments/bench.py`, one row per (dataset, model, fold), the
schema of `experiments/benchkit/schema.py`.

- `benchmark`, `dataset`, `model`, `fold`. `FERRETS` is the batch fit (ridge
  head), `FERRETS_stream` the streaming fit; `_unweighted` drops the
  streaming head's class weights and `_perchannel` uses the per-channel
  layout. `TDE_c50` and `cBOSS_c50` are TDE and cBOSS under a 50-minute
  training contract (`time_limit_in_minutes=50`, cost benchmark), distinct
  from the uncontracted `TDE` and `cBOSS` of `ucr_family_resample10.csv`.
- `split`: `published` (the archive's own train/test split: fold 0 of UCR
  and Multiverse; at MONSTER the hub's five folds, so the MONSTER rows are
  `published` at every fold), `resample` (UCR and Multiverse, fold >= 1:
  aeon's stratified resample seeded by the fold) or `synthetic` (a
  generated cost-benchmark cell). In `bench_cost.csv` the folds of one cell
  are repeats of the same split.
- Sizes: `n_train`, `n_test`, `n_channels`, `m` (length), `n_classes`.
- Quality: `acc`, `f1_macro`, `f1_weighted`, `balanced_acc`.
- Time: `time_fit`, `time_pred`, seconds.
- Memory, all in MB: `mem_base_mb` the process's settled RSS before the fit;
  `mem_peak_mb` and `mem_median_mb` the peak and median of an RSS sampler
  (every 20 ms) over fit and predict, minus that base; `mem_peak_rss_mb` the
  sampled peak itself; `mem_samples` the number of samples;
  `mem_wset_peak_mb` Windows' peak working set of the process over the cell,
  minus its value after the warm-up (under WSL2: the maximum resident set).
- `model_bytes`: the size of the pickled fitted model; `n_features`: the
  columns of the model's design where it exposes them.
- `n_threads` (10 throughout; MrSQM gets the same 10 threads but has no
  parallel implementation: its core uses one), `device`, `os` (`win32`, or `wsl2-linux` for
  MrSQM), `python`, `sklearn`, `aeon`, `started_at`.
- `error`: empty for a measured cell, else what happened instead, told by its
  first words. A failure: `timeout after 3600s` (the cell ran past its cap),
  `MemoryError: ...`, `ValueError: ...` (an input the method refuses, such as
  near-constant channels or series shorter than it accepts) or
  `child failed (rc=N): ...` (the cell's subprocess died: the tail of its
  traceback, local paths written `<repo>` and `<venv>`). A skip, never a
  measured failure: `not run: predicted ...` (the feasibility gate's predicted
  time or peak memory reaches the cap or the budget), `not run: rung ...` (a
  rung of the cost benchmark above one that failed, timed out or was gated
  out), `not run: disk: ...` (a disk exclusion), `not run: an earlier FERRETS
  configuration timed out ...` (the two S2Agri-10pc problems at MONSTER: the
  streaming fit without the level word reached the hour there, and the level
  word only adds time to a streaming fit, so they were not attempted) and
  `not attempted: fold N ...` (the later folds of a cell whose earlier fold
  timed out or failed).

`rivals_ucr.csv` holds the UCR resample-10 timings of MultiRocket-Hydra and
RDST and of four anchors measured in the same session (FERRETS, MiniRocket,
Hydra, QUANT).

In `bench_cost.csv`, the MONSTER anchors' rows measured before 2026-10-08
(MiniRocket and QUANT on FruitFlies and LakeIce, MrSQM on LakeIce) read
their input z-normalised per channel over the whole dataset
(`monster_loader.load_monster`); every other MONSTER row, and `bench.py`
today, reads the input as the hub ships it
(`monster_loader.load_monster_raw`). Their `started_at` tells them apart.

## Ablation rows: `ablation_ucr.csv`, `ablation_multiverse.csv`, `ablation_monster.csv`, `ablation_multiverse_resampled.csv`

Written by `experiments/ablation.py` (its docstring lists the arms), one
row per (benchmark, dataset, fold, arm).

- `arm`: `BASE` is FERRETS with its defaults; every other arm changes the
  option it names. `order`: the arm's position among the arms run back to
  back on one (dataset, fold), the order reversed on alternate cells.
- Sizes, `acc`, `balanced_acc`, `f1_macro`, `f1_weighted`, `time_fit`,
  `time_pred` as in the benchmark rows.
- `n_features` (kept columns), `n_views` and `n_views_drawn` (the
  dictionaries kept and drawn), `solver` (the head's route: `gcv` the exact dual
  ridge, `lsqr` the iterative one, `lsqr_cv3` the 3-fold cross-validated
  ridge, `asgd` the streaming head, `logistic_sgd` the logistic head), `alpha` (the chosen ridge
  penalty), `n_threads`.
- `error`: empty, except the five cells of LongIntervalTask at fold 2 in
  `ablation_multiverse_resampled.csv` (one per arm), banked `not run: an
  earlier FERRETS configuration peaked at ...`: a memory skip, so the paper
  reads LongIntervalTask at fold 1 only there.
- `bg_cpu_pct_before`, `bg_wait_s`, `bg_cpu_pct_during`, `quiet_gate`: the
  other processes' CPU share before and during the fit, and whether the
  quiet-box gate passed. The ablation's times are quoted only as ratios to
  BASE on the same cells.

## Timings at UCR resample 10: `ucr_family_resample10.csv`, `ucr_minirocket_resample10.csv`

One row per (dataset, model, fold), fold 10.

- `ucr_family_resample10.csv` (`experiments/ucr_family_resample10.py`):
  `n_train`, `n_test`, `m`, `n_classes`, `acc`, `time_fit`, `time_pred`,
  `n_threads` (MrSQM gets the same 10 threads but has no parallel
  implementation: its core uses one), `error`. Two kinds of rows. Rows with a thread count ran in a
  subprocess under a 3,600 s cap on the whole cell (`timeout after 3600s`
  is the cell's entry: 16 `TDE` cells). The `Hydra` and `cBOSS` rows and 92
  of the `TDE` rows carry no thread count: they were measured in-process at
  `n_jobs` 10 without a cap, and 7 `TDE` and 2 `cBOSS` cells ran past the
  hour (fit plus predict).
- `ucr_minirocket_resample10.csv`
  (`experiments/ucr_minirocket_resample10.py`): `acc`, `time_fit`,
  `time_pred`, `n_features`, the sizes, and `mem_peak_over_base_mb` (the
  child's peak working set over its post-warm-up value, data included),
  `mem_base_mb`, `data_mb` (the train and test arrays). The released memory
  columns come from a child that ran resamples 10-19 in turn; the paper
  reads only the times.

## Memory runs and the MUSE check: `ucr_memory_runs.csv`, `uea_muse.csv`

One row per (dataset, model, n_threads), one subprocess per row.

- `ucr_memory_runs.csv` (`experiments/ucr_memory_runs.py`): the 20 datasets
  of the memory runs at resample 10, every model at 10 threads and FERRETS,
  MiniRocket, Hydra, WEASEL2 and cBOSS also at one thread. `acc`,
  `time_fit`, `time_pred`; `mem_total_mb` the peak working set over its
  value after the warm-up; `mem_algo_mb` the peak working set over the
  settled RSS after loading the data (the memory the fit and predict add);
  `model_bytes` the pickled model; `head_bytes` the linear head's
  coefficients (-1 where the model exposes none). Two `cBOSS` cells on FordA
  are `timeout after 1800s`.
- `uea_muse.csv` (`experiments/uea_muse.py`): WEASEL+MUSE (`MUSE`, its
  default ridge head) and `MUSE_LR` (its logistic head) on the 26
  equal-length UEA problems at the published split, 10 threads; the same
  columns plus `n_channels`, `n_pseudo_channels` and `n_features`. Failures
  are `timeout after 3600s` or a `MemoryError`.

## Studies: `seeds_ucr.csv`, `lsqr_exact.csv`, `ties.csv`

Untimed (their times are not quoted). Each script's docstring defines its
columns.

- `seeds_ucr.csv` (`experiments/seeds_ucr.py`): FERRETS at seeds 1-4 on the
  ablation's 92 UCR datasets, resamples 1-9, keyed (dataset, fold, seed);
  the quality columns, the chosen `alpha` and `prior_lambda`, `n_features`,
  `solver`.
- `lsqr_exact.csv` (`experiments/lsqr_exact.py`): four routes of the batch
  head (`lsqr_default`, `dual_forced`, `lsqr_fixed`, `exact_fixed`) on the
  problems where the exact ridge is still affordable, one row per
  (benchmark, dataset, fold, route): accuracy (`acc`, `bacc`, `f1m`, `f1w`), the chosen
  penalty, the iteration counts, the agreement of the decisions and the
  relative distances of the coefficients, intercepts and decisions between
  routes.
- `ties.csv` (`experiments/ties.py`): per (benchmark, dataset, fold, split,
  key kind), the identical-window groups and whether they always receive one
  key, the rates of exact ties in the PAA and level comparisons, and the
  checks of the scan (expected 0 failures).

## Published results: `published_ucr.csv`, `published_multiverse.csv`, `published_monster.csv`, `published_tsfm.csv`

Results published by others, quality only: no time or memory of theirs is
used. Of the four, the model names of `published_ucr.csv`,
`published_multiverse.csv` and `published_tsfm.csv` carry a `_pub` suffix;
`published_monster.csv` keeps the workbook's names in its `method` column.

- `published_ucr.csv` (`experiments/import_published.py`): the reference
  results of timeseriesclassification.com, 19 estimators on the 112 UCR
  datasets, 30 resamples (fold 0 the published split); `acc`,
  `balanced_acc`, `f1_weighted` in the benchmark schema.
- `published_multiverse.csv` (`experiments/import_published_multiverse.py`):
  the aeon-toolkit/multiverse repository at commit
  5fdf986eaed93369fe8563c58e183bfbb3221559, the published split; `acc`,
  `balanced_acc`, `f1_macro` in the benchmark schema.
- `published_monster.csv` (`experiments/import_published_monster.py`): the
  MONSTER results workbook (`monster_final_results.xlsx` of the MONSTER
  repository, SHA256
  a231ce5b316bd693c00df78b8f45d493bccdce1769bda49e34065fda76fb63cd), every
  sheet in long form: `source_sheet`, `method`, `dataset` (case variants
  unified; `source_dataset` keeps the workbook's spelling), `fold`,
  `classification_error`, `accuracy` (1 - the error), `Balanced_Accuracy`,
  `Weighted_F1`, `log_loss`, the authors' training times, `source`.
- `published_tsfm.csv` (`experiments/import_published_tsfm.py`): the
  MantisV2 report (arXiv:2602.17868v1), zero-shot frozen encoders with a
  logistic-regression head: MantisV2, MOMENT, TiRex, Chronos-2 and NuTime on
  the 112 UCR datasets at the published split, `acc` only.
