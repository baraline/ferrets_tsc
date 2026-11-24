# Per-dataset results

The accuracy and balanced accuracy of FERRETS and of the methods it is compared
with, one row per problem. These files replace the per-dataset tables of the
paper (Appendix F points here). They are computed from the benchmark rows and
the published results in `results/`.

| File | Problems | Value |
| --- | --- | --- |
| `ucr.csv` | the 112 UCR datasets | mean over the 30 resamples (resample 0 is the published split) |
| `multiverse.csv` | the 133 Multiverse problems | the published split |
| `monster.csv` | the 29 MONSTER problems | mean over the five folds |

Columns:

- `dataset`, then the sizes: `n_train` (training series; `n` at MONSTER, all
  series), `n_channels` (not in `ucr.csv`: the UCR datasets are univariate),
  `m` (length). In `multiverse.csv`, `core` and
  `in_monster` are 1 for a problem of the 65-problem core and for one MONSTER
  also carries, and `m_is_longest` is 1 where the lengths vary and `m` is the
  longest series. `development` is 1 for a problem of the development third,
  the problems FERRETS' defaults were chosen on (37 UCR datasets, 44
  Multiverse problems counting the five MONSTER problems Multiverse also
  carries, 9 MONSTER problems; `DEVELOPMENT_THIRD` in
  `experiments/ablation.py`).
- `<method>_acc` and `<method>_balanced_acc`, six decimals. An empty cell is a
  missing published result or a cell run here without a result.
- `<method>_note`, present where a cell of the method needs one: the state of a
  cell run here without a result (predicted skip, not attempted, disk
  exclusion) or its failure (timeout, out of memory, near-constant channels
  refused, series shorter than the method accepts, unequal lengths). In
  `monster.csv` a note also marks a workbook balanced accuracy that cannot
  come from the same predictions as the accuracy on at least one fold.

Run here: FERRETS on the three archives (batch fit on UCR and Multiverse,
streaming fit at MONSTER), MiniRocket and Hydra on Multiverse, MiniRocket at
MONSTER. Every other column is a published result: the archives' reference
results for UCR and Multiverse, the MONSTER results workbook (its first sheet)
at MONSTER, which gives a balanced accuracy for ConvTran, FCN, H-InceptionTime
and TempCNN only.
