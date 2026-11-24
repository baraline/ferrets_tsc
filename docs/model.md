# The FERRETS model

FERRETS (Fast Exact Random Radix Encodings of Time Series) is a symbolic
transform for time series and a classifier built on it. This document
describes both, top-down. Parameter lists are in the class docstrings
(`help(ferrets.FERRETS)`, `help(ferrets.FERRETSClassifier)`).

## Overview

1. Random dictionaries are drawn: each is a (word source, window length,
   dilation) triple.
2. Every dilated window of a series, and of its first difference, is keyed into
   an exact 8-bit word (256 keys): a binary PAA word or a binary Fourier word.
   A level word per window adds the bins of its mean and standard deviation.
3. A word is an arithmetic column of the output: column = block offset + key.
   There is no hash map and no vocabulary, so the output width is bounded by
   the number of draws.
4. The counts of the words form a sparse bag per series, weighted by
   1 + ln(count) and scaled to unit l2 norm.
5. A linear head scores the rows: an exact ridge classifier (batch) or an
   averaged least-squares SGD (streaming, for data whose design does not fit
   in memory).

Nothing in steps 1 to 4 reads a label. The package has two estimators:

| Class | aeon base | Does |
| --- | --- | --- |
| `ferrets.FERRETS` | `BaseCollectionTransformer` | steps 1 to 4: `fit` draws the dictionaries and cuts and keeps the columns, `transform` returns the weighted rows as a float64 scipy CSR matrix |
| `ferrets.FERRETSClassifier` | `BaseClassifier` | the transform plus the head; the fitted transform is `clf.transformer_` |

`decision_function(X)` equals `clf.transformer_.transform(X) @ clf.coef_ +
clf.intercept_` up to rounding. At prediction the classifier never assembles
the rows: it scores through the keying kernel.

## Glossary

| Term | Meaning | Code name |
| --- | --- | --- |
| series | one `(C, m)` array: C channels, m time points | `n_channels_`, `n_timepoints_` |
| side | the channels as given, or their first differences. A series has two sides (one with `use_first_difference=False`). The first difference is `x[t+1] - x[t]` with the last increment repeated, so its length is m | |
| pseudo-channel | the unit that owns a span of columns. Univariate: the series and its difference (2). `channel_layout="per_channel"`: every channel and its difference (2C). Family layout: the sides | `feature_offsets_` spans |
| dictionary | one random draw: (source, window, dilation). The source is PAA or Fourier | `word_sources`; `"paa"`, `"fourier"` in `views_` and in decoded words |
| view / family | the dictionaries of a pseudo-channel that share source and window and differ only in dilation. In the family layout they also share a channel subset, and the view is called a family | `views_` |
| window | w samples at positions `j, j+d, ..., j+(w-1)d`; w is a multiple of 8 | `windows` |
| dilation | the step d between the samples of a window. A window spans `(w-1)d + 1` samples, and a series of length m has `m - (w-1)d` windows of that length and dilation | |
| chain | the residue chain `x[r], x[r+d], ...`: the windows of dilation d slide along d chains, and the kernels build each chain's prefix sums once | |
| slot | one (view, dilation) pair; it owns a run of blocks | |
| word, key | the 8-bit code of one window, 0 to 255 | |
| block | 256 consecutive columns, one per key, holding one kind of word of one slot | |
| Fourier cuts | the cut tables of the Fourier views, 8 cuts per view (per channel slot in a family) | `fourier_cuts_` |
| level word | the word of a window's mean and standard deviation bins | `level_word`, `level_cuts_` |
| channel-wise word | the word of one channel of a family under the family's dictionary | |
| cross-channel words | the coarse word, change flags and agreement flags of a family | `cross_channel_words` |

## The transform

### The draw

For each pseudo-channel the transform draws `n_dictionaries` times:

- a source, uniformly among `word_sources` (PAA, Fourier, or both);
- a window length, uniformly among the `windows` no longer than m;
- a dilation `d = floor(2^U)`, U uniform on `[0, log2(max_d + 1)]`,
  `max_d = (m - 1) // (w - 1)`, clipped to `[1, max_d]`, so a window always
  fits the series. `dilation="one"` sets every dilation to 1 after the draw.

Repeated draws collapse, so there are at most `n_dictionaries` slots per
pseudo-channel. The draw depends on `random_state` only. Each pseudo-channel
has its own random stream.

### Window moments, flat windows

Each chain is centred on its mean and its prefix sums of the values and of the
squares give every window's mean and standard deviation. A window whose
variance is tiny against the chain's energy or its own is recomputed in two
passes. A window with standard deviation at most 1e-8 is flat.

### The PAA word

The window is cut into 8 segments of w/8 samples. Bit b (b = 0 is the most
significant) is 1 when the sum of segment b exceeds one eighth of the sum of
all eight segment sums: the binary z-normalised PAA symbol at the zero cut.
The threshold is a fixed-order sum of the eight segment sums, so identical
windows get the same key wherever they sit in a chain. A flat window has key 0.

### The Fourier word and its cuts

The window's DFT coefficients 1 to 4, divided by `w x std`, give eight parts:
`re_1, im_1, ..., re_4, im_4`. Bit b is 1 when part b exceeds its cut, with
`re_1` the most significant. The cuts of a Fourier view are fitted from the
training series: the median of each part over randomly drawn training windows
(at most `config.FOURIER_MAX_WINDOWS` = 20,000 per view, shared among its
dilations), plus a margin (below). A flat window gets the word that
zero-valued parts would get.

### Ties

A comparison is decided by the window's own values, not by where the window
sits or how its moments rounded. The stored cuts carry margins that decide a
value landing on a cut:

| Cut | Stored as |
| --- | --- |
| Fourier cut c | `c + 1e-6` |
| level mean cut c | `c + 1e-10 x (abs(c) + spread)`, spread = largest minus smallest mean cut of its set |
| level std cut c > 0 | `c x (1 + 1e-5)`; cuts equal to 0 stay 0 |

### The level word

Both words are normalised by the window's mean and standard deviation, so
they are blind to its level and spread; the level word keys them. Per
(side, channel, window) a cut set is fitted on training windows sampled like
the Fourier cuts' (over the union of the dilations of the views keying that
channel and window): 15 mean cuts and 15 std cuts, the quantiles j/16 of the
sampled windows' means and standard deviations. Std cuts at most 1e-8 are set
to 0. A window's bin is the number of cuts strictly below its value (0 to 15).
The word is the mean bin in the high 4 bits and the std bin in the low 4 bits.
A flat window has std bin 0.

The level word has one block per (pseudo-channel, window, dilation), shared by
every view keying that window and dilation (in the family layout also that
channel subset): the first view in plan order owns it, as the last block of
its slot. `level_word=False` drops it.

### Multivariate input

Under `channel_layout="family"` (the default) and C >= 2, each side draws
`n_dictionaries` families. A family is a dictionary plus a channel subset of
size k, k uniform among {2, 4, 8} not above C, channels drawn without
replacement. Families that share (source, window, channels) merge their
dilations. A slot of a family has these blocks, in order:

| Block | Word |
| --- | --- |
| k channel-wise words | each channel keyed under the family's dictionary, with its own Fourier cuts |
| coarse word | the top b = 8/k bits of every channel's word, channel 0 most significant (a PAA word over max(b, 2) segments, top b bits) |
| change flags | k bits: 1 when the channel's word differs from its word one dilation step back on the same chain (0 for the first window of a chain) |
| agreement flags | k-1 bits: bit c is 1 when channel c's word equals channel 0's, c >= 1 |
| level word | per channel b = 8/k bits (ceil(b/2) mean bits above floor(b/2) std bits), channel 0 most significant, one cut set per channel |

`cross_channel_words` picks the cross-channel words: `"all"` (coarse, change
and agreement flags), `"coarse"`, `"flags"` (change and agreement), `"none"`.
At most eight channel words are keyed per family, so the cost follows the
draws, not C.

`channel_layout="per_channel"` gives every channel and its difference its own
univariate draw and keys no cross-channel words (it accepts only the default
`cross_channel_words="all"`). With one channel the two layouts are the same
model.

### The bag

Every window increments the count of its key in its block. Four options shape
the rows:

| Parameter | Default | Effect |
| --- | --- | --- |
| `min_doc_freq` | 1 | columns reached by fewer training series are dropped; a value that drops every column raises |
| `use_idf` | False | True multiplies a column by `ln((1 + n) / (1 + df)) + 1`, df the document frequency |
| `numerosity_reduction` | False | True counts a key once while it repeats at consecutive window starts of a block |
| `sublinear_tf` | True | a count c is weighted 1 + ln(c); False weights c |

Each row is then scaled to unit l2 norm (a row with no kept key stays zero).
The design is computed in float64 and stored in float32.

### The column layout

A pseudo-channel's columns are its slots in plan order (views sorted by
(source, window), dilations ascending), each slot being its blocks (plus its
level block when it owns one) of 256 columns. After the training pass the
columns no training series reaches are dropped and the kept columns are
renumbered per pseudo-channel; `feature_offsets_` gives where each
pseudo-channel starts and `n_features_` the width. A row has at most
`min(n_windows, 256)` entries per block. The width is bounded by the draws: a
univariate model has at most `2 x n_dictionaries x 2 x 256` columns (102,400 at
the defaults).

## Classification

`FERRETSClassifier` fits its `transformer_` (the same draw and cuts as a
`FERRETS` with the same parameters), then a linear head on the rows. Targets
are one-vs-rest values in {-1, +1}, one column per class; two classes use one
column whose positive class is `classes_[1]`. The prediction is the column
with the largest score (two classes: the sign of the score).

### The batch head

`use_streaming=False` fits a ridge regression with an unpenalised intercept:
`min sum_i w_i ||y_i - b - beta' x_i||^2 + alpha ||beta||^2`, with the weights
`w_i` of the class weights below (all 1 when unweighted). The training size
picks the route:

| Route (`solver_`) | When | How |
| --- | --- | --- |
| `"gcv"` | n_train <= `config.DUAL_MAX_N` (3000) | The Gram matrix X X' is accumulated in dense column slabs with BLAS syrk, centred by the (weighted) column means and eigendecomposed. For every alpha the exact leave-one-out squared error comes in closed form; the alpha with the smallest error wins (ties keep the earlier). The weights `coef_` follow from the dual solution |
| `"lsqr"` | larger | LSQR (Golub-Kahan bidiagonalisation, re-orthogonalised basis, tolerance `config.RIDGE_TOL` = 1e-2, at most 200 iterations), classes solved in lockstep groups so that one pass over the design serves a group, every alpha from one bidiagonalisation. The alpha is the most accurate on a 25% hold-out (stratified when the classes allow it; the first wins ties); the final fit uses all rows |

The Gram route keeps the design in a compact encoding (about 2 bytes per
non-zero) and decodes it piece by piece; the LSQR route decodes it once into a
flat CSR (6 bytes per non-zero).

`alphas` is the grid (default 1e-3 to 10, decades). `design_ram_mb` caps the
RAM of the encoded blocks on the Gram route; blocks past it are encoded again
when needed. `alpha_`, `alpha_scores_` and `cv_score_` hold the choice.

### Class weights and the prior shift

| Option | Default | Effect |
| --- | --- | --- |
| `prior_shift` | True | batch head only, see below |
| `batch_class_weight` | `"sqrt"` | row weight of class c: `w_c = n_c^-0.5 x n / sum_c' n_c'^0.5` (mean 1 over the rows). It weights the ridge loss, the leave-one-out score, the hold-out fit and the column and target means. `None`, or exactly balanced classes, fits the unweighted ridge |
| `stream_class_weight` | `"sqrt"` | the streaming head scales each row's SGD update by `(n / (K n_c))^0.5`, divided by its largest value over classes. `None`, or balanced classes, leaves updates unweighted |

A least-squares head on {-1, +1} targets carries the training prior
`ybar_k = 2 pi_k - 1` (the class's weighted share `pi_k`) in its intercept.
With many more columns than series, the learned part of a score on an unseen
series is small and that prior can decide the class. The prior shift scores
`d - lambda x ybar`, with `lambda` in `config.PRIOR_LAMBDAS` = (0, .25, .5,
.75, 1) chosen on out-of-sample decisions of the training set: the leave-one-out
decisions of the Gram route, or the hold-out decisions of the LSQR route (a
single alpha has no hold-out, so no shift). The chosen lambda is the smallest
whose equal-weight score (mean of accuracy, balanced accuracy and macro F1) is
within one binomial standard error, `sqrt(s (1 - s) / n)`, of the best. It is
folded into `intercept_` and kept in `prior_lambda_`. The streaming head is
never shifted.

### The streaming head

`use_streaming=True` fits an averaged least-squares SGD: squared loss, L2
penalty `config.SQ_ALPHA` = 1e-5, constant step `config.SQ_ETA0` = 1, averaged
weights, `config.SQ_EPOCHS` = 20 epochs, one weight column per class (one for
two classes). It follows sklearn's averaged SGD with float32 weights behind a
lazy L2 scale that is folded back when it falls below
`config.WSCALE_RESET`.

1. One transform pass over row chunks encodes every chunk into a cache and
   counts the document frequencies, so the kept columns, the idf and the cache
   come from the same pass. A chunk holds about `config.STREAM_CHUNK_NNZ`
   non-zeros, estimated from the first 64 rows.
2. Each epoch visits the chunks in a random order and the rows of a chunk in a
   random order, both derived from the seed. A chunk's pass runs on all threads
   at once; the next chunk is decoded after it, and read ahead from disk when
   the cache is there.

The cache stays in RAM when its estimate is at most `config.STREAM_RAM_MB`
(256 MB). Otherwise it is written to the temporary directory when it fits half
of the free disk (`config.STREAM_DISK_FRAC`). Otherwise a `RuntimeWarning` is
raised and the transform is recomputed every epoch. The result does not
depend on the cache mode. `fit_stats_["cache_mode"]` records it.

### Prediction

`predict` and `decision_function` score a collection without assembling its
rows: the keying kernel counts a row's words and adds the head weights of the
kept columns while it counts, tile by tile, then divides by the row's norm.
`predict_proba` is `expit` of the decision, per class, normalised to sum to 1
over classes (two classes: `[1 - p, p]`). It is a monotone map of the scores,
not a calibrated probability.

## Settings

`ferrets.config` holds the runtime settings, read as `config.NAME` when used:
patch them on the module, for instance `ferrets.config.DUAL_MAX_N = 8`. The
file groups them in the order of the tables below. `ferrets.constants` holds
what numba compiles into the kernels (word length 8, the flat-window
threshold, the tie margins, the key space): not settings.

Marked **model**, a setting changes the fitted model; the others change memory
and speed only (the partition counts can move results at rounding level).

Process-wide tables (not settings), built once and shared by every model and
call with the same key, never written after the build: the trigonometric
tables of a window set and a series length (64 x windows x m bytes, 98 KB at
m = 256 and 6.1 MB at m = 16,000 for the six default windows; the last 4 are
kept) and the tf tables of a length (12 x (m + 1) bytes, 3 KB and 192 KB; the
last 8 are kept). No fitted model holds them.

Cut sampling (read at fit):

| Name | Default | Controls |
| --- | --- | --- |
| `FOURIER_MAX_WINDOWS` (**model**) | 20,000 | windows sampled per Fourier or level cut job |
| `SAMPLER_BATCH_MB` | 64 | replayed draws held per sampler batch |
| `LEVEL_SAMPLER_BLOCK` | 64 | samples per block of the level cut sampler's sums (a schedule: same floats for every size) |

Keying (read at every call):

| Name | Default | Controls |
| --- | --- | --- |
| `PREDICT_ROW_TILE` | 16 | rows per tile in prediction (1: row by row). A tile's rows are keyed slot by slot, so they read one slot's head rows back to back; a row still gets its blocks in the same order |
| `KEY_SCRATCH_MB`, `RESULT_STORE_MB` | 2, 4 | chain-major keying: key scratch of a row block (it also sets predict's row tile, at most `PREDICT_ROW_TILE`) and the result store; a shared dilation that does not fit runs the per-pair body |
| `WORD_TABLE_MB` | 1 | table of one-byte words that lets a family key a shared channel word once and copy it (0: no sharing); the plan shares as many words as fit `WORD_TABLE_MB / m`, most used first |

Chain-major keying: within the views of one pseudo-channel, a residue chain
read by two or more (view, dilation) pairs is gathered and its prefix sums are
built once per row, its window moments once per window length. The pairs' keys
go to a scratch, their blocks to a result store, and every row receives its
blocks in plan order: the same floats in the same order as the per-pair kernel.
A shared word is the same side, channel, source, window and dilation and, for
a Fourier word, the same eight cut values bit for bit; a cached word skips the
keying, while the gather and the moments still run where the level word or the
PAA coarse word needs them.

Transform buffers:

| Name | Default | Controls |
| --- | --- | --- |
| `TRANSFORM_CHUNK_MB` | 256 | id and count buffers of one transform chunk |
| `TRANSFORM_SLICE_SLOTS` | 2^20 | (row bound x row) slots per slice when `transform` maps keys to columns |
| `ROW_PARTS` | 40 | row partitions of the parallel reductions |

Batch head, Gram route:

| Name | Default | Controls |
| --- | --- | --- |
| `DUAL_MAX_N` (**model**) | 3000 | largest training set on the Gram route |
| `PRIOR_LAMBDAS` (**model**) | (0, .25, .5, .75, 1) | grid of the prior shift (both batch routes) |
| `GRAM_SLAB_MB`, `GRAM_PACK_MB` | 256, 64 | widest dense Gram slab; narrow blocks share a slab up to this size |

Batch head, LSQR route:

| Name | Default | Controls |
| --- | --- | --- |
| `RIDGE_TOL`, `RIDGE_MAX_ITER` (**model**) | 1e-2, 200 | LSQR stopping tolerance and iteration cap |
| `HOLDOUT` (**model**) | 0.25 | hold-out share of the alpha choice |
| `LSQR_GROUP`, `LSQR_STATE_MB`, `LSQR_K_GUESS` | 8, 1024, 16 | classes per lockstep group and the budget and iteration guess that size it |
| `RMV_PARTS`, `RMV_BAND`, `RMV_SCRATCH_MB` | 8, 65536, 2 | transposed product: row partitions, column band, and the scratch limit (band columns x padded column count x 8 bytes per partition) above which it goes by column ranges, because past the cache every entry of the row-parts kernel misses |
| `FLAT_BAND_BITS` | 16 | design blocks wider than 2^bits columns are cut into sub-blocks (uint16 ids); above 16 each pseudo-channel keeps one block |

Streaming head:

| Name | Default | Controls |
| --- | --- | --- |
| `ASGD_PARTS` | 40 | entry partitions of a row's dot products |
| `SQ_ETA0`, `SQ_ALPHA`, `SQ_EPOCHS` (**model**) | 1.0, 1e-5, 20 | step, L2 penalty, epochs |
| `INTERCEPT_DECAY`, `MAX_DLOSS` (**model**) | 0.01, 1e12 | the intercept's step scale and loss-derivative clip (sklearn's sparse SGD) |
| `WSCALE_RESET` (**model**) | 1e-2 | the lazy L2 scale is folded into the float32 weights below this (about every 460,000 steps per head); the averaged weights come out of a cancellation whose terms grow like 1 / scale, so a much smaller threshold loses them in float32 |
| `STREAM_CHUNK_NNZ`, `STREAM_PROBE_ROWS` (**model**) | 2.5e7, 64 | chunk size in non-zeros, and the rows probed to size it |
| `STREAM_RAM_MB`, `STREAM_DISK_FRAC`, `STREAM_EST_BYTES` | 256, 0.5, 2.4 | cache placement: RAM limit, share of free disk, bytes per non-zero of the estimate |

Test switches (the off paths are slower and compute the same values; the
tests use them as oracles), all True by default: `CHAIN_KEYING` (chain-major
keying of the univariate kernel), `DEDUP_FOURIER_JOBS` (a Fourier cut job that
repeats another, such as a channel two families share at the same window and
dilations, runs once), `FUSE_LEVEL_CUTS` (a level cut job whose draw and data
are a Fourier job's reuses that job's window means and deviations).

## The plan

The fit builds one flat plan, a dict of arrays (int64 unless noted) for V
views, S dilation slots and P blocks (`PlanMixin._build_plan`). A view is a
(source, window, dilations) configuration (a family view also has its
channels); a block is one pseudo-channel of
the per-channel layout (univariate: the series, then its first difference) or
one side of the family layout, and owns a span of columns.

| Key | Shape | Meaning |
| --- | --- | --- |
| `kind` | (V,) | `_KIND_PAA` or `_KIND_FOURIER` |
| `win` | (V,) | window length |
| `ch` | (V,) | channel (family: the first one) |
| `diff` | (V,) | 1 on the first-difference side |
| `dptr` | (V+1,) | view v owns slots `dptr[v]:dptr[v+1]` |
| `cuts` | (V, W) | float64 Fourier cut row, zeros for PAA |
| `col`, `tab` | (V,) | first column in its block / in the table of all blocks |
| `lrow` | (V,), family plan: (`cptr[-1]`,) | level cut-set row of the view; in the family plan, of each channel slot |
| `trig` | (V,) | index of the window in `windows` |
| `woff` | (V,) | first kept column of the view's block: zero in the plan, set at fit once the kept columns are known |
| `dils` | (S,) | dilation of the slot |
| `dcol`, `dtab` | (S,) | as `col`, `tab`, per slot |
| `own_level` | (S,) | 1: the slot owns the level block |
| `pv_ptr` | (P+1,) | block p owns views `pv_ptr[p]:pv_ptr[p+1]` |
| `n_cols` | (P,) | columns of the block |
| `pc_tab` | (P,) | offset of the block in the table |
| `row_bound` | (P,) | most entries one row has in the block |
| `lcuts` | (J, 30) | float64 level cut sets, (0, 30) without the level word |
| `windows` | (nw,) | sorted distinct windows |
| `level`, `tab_total`, `P` | scalars | level word flag, table width, block count |

The univariate kernel's plan adds `flat` (V,), the key a flat window takes.
The family layout adds `cptr` (V+1,) and `chs` (the channel slots of each view
and their channels), `cross` (the cross-channel code) and `wptr`, `wref`,
`n_wslots` (the channel-word numbers). Fit adds `sublinear_tf` and
`numerosity_reduction` (the fitted options), the ablation `word_bits`.

## The package

`ferrets/transformer.py` and `ferrets/classifier.py` hold the two estimators;
each is assembled from mixins of `ferrets/transform/` and
`ferrets/classification/`, over numba kernels (`ferrets/kernels/`), settings
(`config.py`, `constants.py`) and helpers (`ferrets/helpers/`). The transform
imports nothing from the heads.

| Module | Holds |
| --- | --- |
| `transform/views.py`, `plan.py`, `cuts.py`, `chains.py` | the draw of the views, the plan, the cut tables, the keying schedule |
| `transform/run.py`, `encoding.py` | the keying kernels' entry, the encoded blocks and the term-frequency tables |
| `transform/features.py` | the kept columns and idf, the batch counting pass, `_rows` (the weighted rows) and `_score` |
| `transform/design.py` | the encoded training design of a batch fit and its decoders |
| `transform/chunks.py`, `chunk_cache.py` | the encoded training set as chunks for the streaming head: the sizing probe, the counting pass, the RAM or disk cache, the feed that decodes a chunk |
| `classification/gram.py`, `lsqr.py`, `flat_design.py` | the batch ridge heads |
| `classification/stream.py` | the streaming head: class weights, the SGD scalars, the epochs |
| `classification/prior.py`, `weights.py`, `common.py` | the prior shift, the class weights, targets and decisions |

Three private methods of `FERRETS` are the whole contract between the two
classes. They are private because they serve the classifier, not users:

| Method | Returns | Used by |
| --- | --- | --- |
| `_fit_design(X, stats, keep_all, ram_mb, metadata)` | the encoded training design (`transform/design.py`) | `FERRETS.fit` drops it, `fit_transform` builds its rows from it, the batch heads learn from it |
| `_fit_chunks(X, stats, metadata)` | the training set as cached chunks (`transform/chunks.py`) | the streaming head runs its epochs over them |
| `_score(X, coef)` | `transform(X) @ coef` without assembling the rows | `predict` and `decision_function` |

Both fit methods fit the transform (dictionaries, cuts, counting pass, kept
columns, idf) and return an object that is a context manager: leaving its
block removes the fit-time tables (and the chunk cache), and, when the block
raised nothing and `metadata` was given, marks the transformer fitted. The
heads read what they need from that object (the `_Design` and `_Chunks`
class docstrings list what each exposes) and from the fitted public
attributes; they touch no private attribute of the transformer. The seam test
(`test_the_heads_touch_the_transformer_only_through_the_seam`) allows two
exceptions: the classifier validates the transform parameters with
`_check_params` before it holds the transformer, and the ablation configures
and reads its own `AblationTransformer` subclass.

## Threads and reproducibility

- `n_jobs` is the thread count of `fit`, `transform` and `predict`: 1 by
  default, `None` is 1, `-1` is every numba thread, larger values are capped at
  numba's thread count. numba and BLAS share this budget for the length of a
  call (BLAS through scikit-learn's thread-pool controller;
  `fit_stats_["blas_limited"]` says whether it applied).
- The streaming pass needs numba's OpenMP layer. FERRETS selects it at its
  first parallel call when no layer has been chosen yet. Under another layer,
  for instance tbb, the pass runs on one thread and computes the same values
  (`fit_stats_["team_threads"]` is the team it got). The package leaves the
  process environment alone.
- `random_state` fixes the draws, the cut samples and the streaming order. An
  int is reduced modulo 2^32; `None` or a `RandomState` supplies a seed at
  `fit`, kept in `seed_` (`clf.transformer_.seed_` for a classifier).
- The dictionaries, cuts, kept columns and keys depend on the seed and the data
  only. The streaming head's result does not depend on the thread count or the
  cache mode. The batch head's BLAS products can differ at rounding level
  between thread counts.
- numba compiles the kernels at first use and caches them on disk under the
  package. The first fit in a fresh install is slower, and the cache is rebuilt
  when a kernel file changes.

## Research switches (`ferrets.ablation`)

`FERRETSAblation` is a `FERRETSClassifier` with four more switches, outside
the public API, for ablation studies. At the defaults it is
`FERRETSClassifier`.

| Parameter | Default | Effect |
| --- | --- | --- |
| `head` | `"default"` | `"logistic"`: an sklearn SGD logistic head on the batch design (batch only) |
| `prune_frac` | 1.0 | keep that fraction of views per pseudo-channel (per side in the family layout), ranked by the norm of their head weights, and refit |
| `ridge_cv` | None | k-fold cross-validated alpha on the LSQR route at every n (batch only) |
| `word_bits` | 8 | 4 or 2 bits per PAA and Fourier word; needs the univariate kernel (univariate input or the per-channel layout) |

The logistic head and `ridge_cv` have their own solvers and refuse class
weights: pass `batch_class_weight=None`.

## Decoding (`ferrets.decode`)

`FERRETSDecoder(model)` reads a fitted `FERRETSClassifier` column by column.
`decode(head_col)` returns the word behind a head column (view, dilation,
8-bit pattern, window and span, source, channel; for a level word its mean and
std bin ranges). `encode(word)` returns the word's global column (the inverse
of `decode_global`; `head_column(g)` maps it to a head column, or None if
pruned). `positions(x, word)` and
`contains(X, word)` locate a word in series with a numpy scan that follows the
kernel's arithmetic; `exact_contains` reads the model's own counts. The decoder
reads univariate models and the per-channel layout, and refuses family models.

## Limits

- Series have equal length; aeon's input check rejects unequal-length
  collections. A series needs at least 8 time points; `windows` longer than the
  series are skipped.
- Missing and infinite values are rejected. Small values are not: aeon's
  variance floor (it refuses a channel that varies with a standard deviation of
  at most 1e-7) is not applied, and a window with standard deviation at most
  1e-8 is keyed as flat.
- `y` is a numpy array or a pandas Series; `X` is `(n, m)` or `(n, C, m)`.
  Prediction needs the fitted length and channel count.
- There is no channel limit: a family keys at most eight channel words.
