"""Runtime settings, read as ``config.NAME`` when used: set a value on this
module to change it. The kernels get them as arguments. Sizes in MB are
10^6 bytes."""


# --- Cut sampling (read at fit) ---------------------------------------------

# windows sampled per Fourier cut job, shared by the job's dilations
FOURIER_MAX_WINDOWS = 20_000
# replayed draws held per sampler batch
SAMPLER_BATCH_MB = 64.0
# samples per t-major block of the level cut sampler's sums (a schedule: every
# block size gives the same floats)
LEVEL_SAMPLER_BLOCK = 64


# --- Keying (read at every call) --------------------------------------------

# Predict only: rows per tile of the keying kernels. 1 keys row by row, below 1
# acts as 1. The fit keys one row at a time.
PREDICT_ROW_TILE = 16
# Chain-major keying: key scratch of one row block (it also sets predict's row
# tile, at most PREDICT_ROW_TILE) and the result store, in MB.
KEY_SCRATCH_MB = 2.0
RESULT_STORE_MB = 4.0
# MB of the table of shared one-byte channel words of a block, and the budget
# that picks which words the plan shares (WORD_TABLE_MB / m). 0 shares nothing.
WORD_TABLE_MB = 1.0


# --- Transform buffers ------------------------------------------------------

TRANSFORM_CHUNK_MB = 256.0  # id and count buffers of one transform chunk
# (row bound x row) slots per slice when keys are mapped to head columns
TRANSFORM_SLICE_SLOTS = 1 << 20
ROW_PARTS = 40            # row partitions of the parallel reductions (fixed)


# --- Batch head: Gram route -------------------------------------------------

DUAL_MAX_N = 3000         # Gram/GCV route up to this many training series
PRIOR_LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # prior-shift grid (both routes)
GRAM_SLAB_MB = 256.0      # widest dense Gram slab
GRAM_PACK_MB = 64.0       # narrow blocks share a slab only up to this size


# --- Batch head: LSQR route -------------------------------------------------

RIDGE_TOL = 1e-2          # LSQR tolerance
RIDGE_MAX_ITER = 200      # LSQR iteration cap
HOLDOUT = 0.25            # share of rows held out to pick alpha
LSQR_GROUP = 8            # classes per lockstep Golub-Kahan group
LSQR_STATE_MB = 1024.0    # budget for a group's Krylov bases
LSQR_K_GUESS = 16         # iterations assumed when sizing a group
RMV_PARTS = 8             # row partitions of the transposed product
RMV_BAND = 65536          # ... over bands of this many columns
# The transposed product goes by column ranges when a row partition's scratch
# (band columns x padded column count x 8 bytes) would pass this many MB.
RMV_SCRATCH_MB = 2.0
# The flat design cuts every pseudo-channel block wider than
# 1 << FLAT_BAND_BITS columns into sub-blocks (uint16 ids); above 16 it keeps
# one block each.
FLAT_BAND_BITS = 16


# --- Streaming head ---------------------------------------------------------

ASGD_PARTS = 40           # max entry partitions of a row's dot products
SQ_ETA0 = 1.0             # step size
SQ_EPOCHS = 20            # passes over the chunks
SQ_ALPHA = 1e-5           # l2 penalty
INTERCEPT_DECAY = 0.01    # sklearn's sparse intercept decay
MAX_DLOSS = 1e12          # bound on |loss derivative| of one update
# The lazy l2 scale is folded into the float32 weights once it falls below
# this.
WSCALE_RESET = 1e-2
STREAM_CHUNK_NNZ = 2.5e7  # entries per chunk, counted before pruning
STREAM_PROBE_ROWS = 64    # rows transformed to size the chunks
STREAM_RAM_MB = 256.0     # a cache up to this size stays in RAM
STREAM_DISK_FRAC = 0.5    # largest share of the free disk the cache may take
STREAM_EST_BYTES = 2.4    # cache bytes per nonzero (pre-flight estimate)


# --- Test switches ----------------------------------------------------------
# The tests use the off paths as oracles; the values are the same either way.

CHAIN_KEYING = True       # chain-major keying of the univariate kernel
DEDUP_FOURIER_JOBS = True     # a repeated Fourier cut job runs once
FUSE_LEVEL_CUTS = True    # a level job reuses a matching Fourier job's pass
