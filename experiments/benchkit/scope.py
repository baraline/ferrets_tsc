"""What the queue runs, in what order, and which cells the gate may prune.

``STEPS`` lists every cell whose row the release holds, and nothing else:
``bench.py queue --target all`` runs them in this order, skipping the
(dataset, arm, fold) keys already banked, so a dry run on the released
results plans nothing, deleting rows plans exactly those cells again, and
relaunching after a pause resumes the run.

- UCR: FERRETS and BORF on the 112 datasets, resamples 0-29 (0 is the
  published split, i >= 1 aeon's stratified resample seeded by i); the
  streaming head with and without class weights on the ablation's 92
  datasets, resamples 1-9.
- Multiverse: FERRETS, MiniRocket, Hydra and BORF on the 133 problems at
  the published split; FERRETS on 76 problems at resamples 1-2; the
  per-channel layout on the 115 problems MONSTER does not carry (resample 0)
  and on the 76 (resamples 1-2). The 18 problems MONSTER also carries run
  last, after their data is cached, and every cell of them is gated.
- MONSTER: the streaming head and MiniRocket on the 29 problems, folds 0-4;
  the batch head on the nine problems whose design fits 8 GB
  (MONSTER_BATCH_PROBLEMS), folds 0-4; the unweighted streaming head, fold 0.
- Cost: the synthetic grid and the six anchors, three repeats each.
- The rival timings: FERRETS, MiniRocket, Hydra, QUANT, MultiRocket-Hydra
  and RDST on UCR resample 10, written to results/rivals_ucr.csv.
"""

import functools
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .arms import is_slow
from .data import datasets, multiverse_names, multiverse_overlap
from .parent import bank_path, plan_cells

REPO_ROOT = Path(__file__).resolve().parents[2]

# The batch head also runs at MONSTER, on the problems whose LSQR design fits
# 8 GB (chosen by design size alone), five folds like every MONSTER arm.
MONSTER_BATCH_ARM = "FERRETS"
MONSTER_BATCH_PROBLEMS = (
    "FordChallenge", "LakeIce", "Opportunity", "Pedestrian", "Skoda",
    "Tiselac", "Traffic", "UCIActivity", "WISDM")
# The MONSTER problems of the ablation (experiments/ablation.py --datasets):
# the 18 the Multiverse also carries.
MONSTER_HELDOUT_PROBLEMS = (
    "WISDM", "Traffic", "Pedestrian", "FordChallenge", "LakeIce", "UCIActivity",
    "Opportunity", "InsectSound", "CrowdSourced", "Skoda",
    "CornellWhaleChallenge", "AudioMNIST-DS", "FruitFlies", "Tiselac", "STEW",
    "USCActivity", "WISDM2", "PAMAP2")
# the ablation's 92 UCR datasets (the 112 minus the 20 of the memory runs)
ABLATION_UCR = (
    'Chinatown', 'SonyAIBORobotSurface1', 'ItalyPowerDemand', 'MoteStrain',
    'SonyAIBORobotSurface2', 'TwoLeadECG', 'SmoothSubspace', 'ECGFiveDays',
    'BME', 'CBF', 'UMD', 'DiatomSizeReduction', 'Coffee', 'FaceFour',
    'FreezerSmallTrain', 'ArrowHead', 'ECG200', 'Symbols', 'ShapeletSim',
    'InsectEPGSmallTrain', 'BeetleFly', 'BirdChicken', 'ToeSegmentation1',
    'ToeSegmentation2', 'Wine', 'Beef', 'Plane', 'OliveOil',
    'SyntheticControl', 'GunPointAgeSpan', 'GunPointMaleVersusFemale',
    'GunPointOldVersusYoung', 'Lightning7', 'PowerCons', 'FacesUCR', 'Meat',
    'Trace', 'MiddlePhalanxTW', 'DistalPhalanxOutlineAgeGroup',
    'DistalPhalanxTW', 'MiddlePhalanxOutlineAgeGroup',
    'ProximalPhalanxOutlineAgeGroup', 'ProximalPhalanxTW', 'Herring', 'Car',
    'InsectEPGRegularTrain', 'MedicalImages', 'Lightning2',
    'FreezerRegularTrain', 'Ham', 'DistalPhalanxOutlineCorrect',
    'MiddlePhalanxOutlineCorrect', 'ProximalPhalanxOutlineCorrect',
    'InsectWingbeatSound', 'WordSynonyms', 'FaceAll', 'Fish', 'OSULeaf',
    'MixedShapesSmallTrain', 'CricketY', 'CricketZ',
    'PhalangesOutlinesCorrect', 'Strawberry', 'ACSF1', 'Worms',
    'WormsTwoClass', 'Earthquakes', 'Haptics', 'Computers', 'InlineSkate',
    'PigAirwayPressure', 'PigArtPressure', 'PigCVP',
    'LargeKitchenAppliances', 'RefrigerationDevices', 'ScreenType',
    'SmallKitchenAppliances', 'UWaveGestureLibraryX',
    'UWaveGestureLibraryY', 'UWaveGestureLibraryZ', 'ShapesAll', 'Crop',
    'EOGHorizontalSignal', 'EOGVerticalSignal', 'SemgHandMovementCh2',
    'SemgHandSubjectCh2', 'ElectricDevices', 'EthanolLevel',
    'NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax2', 'FordB',
    'HandOutlines')
# the 76 Multiverse problems FERRETS also ran at resamples 1-2
MV_RESAMPLED = (
    'Alzheimers', 'AppliancesEnergy_disc', 'ArticularyWordRecognition',
    'BeijingPM10Quality_disc', 'BeijingPM25Quality_disc', 'ButtonPress',
    'DuckDuckGeese', 'EmoPain', 'EyesOpenShut', 'FaceDetection',
    'FeedbackButton', 'FeetHands', 'FingerMovements',
    'HandMovementDirection', 'Heartbeat', 'IRDS-EFL', 'IRDS-EFR',
    'IRDS-SAL', 'IRDS-SAR', 'IRDS-SFE', 'IRDS-SFL', 'IRDS-SFR', 'IRDS-STL',
    'IRDS-STR', 'ImaginedFeetHands', 'ImaginedOpenCloseFist', 'InnerSpeech',
    'JapaneseVowels', 'KERAAL-CTK', 'KERAAL-CTK-MC', 'KERAAL-ELK',
    'KERAAL-ELK-MC', 'KERAAL-RTK', 'KERAAL-RTK-MC', 'KIMORE-LA-C',
    'KIMORE-LT-C', 'KIMORE-PR-C', 'KIMORE-Sq-C', 'KIMORE-TR-C',
    'KINECAL-3WFV', 'KINECAL-GGFV', 'KINECAL-QSEC', 'KINECAL-QSEO',
    'Locust2022', 'LongIntervalTask', 'LowCost', 'MatchingPennies',
    'MindReading', 'MotionSenseHAR', 'MotorImagery', 'NATOPS',
    'OpenCloseFist', 'PEMS-SF', 'PhonemeSpectra', 'PhotoStimulation',
    'PronouncedSpeech', 'SPHERE-WUS', 'ShortIntervalTask', 'SitStand',
    'SongFamiliarity', 'SpokenArabicDigits', 'UCDHE-MP', 'UCDHE-MP-MC',
    'UCDHE-Rowing', 'UCDHE-Rowing-MC', 'UIPRMD-DS-C', 'UIPRMD-HS-C',
    'UIPRMD-IL-C', 'UIPRMD-SASLR-C', 'UIPRMD-SL-C', 'UIPRMD-SSA-C',
    'UIPRMD-SSE-C', 'UIPRMD-SSIER-C', 'UIPRMD-SSS-C', 'UIPRMD-STS-C',
    'VisualSpeech')
# the channel axis of the cost grid and its two multichannel anchors
COST_CHANNEL_AXIS = (
    "syn_n1000_m256_c4", "syn_n1000_m256_c16", "syn_n1000_m256_c64",
    "syn_n1000_m256_c256", "multiverse:BasicMotions", "multiverse:PEMS-SF")
COST_ARMS = ("FERRETS", "FERRETS_stream", "MiniRocket", "QUANT", "Hydra",
             "BORF", "MrSQM", "WEASEL2", "MUSE_LR", "cBOSS_c50", "TDE_c50")
RIVALS = ("FERRETS", "MiniRocket", "Hydra", "QUANT", "MultiRocketHydra", "RDST")
RIVALS_OUT = "results/rivals_ucr.csv"
# arms that run on a named subset of a benchmark whoever launches them:
# {benchmark: {arm: datasets}}
RESTRICTED = {"monster": {MONSTER_BATCH_ARM: MONSTER_BATCH_PROBLEMS}}
TARGETS = ("all",)


@dataclass(frozen=True)
class Step:
    kind: str                     # "run" | "prefetch"
    benchmark: str = ""
    arms: tuple = ()
    folds: tuple = ()
    # None: every dataset of the benchmark; else these, in the benchmark's
    # order (a prefetch step caches these)
    datasets: tuple = None
    out: str = None               # None: results/bench_<benchmark>.csv
    label: str = ""

    def names(self):
        names = datasets(self.benchmark)
        if self.datasets is None:
            return list(names)
        keep = set(self.datasets)
        return [d for d in names if d in keep]


@functools.lru_cache(maxsize=None)
def _overlap():
    return frozenset(multiverse_overlap())


def _rest():
    return tuple(d for d in multiverse_names() if d not in _overlap())


def _steps():
    rest, over = _rest(), tuple(multiverse_overlap())
    mv = ("FERRETS", "MiniRocket", "Hydra", "BORF")
    return (
        Step("run", "ucr", ("FERRETS", "BORF"), tuple(range(30)),
             label="UCR, FERRETS and BORF, resamples 0-29"),
        Step("run", "ucr", ("FERRETS_stream", "FERRETS_stream_unweighted"),
             tuple(range(1, 10)), ABLATION_UCR,
             label="UCR, the streaming head on the ablation's 92 datasets"),
        Step("prefetch", "multiverse", datasets=rest,
             label="Multiverse: cache the 115 problems MONSTER does not carry"),
        Step("run", "multiverse", mv, (0,), rest,
             label="Multiverse, the 115, published split"),
        Step("run", "multiverse", ("FERRETS",), (1, 2), MV_RESAMPLED,
             label="Multiverse, FERRETS at resamples 1-2"),
        Step("run", "multiverse", ("FERRETS_perchannel",), (0,), rest,
             label="Multiverse, the per-channel layout, published split"),
        Step("run", "multiverse", ("FERRETS_perchannel",), (1, 2), MV_RESAMPLED,
             label="Multiverse, the per-channel layout, resamples 1-2"),
        Step("run", "monster", ("FERRETS_stream", "MiniRocket"), (0, 1, 2, 3, 4),
             label="MONSTER, folds 0-4 (gated)"),
        Step("run", "monster", (MONSTER_BATCH_ARM,), (0, 1, 2, 3, 4),
             MONSTER_BATCH_PROBLEMS, label="MONSTER, the batch head (gated)"),
        Step("run", "monster", ("FERRETS_stream_unweighted",), (0,),
             label="MONSTER, the unweighted streaming head (gated)"),
        Step("run", "cost", COST_ARMS, (0, 1, 2),
             label="cost benchmark (gated)"),
        Step("run", "cost", ("FERRETS_perchannel", "FERRETS_stream_perchannel"),
             (0, 1, 2), COST_CHANNEL_AXIS,
             label="cost benchmark, the per-channel layout (gated)"),
        Step("prefetch", "multiverse", datasets=over,
             label="Multiverse: cache the 18 problems MONSTER also carries"),
        Step("run", "multiverse", mv, (0,), over,
             label="Multiverse, the 18 MONSTER also carries (gated)"),
        Step("run", "ucr", RIVALS, (10,), out=RIVALS_OUT,
             label="UCR resample 10, the rival timings"),
    )


def steps(target="all"):
    if target != "all":
        raise ValueError(f"unknown target {target!r}; one of {TARGETS}")
    return list(_steps())


def always_gated(benchmark, dataset):
    """Cells the gate decides whoever launches them: a Multiverse cell of a
    problem MONSTER also carries (MONSTER-scale data on the batch loader)."""
    return benchmark == "multiverse" and dataset in _overlap()


def gated(benchmark, arm, dataset=None):
    """The feasibility gate applies to every slow arm, every MONSTER cell,
    every cost cell and every ``always_gated`` cell; fast arms elsewhere
    always run."""
    return (benchmark in ("monster", "cost") or is_slow(arm)
            or (dataset is not None and always_gated(benchmark, dataset)))


def runnable_by_gate(predictor=None):
    """A predicate ``cell -> bool`` for the previews (``bench.py queue
    --dry-run``, disk.project) and the cache reaper (``still_needed``), so
    both count the same cells: False for a cell the feasibility gate would
    refuse. One Predictor serves every call, built on the first gated cell
    (or ``predictor``); each (benchmark, arm, dataset) is decided once."""
    state = {"predictor": predictor}
    seen = {}

    def get():
        if state["predictor"] is None:
            from .gate import Predictor
            state["predictor"] = Predictor()
        return state["predictor"]

    def runnable(cell):
        if not gated(cell.benchmark, cell.arm, cell.dataset):
            return True
        key = (cell.benchmark, cell.arm, cell.dataset)
        if key not in seen:
            seen[key] = get().decide(
                cell.benchmark, cell.arm, cell.dataset,
                conservative=always_gated(cell.benchmark, cell.dataset)).run
        return seen[key]
    runnable.predictor = get
    return runnable


def step_path(step, results_dir=None):
    if step.out:
        root = Path(results_dir) if results_dir else REPO_ROOT / "results"
        return root / Path(step.out).name
    return bank_path(step.benchmark, results_dir)


def _bank_keys(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if not len(df):
        return set()
    return set(zip(df["dataset"], df["model"], df["fold"].astype(int)))


def step_cells(step, results_dir=None):
    """The unbanked cells of one run step, dataset-major."""
    names = step.names()
    folds = {a: tuple(step.folds) for a in step.arms}
    cells = plan_cells(step.benchmark, list(step.arms), names, folds,
                       _bank_keys(step_path(step, results_dir)))
    if step.benchmark == "cost":
        from .cost import filter_cells
        cells = filter_cells(cells)
    order = {d: i for i, d in enumerate(names)}
    cells.sort(key=lambda c: (order[c.dataset], step.arms.index(c.arm)))
    return cells


def step_keys():
    """(out, benchmark, dataset, arm, fold) of every cell of every run step,
    banked or not: what the release's result files hold. ``out`` is "" for
    a step writing results/bench_<benchmark>.csv."""
    from .cost import admissible
    keys = []
    for step in _steps():
        if step.kind != "run":
            continue
        for d in step.names():
            for a in step.arms:
                if step.benchmark == "cost" and not admissible(a, d):
                    continue
                for f in step.folds:
                    keys.append((step.out or "", step.benchmark, d, a, f))
    return keys


def needed_cells(results_dir=None, runnable=None):
    """{(benchmark, dataset): [the UNBANKED cells that read it]}, anywhere in
    the plan. A cost anchor ``<benchmark>:<dataset>`` counts for the
    benchmark it borrows from too. ``runnable`` (``runnable_by_gate``): a
    cell it refuses is left out (the gate banks it "not run" when its step
    comes, so it reads no data)."""
    need = {}
    for step in _steps():
        if step.kind != "run":
            continue
        for cell in step_cells(step, results_dir):
            if runnable is not None and not runnable(cell):
                continue
            need.setdefault((cell.benchmark, cell.dataset), []).append(cell)
            if cell.benchmark == "cost" and ":" in cell.dataset:
                inner, name = cell.dataset.split(":", 1)
                need.setdefault((inner, name), []).append(cell)
    return need


def still_needed(results_dir=None, runnable=None):
    """{(benchmark, dataset)} that at least one UNBANKED cell still reads:
    what the cache reaper must not delete."""
    return set(needed_cells(results_dir, runnable))
