# Results tables

The results tables of the paper "FERRETS: A Scalable Symbolic Transform for Time Series Classification", one CSV
file per table (its Appendix B lists them, and the paper links each file). The values are rounded as the paper
quotes them, and the paper's number scripts check each one against the result rows; the per-problem values are in
`results/per_dataset/`.

Conventions:

- `acc` and `bal` are accuracy and balanced accuracy. A UCR mean is over the 30 resamples, a MONSTER mean over the
  five folds, a Multiverse value the result at the published split.
- `delta` is FERRETS' score minus the other method's (in the ablation files: the changed model minus the shipped
  one), the mean over the problems of the row. `wins`, `losses` and `ties` count the problems on which FERRETS (in
  the ablation files: the changed model) scores higher, lower or within floating-point tolerance, which the
  Wilcoxon test drops as a zero difference.
- `p` is the two-sided Wilcoxon signed-rank test's p value unless the entry says otherwise, `p_holm` its Holm
  correction over the family the entry names. `ci_lo` and `ci_hi` bound the percentile bootstrap interval of the
  paired mean difference (10,000 draws).
- `sd` is the standard deviation, over the resamples (UCR) or folds (MONSTER), of the mean over the table's
  problems on one resample or fold.
- `from` is M for a result run here and A for the archive's published one; `common` is the mean over the problems
  every method run here completes, `own` over the method's own problems.
- `block` and `group` name the parts of a table; an empty cell is one the table leaves blank (the reference row, a
  test not run, a value not measured). The tables' bold marks are dropped, except in `domains.csv`, where a
  `<method>_holm_separates` column keeps them; a star becomes a `<method>_raw_p_below_005` column and a dagger the
  `below_coverage_rule` column (1 when marked).
- Memory: MB = 10^6 bytes, GiB = MB / 1073.74.

## overall.csv

FERRETS against MiniRocket and QUANT under each archive's protocol, on the problems all three complete (common). Done counts the problems with a result. The sd is taken over the 30 resamples (UCR) or five folds (MONSTER). The last block counts a problem shared by Multiverse and MONSTER once, with its MONSTER (streaming) result. The sd beside a mean is the standard deviation, over the resamples (UCR) or folds (MONSTER), of the mean over the table's problems on one resample or fold, rounded to its first nonzero digit; Multiverse has one run and no sd. The interval is uncorrected for the family, so it can include zero where p_Holm separates. Holm's family is the comparisons of one archive and one measure: two on UCR and at MONSTER (one for MONSTER's balanced accuracy), four on Multiverse, its whole archive and its core together. The last block is not tested.

Columns: `block`, `method`, `done`, `acc_mean`, `acc_sd`, `acc_delta`, `acc_ci_lo`, `acc_ci_hi`, `acc_wins`, `acc_losses`, `acc_ties`, `acc_p`, `acc_p_holm`, `bal_mean`, `bal_sd`, `bal_delta`, `bal_ci_lo`, `bal_ci_hi`, `bal_wins`, `bal_losses`, `bal_ties`, `bal_p`, `bal_p_holm`.

## family.csv

The dictionary methods and Hydra on UCR (Section 2.2), with Holm over these six comparisons, separately for accuracy and balanced accuracy. The sd is taken over the 30 resamples.

Columns: `method`, `from`, `acc`, `acc_sd`, `bal`, `bal_sd`, `acc_delta`, `acc_wins`, `acc_losses`, `acc_ties`, `acc_p`, `acc_p_holm`, `bal_delta`, `bal_p`, `bal_p_holm`.

## ucr-strata.csv

UCR by length m and training-set size n, with raw p. The last two columns give Spearman's rho of the difference with log m and log n (Deltaacc row) and its p (p row). Holm over the eight trend tests keeps the length trend against MiniRocket and WEASEL 2 alone (p_Holm = 0.013).

Columns: `method`, `statistic`, `all`, `m_le_150`, `m_151_700`, `m_gt_700`, `n_lt_50_m_le_700`, `n_ge_50_m_le_700`, `rho_log_m`, `rho_log_n`.

## domains.csv

FERRETS' accuracy minus each method's, in points, by application domain, for both archives and for each, with the problems n FERRETS completes and their median length m. The strata are descriptive. A star marks a raw p < 0.05, bold a pair that Holm over the 31 tested cells of Figure 17 separates. The domains are Multiverse's eight types, into which UCR's types fold. Trajectory joins Motion, ECG, EOG, EPG and Hemodynamics join Biosignal, Traffic joins Sensor, and Device and Power form one group with Multiverse's Power. UCR's Simulated type has no Multiverse counterpart, and Multiverse's Audio and Digital types, too small to stand alone, share a row. MONSTER is left out, since only MiniRocket and QUANT have results there and 18 of its 29 problems are Multiverse problems.

Columns: `domain`, `archive`, `n`, `median_m`, `MiniRocket`, `MiniRocket_raw_p_below_005`, `MiniRocket_holm_separates`, `QUANT`, `QUANT_raw_p_below_005`, `QUANT_holm_separates`, `MultiRocket-Hydra`, `MultiRocket-Hydra_raw_p_below_005`, `MultiRocket-Hydra_holm_separates`, `RDST`, `RDST_raw_p_below_005`, `RDST_holm_separates`, `HC2`, `HC2_raw_p_below_005`, `HC2_holm_separates`, `LITETime`, `LITETime_raw_p_below_005`, `LITETime_holm_separates`, `FreshPRINCE`, `FreshPRINCE_raw_p_below_005`, `FreshPRINCE_holm_separates`.

## mv-channels.csv

Multiverse by channel count C, on the 122 problems FERRETS, MiniRocket and QUANT complete, with W/L/T on accuracy. The last row is overall.csv's Multiverse block. The raw p of the accuracy difference to MiniRocket is 0.80 between C <= 8 and C > 8 (Mann-Whitney) and 0.13 on C > 8 alone (Wilcoxon).

Columns: `channels`, `problems`, `minirocket_delta_acc`, `minirocket_wins`, `minirocket_losses`, `minirocket_ties`, `minirocket_delta_bal`, `quant_delta_acc`, `quant_wins`, `quant_losses`, `quant_ties`, `quant_delta_bal`.

## monster-pairs.csv

FERRETS against each method at MONSTER where both complete (five-fold means), n the number of problems. Balanced accuracy covers MiniRocket and ConvTran only (Appendix B). Holm's family is the rows of one block (one measure).

Columns: `block`, `method`, `n`, `delta`, `ci_lo`, `ci_hi`, `wins`, `losses`, `ties`, `p`, `p_holm`.

## mv-core.csv

The 65 problems of the Multiverse core at the published split (common set of 55 problems), with Holm over the 17 comparisons of the table, separately for accuracy and balanced accuracy. Compl. counts the problems with a result, common and own are the means over the common set and over the method's own problems. mv-core.csv and mv-all.csv group the methods into those run here, the published estimators and, marked by a dagger, those below the coverage rule. The published block holds the estimators covering at least 119 of the 125 problems with any published result, and 1NN-DTW as a reference. The dagger block holds those covering down to 113, and WEASEL+MUSE; each is paired on its own problems with an uncorrected p outside the Holm family.

Columns: `group`, `method`, `completed`, `acc_common`, `acc_own`, `bal_common`, `bal_own`, `acc_delta`, `acc_wins`, `acc_losses`, `acc_ties`, `acc_p`, `acc_p_holm`, `bal_delta`, `bal_wins`, `bal_losses`, `bal_ties`, `bal_p`, `bal_p_holm`, `below_coverage_rule`.

## mv-all.csv

All 133 Multiverse problems at the published split, columns as in mv-core.csv, the common set being 102 problems, with Holm over the 17 comparisons of the table, separately for accuracy and balanced accuracy.

Columns: `group`, `method`, `completed`, `acc_common`, `acc_own`, `bal_common`, `bal_own`, `acc_delta`, `acc_wins`, `acc_losses`, `acc_ties`, `acc_p`, `acc_p_holm`, `bal_delta`, `bal_wins`, `bal_losses`, `bal_ties`, `bal_p`, `bal_p_holm`, `below_coverage_rule`.

## ablation-ucr.csv

Ablation on 92 UCR datasets, resamples 1 to 9, which contain the development third (Appendix A). The shipped model has accuracy 0.873 and balanced accuracy 0.860, with Delta the variant - shipped. Holm's family is every tested row of the file, accuracy and balanced accuracy together; the oracle row is not tested.

Columns: `block`, `change`, `acc_delta`, `acc_wins`, `acc_losses`, `acc_ties`, `acc_p`, `acc_p_holm`, `bal_delta`, `bal_wins`, `bal_losses`, `bal_ties`, `bal_p`, `bal_p_holm`, `macro_f1_delta`.

## ablation-mv-monster.csv

Ablation on Multiverse and MONSTER, with n the problems of the row and Delta the variant - shipped. The first block uses resamples 1 and 2 where C >= 9, else the published split. The problems contain the development third (Appendix A). Holm's family is every tested row of the file, its blocks pooled, accuracy and balanced accuracy together; the oracle rows are not tested.

Columns: `block`, `change`, `acc_delta`, `acc_wins`, `acc_losses`, `acc_ties`, `acc_p`, `acc_p_holm`, `bal_delta`, `bal_wins`, `bal_losses`, `bal_ties`, `bal_p`, `bal_p_holm`, `macro_f1_delta`.

## cost-choices.csv

Time ratio to the shipped model (below one is cheaper), on UCR over 92 datasets, Multiverse 57 and MONSTER 18 (streaming class weights 23). The last three rows are on the problems both complete, and a dash means not run.

Columns: `variant`, `ucr_fit`, `ucr_predict`, `mv_fit`, `mv_predict`, `monster_fit`, `monster_predict`.

## cost-scaling.csv

Log-log cost exponents over the finished grid points, median over the repeats, with mem the memory above the data, a dash too few grid points and c a contract stop. Section 5.2 reads FERRETS' n_fit, n_pred, m_fit, m_pred, c_fit, c_pred and c_mem, the n_mem of FERRETS' streaming fit, MiniRocket's n_fit and the per-channel layout's c_fit, c_pred and c_mem.

Columns: `block`, `method`, `n_fit`, `n_pred`, `n_mem`, `m_fit`, `m_pred`, `m_mem`, `c_fit`, `c_pred`, `c_mem`.

## cost-ucr.csv

Cost on UCR at resample ten. The cells column counts the datasets completed in the hour, s is their total time and x its ratio to FERRETS' total over all 112 cells, rounded to two leading digits, so TDE (89 cells) and cBOSS (110) are understated. Memory above the data and largest model are in MB, a dash where not sampled.

Columns: `method`, `cells`, `fit_s`, `fit_ratio`, `predict_s`, `predict_ratio`, `mem_above_data_median_mb`, `mem_above_data_max_mb`, `model_max_mb`.

## cost-mv.csv

Cost on the 133 Multiverse problems at the published split, cells counting the completed ones. Totals are over the 102 problems all four complete, memory and model over each method's own problems.

Columns: `method`, `cells`, `fit_total_s`, `predict_total_s`, `peak_mem_median_gib`, `peak_mem_max_gib`, `model_max_mb`.

## cost-monster.csv

Cost at MONSTER on the problems both complete (five-fold means), as time totals and median and maximum memory (GiB) and model (MB), FERRETS being the streaming fit.

Columns: `method`, `fit_total_s`, `predict_total_s`, `mem_above_data_median_gib`, `mem_above_data_max_gib`, `mem_total_median_gib`, `mem_total_max_gib`, `model_median_mb`, `model_max_mb`.

## ties.csv

Ties and key mismatches in the samples, with the datasets where each occurs and its largest share of windows.

Columns: `quantity`, `datasets`, `largest_share_pct`.
