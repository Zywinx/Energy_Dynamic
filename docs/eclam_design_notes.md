# ECLAM-SB design notes

## Baseline

The baseline is CLAM-SB with instance-level clustering on the HZ-EY Stage-1 A-only/A-prioritized 99-case benign/malignant parotid WSI cohort.

The first-stage comparison must align with the existing baseline setting: same 99 cases, same repeated patient-level stratified splits, same ResNet50 features, same seed, same max_epochs=50, and validation-best checkpoint selection.

## Why reuse existing `.pt` features

This ECLAM branch changes the slide-level MIL head/loss scheduling and uncertainty export only. It does not evaluate a new encoder. Therefore it reuses the same ResNet50 features to isolate the contribution of dynamic instance supervision and energy-based uncertainty analysis.

## Why not regenerate splits

The goal is a fair paired comparison against the reproduced CLAM-SB baseline. Regenerating splits would confound method differences with cohort sampling differences. The correct description is five repeated patient-level stratified splits, not strict mutually exclusive 5-fold cross-validation.

## Energy formula

For slide-level logits `f(S)`, free energy is:

```text
E(S) = -T * logsumexp(f_c(S) / T)
```

Energy is computed from slide-level logits and is used for post-hoc reliability analysis and active-learning candidate ranking.

## Dynamic instance loss

Original CLAM:

```text
L = bag_weight * L_slide + (1 - bag_weight) * L_inst
```

ECLAM:

```text
lambda_max = (1 - bag_weight) / bag_weight
lambda_t = schedule_t * lambda_max
L = bag_weight * (L_slide + lambda_t * L_inst)
```

Constant mode has `schedule_t=1`, so it exactly recovers the original CLAM loss scale.

## Why no unknown three-class training

The current cohort only has slide-level benign/malignant labels. It has no true unknown slide labels and no patch-level unknown labels. Therefore the current method is not open-set classification and must not report unknown-class accuracy.

## Low-attention patches are not unknown

Attention reflects the relative contribution of patches within the current WSI for the slide-level diagnosis. Low attention can be described as weak negative evidence or non-diagnostic candidates, but it is not a supervised unknown class.

## Current publishable positioning

This method can be described as Energy-aware CLAM-SB with reliability-aware dynamic instance supervision. It keeps CLAM-SB weakly supervised slide classification and instance clustering, adds slide-level energy-based uncertainty scoring, and reduces early noisy pseudo-label influence with dynamic instance-loss weighting.

## Heatmap comparison boundary

Do not compare heatmaps in the first-stage method table unless baseline and ECLAM attention maps are exported with the same script and the same visualization protocol.
