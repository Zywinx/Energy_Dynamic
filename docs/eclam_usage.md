# ECLAM-SB usage for HZ-EY Stage-1 A-only BM

## Scope

This branch adds Energy-aware CLAM-SB with reliability-aware dynamic instance supervision.
It does not modify dataset CSVs, splits, features, or the original `models/model_clam.py`.

Recommended directory separation:

```text
/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM        # original CLAM-SB baseline
/data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM  # ECLAM experimental branch
```

The baseline comparison in this stage uses the already reproduced 99-case Stage-1 A-only/A-prioritized cohort:

- task: `task_hzey_stage1_aonly_bm`
- features: `/data/xuewz/WSI_PRE/CLAM_0423/data/features/hzey_sdpc_resnet50/pt_files`
- repeated patient-level stratified splits: `splits/task_hzey_stage1_aonly_bm_100`
- max epochs: 50
- seed: 1
- early stopping enabled
- checkpoint: validation-best checkpoint, not fixed final epoch

Do not describe this as strict mutually exclusive 5-fold cross-validation. Use: five repeated patient-level stratified splits.

## Dynamic loss definition

Original CLAM:

```text
total_loss = bag_weight * bag_loss + (1 - bag_weight) * instance_loss
```

ECLAM defines `lambda_t` as the instance-loss weight relative to bag loss:

```text
lambda_max = (1 - bag_weight) / bag_weight
lambda_t = schedule_t * lambda_max
total_loss = bag_weight * (bag_loss + lambda_t * instance_loss)
```

With `bag_weight=0.7`, `lambda_max=0.3/0.7=0.4286`. In constant mode this exactly recovers the original CLAM loss scale.

## Baseline smoke test

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 1 \
  --k_start 0 \
  --k_end 1 \
  --max_epochs 1 \
  --exp_code SMOKE_hzey_stage1_aonly_bm_clam_sb_r50 \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_hzey_stage1_aonly_bm \
  --model_type clam_sb \
  --log_data \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --embed_dim 1024 \
  --seed 1
```

## ECLAM constant smoke test

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 1 \
  --k_start 0 \
  --k_end 1 \
  --max_epochs 1 \
  --exp_code SMOKE_hzey_stage1_aonly_bm_eclam_sb_constant_check \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_hzey_stage1_aonly_bm \
  --model_type eclam_sb \
  --dynamic_inst_weight constant \
  --energy_enable \
  --energy_temperature 1.0 \
  --log_data \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --embed_dim 1024 \
  --seed 1
```

## Formal ECLAM constant check

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 5 \
  --max_epochs 50 \
  --exp_code hzey_stage1_aonly_bm_eclam_sb_constant_check \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_hzey_stage1_aonly_bm \
  --model_type eclam_sb \
  --dynamic_inst_weight constant \
  --energy_enable \
  --energy_temperature 1.0 \
  --log_data \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --embed_dim 1024 \
  --seed 1 \
  2>&1 | tee /data/xuewz/WSI_PRE/CLAM_0423/logs/hzey_stage1_aonly_eclam_constant_check.log
```

## Formal ECLAM linear warmup

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 5 \
  --max_epochs 50 \
  --exp_code hzey_stage1_aonly_bm_eclam_sb_linear_warmup \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_hzey_stage1_aonly_bm \
  --model_type eclam_sb \
  --dynamic_inst_weight linear_warmup \
  --inst_lambda_max 0.43 \
  --inst_warmup_epochs 10 \
  --energy_enable \
  --energy_temperature 1.0 \
  --log_data \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --embed_dim 1024 \
  --seed 1 \
  2>&1 | tee /data/xuewz/WSI_PRE/CLAM_0423/logs/hzey_stage1_aonly_eclam_linear_warmup.log
```

## Formal ECLAM sigmoid warmup

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

python main.py \
  --drop_out 0.25 \
  --early_stopping \
  --lr 2e-4 \
  --k 5 \
  --max_epochs 50 \
  --exp_code hzey_stage1_aonly_bm_eclam_sb_sigmoid_warmup \
  --weighted_sample \
  --bag_loss ce \
  --inst_loss svm \
  --task task_hzey_stage1_aonly_bm \
  --model_type eclam_sb \
  --dynamic_inst_weight sigmoid_warmup \
  --inst_lambda_max 0.43 \
  --inst_warmup_epochs 10 \
  --inst_warmup_gamma 10.0 \
  --energy_enable \
  --energy_temperature 1.0 \
  --log_data \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --embed_dim 1024 \
  --seed 1 \
  2>&1 | tee /data/xuewz/WSI_PRE/CLAM_0423/logs/hzey_stage1_aonly_eclam_sigmoid_warmup.log
```

## Export energy scores, fold 0 example

ECLAM:

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0

python tools/export_energy_scores.py \
  --task task_hzey_stage1_aonly_bm \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --split_dir splits/task_hzey_stage1_aonly_bm_100 \
  --ckpt_path results/hzey_stage1_aonly_bm_eclam_sb_linear_warmup_s1/s_0_checkpoint.pt \
  --model_type eclam_sb \
  --temperature 1.0 \
  --save_dir eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy \
  --fold 0 \
  --embed_dim 1024 \
  --drop_out 0.25
```

Baseline CLAM-SB post-hoc energy:

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
conda activate clam_latest
export CUDA_VISIBLE_DEVICES=0

python tools/export_energy_scores.py \
  --task task_hzey_stage1_aonly_bm \
  --data_root_dir /data/xuewz/WSI_PRE/CLAM_0423/data/features \
  --split_dir splits/task_hzey_stage1_aonly_bm_100 \
  --ckpt_path ../CLAM/results/hzey_stage1_aonly_bm_clam_sb_r50_s1/s_0_checkpoint.pt \
  --model_type clam_sb \
  --temperature 1.0 \
  --save_dir eval_results/EVAL_hzey_stage1_aonly_bm_clam_sb_r50_energy \
  --fold 0 \
  --embed_dim 1024 \
  --drop_out 0.25
```

Repeat fold 0-4 by changing `--fold` and checkpoint path.

## Collect binary metrics

```bash
cd /data/xuewz/WSI_PRE/CLAM_0423/code/CLAM_ECLAM
python tools/collect_binary_metrics.py \
  --input_dir eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy \
  --glob 'energy_scores_fold_*.csv' \
  --which_split test \
  --out_dir eval_results/EVAL_hzey_stage1_aonly_bm_eclam_sb_linear_warmup_energy/metrics_test
```
