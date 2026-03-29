# Paper Sweep Lab

Config-driven sweep framework for Phase 2 experiments.

## Supported Variants

- `mae_multitask`
  Pretrained Phase 1 MAE encoder with multitask position/velocity decoding.

- `mae_context`
  Pretrained Phase 1 MAE encoder with session-context conditioning and partial fine-tuning.

- `transformer_multitask`
  Scratch multitask temporal Transformer.

- `spint_like`
  Paper-inspired channel-set encoder: per-time-step channel tokens are pooled into
  temporal features before temporal decoding. This is a lightweight approximation of
  permutation-invariant neural decoding ideas.

## Starter Configs

- `mae_mt_p1best_vw4.json`
- `mae_mt_m2_vw4.json`
- `mae_ctx_p1_uf2.json`
- `mae_ctx_m2_uf4.json`
- `txf_mt_base.json`
- `txf_mt_longctx.json`
- `spint_small.json`
- `spint_wide.json`

These sweep both:
- structure family
- pretrained checkpoint choice
- sequence length
- loss weighting
- partial-fine-tuning depth

## Usage

List configs:

```bash
cd /scratch/ml8347/neuroinformatics/project2/phase2/paper_sweep_lab
python sweep.py --list
```

Run one config locally:

```bash
python sweep.py --run mae_mt_p1best_vw4
```

Quick sanity check:

```bash
python sweep.py --run spint_small --quick
```

Run full SLURM array:

```bash
sbatch sweep.sbatch
```

## Outputs

Each config writes to:

```text
paper_sweep_lab/runs/<config_name>/
```

with:
- `best_model.pt`
- `last_model.pt`
- `config.json`
- `submission.csv`

## Notes

- The trainer logs both raw and weighted loss components so it is easier to see
  whether velocity loss is actually affecting optimization.
- For pretrained variants, you can duplicate a config and change
  `pretrained_checkpoint` to test other `.pt` files.
