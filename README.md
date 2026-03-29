# Project 2 Phase 2 — Intracortical Neural Activity Decoding

## Competition Overview

**Challenge:** Decode finger movement positions from intracortical SBP recordings that suffer from both neural drift (cross-session) and channel dropout (permanent per-session zeroing of ~28/96 channels).

**Task:** For each test session, predict `index_pos` and `mrp_pos` at every time bin. Positions are in [0, 1] (0=extended, 1=flexed).

**Metric:** Mean R² across all (session, position_channel) groups — 125 sessions × 2 channels = 250 groups. Higher is better; R²=0 means predicting the session mean.

**Split:** 187 train sessions, 125 test sessions. Chronological — test sessions contain the latest recordings (largest neural drift from training data).

**Baselines:**

| Model | R² |
|-------|----|
| Ridge Regression (Wiener filter) | -4.119 |
| LSTM (2-layer, hidden=128) | 0.199 |
| GRU (2-layer, hidden=128) | 0.213 |
| Transformer + SSL pretraining | 0.448 |

---

## Key Differences from Phase 1

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| Task | SBP reconstruction (self-supervised) | Finger position decoding (supervised regression) |
| Input | SBP with random per-timebin masking | SBP with permanent per-session channel zeroing |
| Output | Reconstructed SBP at masked entries | `index_pos`, `mrp_pos` at every time bin |
| Metric | NMSE ↓ | R² ↑ |
| Test sessions | 24 | 125 |
| Train sessions | 226 | 187 |
| Kinematics in test | Available (not masked) | **Not available** (the prediction target) |
| Submission rows | ~468K | ~2.99M |

**Phase 1 relevance:** The `mae_finetune` method directly uses pretrained Phase 1 MAE encoder weights. The SSL pretraining baseline at R²=0.448 confirms this is the most promising starting point.

---

## Directory Structure

```
phase2/
├── README.md                        ← this file
├── data_utils.py                    ← shared data loading, normalisation, evaluation
├── kaggle_data/
│   ├── train/     (187 sessions: _sbp.npy, _kinematics.npy, _trial_info.npz)
│   ├── test/      (125 sessions: _sbp.npy, _trial_info.npz)
│   ├── test_index.csv               ← 2,993,930 rows to predict
│   ├── sample_submission.csv        ← template
│   └── metric.py                   ← R² scorer
├── gru_baseline/                   ← Method 1: BiGRU decoder
├── mae_finetune/                   ← Method 2: Pretrained MAE encoder + decode head
├── transformer_decoder/            ← Method 3: Transformer from scratch
├── ensemble_decoder/               ← Method 4: Ensemble from multiple Phase 1 inits
├── mae_multitask_decoder/          ← Method 5: MAE + multitask position/velocity decoding
├── transformer_multitask_scratch/  ← Method 6: Scratch multitask transformer
└── mae_context_partial_ft/         ← Method 7: Context-conditioned partial fine-tune
```

Typical method directory contents:
```
<method>/
├── train.py          ← training + validation
├── submit.py         ← inference → submission.csv
├── <method>.sbatch   ← single SLURM job (train + submit)
├── sweep.sbatch      ← optional hyperparameter sweep (some methods only)
├── pick_best.sh      ← optional best-run helper (some methods only)
└── ...               ← checkpoints, submission CSVs, wandb logs
```

---

## Data Details

### Per-Session Files

| Array | Shape | Description |
|-------|-------|-------------|
| `*_sbp.npy` | (N_bins, 96) | SBP at 50 Hz, float32. ~28 channels permanently zeroed (channel dropout) |
| `*_kinematics.npy` | (N_bins, 4) | `[index_pos, mrp_pos, index_vel, mrp_vel]`, all in [0,1]. **Train only** |
| `*_trial_info.npz` | — | `start_bins`, `end_bins`, `n_trials` |

### Channel Dropout

Each session has ~28/96 channels permanently zeroed — different channels per session. This simulates real electrode degradation. The `dropout_ind` vector (96, float32) in `load_session()` is 1.0 for zeroed channels.

### Submission Format

Aligned with `test_index.csv` (columns: `sample_id`, `session_id`, `time_bin`):

```
sample_id,index_pos,mrp_pos
0,0.432,0.876
1,0.431,0.875
...
```

---

## Shared Utilities (`data_utils.py`)

| Function | Description |
|----------|-------------|
| `list_session_ids(data_dir, split)` | Session IDs sorted by D-number |
| `load_session(data_dir, sid, is_test)` | Load SBP, kinematics, trial info, dropout_ind |
| `get_validation_sessions(data_dir)` | Chronological split: last 20 = hard val, 8 random = easy val |
| `session_zscore_params(sbp)` | Per-channel mean/std computed on non-zero timepoints only |
| `zscore_normalize / zscore_denormalize` | Normalisation preserving zeroed channels |
| `compute_r2 / compute_r2_multi` | Local R² evaluation matching metric.py |
| `build_submission(predictions, data_dir, output)` | Align with test_index.csv → submission CSV |

---

## Methods

### Method 1: GRU Baseline (`gru_baseline/`)

Direct position decoding with a 2-layer bidirectional GRU. No pretraining.

- **Input:** `[sbp_z (96) | dropout_ind (96)]` = 192 dims per timestep
- **Output:** `[index_pos, mrp_pos]` per timestep via Sigmoid
- **Architecture:** BiGRU(192 → hidden=256×2) → LayerNorm → Linear(512→256) → GELU → Linear(256→2) → Sigmoid
- **GPU:** Yes (1× H200/L40S)
- **Key params:** `--epochs 30 --batch-size 128 --lr 5e-4`

### Method 2: MAE Pretrained Fine-tune (`mae_finetune/`)

Loads the Phase 1 MAE encoder (pretrained on masked SBP reconstruction) and adds a new position decode head. The encoder has learned temporal SBP dynamics and cross-channel structure — directly applicable to decoding.

- **Input:** `[sbp_z (96) | dropout_ind (96) | zeros (4)]` = 196 dims (matches Phase 1 contract)
- **Architecture:** Phase 1 Transformer encoder (4L, d=256, norm-first) + new decode head
- **Weight transfer:** `input_proj`, `pos_enc`, `encoder` loaded from Phase 1; `output_head` → `recon_head` (for TTA); `decode_head` randomly initialised
- **Training:** Differential LRs — `encoder_lr=1e-5` (pretrained), `head_lr=5e-4` (new head)
- **TTA (submit.py):** Self-supervised per-session fine-tuning using masked-SBP reconstruction objective (no labels needed). Adapts the encoder to each test session's channel statistics.
- **GPU:** Yes (1× H200/L40S)
- **Key params:** `--epochs 40 --encoder-lr 1e-5 --head-lr 5e-4`
- **TTA params:** `--tta-epochs 5 --tta-lr 1e-5`

**Setting the Phase 1 checkpoint:**
```bash
# Find the best Phase 1 sweep checkpoint:
bash /scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/pick_best.sh
# Then set before sbatch:
export PHASE1_CKPT=/path/to/sweep_JOBID_N/best_model.pt
```

### Method 3: Transformer Decoder from Scratch (`transformer_decoder/`)

Same Transformer architecture as Phase 1 MAE but trained from scratch for decoding. Acts as a control experiment — quantifies how much Phase 1 pretraining actually helps.

- **Input:** `[sbp_z (96) | dropout_ind (96)]` = 192 dims per timestep
- **Architecture:** Input projection + LayerNorm → Sinusoidal PE → 4L Transformer (d=256, norm-first) → per-timestep decode head → Sigmoid
- **GPU:** Yes (1× H200/L40S)
- **Key params:** `--epochs 50 --batch-size 64 --lr 5e-4`

### Method 4: Ensemble Decoder (`ensemble_decoder/`)  ← **Recommended**

Implements teammate's recommendation: fine-tune Phase 2 decoders from multiple Phase 1
checkpoints (user's own + teammate's m1/m2/m5) and average predictions.

**Why this helps:**
1. **Ensemble diversity**: Each Phase 1 checkpoint (different seed / EMA) produces a
   different encoder representation → averaging lowers prediction variance
2. **Balanced early stopping**: The base mae_finetune code evaluates only `val_ids[:4]`
   (the 4 easiest val sessions) → biased model selection. Method 4 uses a balanced
   4 easy + 4 hard session subset, so early stopping accounts for chronologically-hard
   sessions that resemble the test distribution
3. **More epochs (50 vs 40)**: More training with the balanced val signal

- **Input:** `[sbp_z (96) | dropout_ind (96) | zeros (4)]` = 196 dims (matches Phase 1)
- **Architecture:** Same MAEFinetuneDecoder as Method 2 (compatible checkpoints)
- **Init:** 4 different Phase 1 checkpoints: user's own, teammate's m1, m2, m5
- **Seeds:** 0–3 for reproducible diversity across models
- **Ensemble:** Plain average of 4 model predictions (no TTA — TTA was found to hurt)
- **GPU:** Yes (1× H200/L40S, ~20h total for 4 × 50ep training + inference)

**Running:**
```bash
# Full pipeline (train 4 models + ensemble inference):
sbatch ensemble_decoder/ensemble.sbatch

# Or sequentially without SLURM:
cd ensemble_decoder
bash run_from_p1_weights.sh
```

### Method 5: MAE Multitask Decoder (`mae_multitask_decoder/`)

Extends the plain MAE fine-tune baseline by training on all four kinematic targets
(`index_pos`, `mrp_pos`, `index_vel`, `mrp_vel`) instead of positions alone.
Inference predicts positions with a hybrid direct-position / integrated-velocity
blend, averaged across overlapping windows to reduce chunk-boundary artifacts.

**Why this may help:**
1. **Uses extra supervision**: the provided velocity channels are exact finite
   differences of position, so they add dense learning signal without extra labels
2. **Temporal consistency**: an explicit consistency loss encourages
   `pos[t+1] - pos[t] ≈ vel[t]`, which better matches smooth finger trajectories
3. **Robustness to dropout**: training applies synthetic additional channel dropout
   on top of the permanent per-session missing channels

- **Input:** `[sbp_z (96) | dropout_ind (96) | zeros (4)]` = 196 dims
- **Output:** multitask heads for 2 positions + 2 velocities
- **Architecture:** Phase 1 MAE encoder + shared decode trunk + position head + velocity head
- **Init:** best Phase 1 checkpoint (default:
  `phase1/masked_autoencoder/sweep_3325164_5/best_model.pt`)
- **Validation:** balanced easy+hard subset during training, full 28-session val at the end
- **Inference:** overlap-averaged windows (`max_seq_len=192`, `stride=96`) with
  `position_blend=0.75`
- **GPU:** Yes (1× H200/L40S)
- **Key params:** `--epochs 50 --encoder-lr 1e-5 --head-lr 5e-4 --vel-weight 0.5 --consistency-weight 0.25`

**Running:**
```bash
export PHASE1_CKPT=/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_3325164_5/best_model.pt
sbatch mae_multitask_decoder/mae_multitask_decoder.sbatch
```

### Method 6: Transformer Multitask Scratch (`transformer_multitask_scratch/`)

Implements the "train from scratch end-to-end" direction suggested in teammate
notes. This is a pure supervised decoder with no Phase 1 initialisation, but it
keeps the multitask position/velocity objective and hybrid inference used in
Method 5.

**Why this may help:**
1. **Avoids representation mismatch**: if Phase 1 reconstruction pretraining is
   biasing the encoder toward SBP reconstruction features rather than decoding,
   scratch training removes that constraint
2. **Strong control experiment**: it measures whether the new multitask objective
   is helping on its own, independent of pretraining
3. **Same robustness tricks**: balanced validation, synthetic extra dropout, and
   overlap inference are retained

- **Input:** `[sbp_z (96) | dropout_ind (96)]` = 192 dims
- **Output:** multitask heads for 2 positions + 2 velocities
- **Architecture:** 6-layer Transformer decoder from scratch
- **GPU:** Yes (1× H200/L40S)
- **Key params:** `--epochs 60 --lr 3e-4 --num-layers 6 --vel-weight 0.5 --consistency-weight 0.25`

**Running:**
```bash
sbatch transformer_multitask_scratch/transformer_multitask_scratch.sbatch
```

### Method 7: MAE Context Partial Fine-tune (`mae_context_partial_ft/`)

Uses the Phase 1 MAE encoder, but conditions the decoder on session-level context
available at test time: the dropout mask plus per-channel SBP mean/std. It also
fine-tunes only the last encoder layers by default, aiming to preserve useful
pretrained structure while adapting to session drift more cautiously.

**Why this may help:**
1. **Session-aware decoding**: different sessions have different active channels
   and SBP statistics, so explicit context lets the decoder adapt predictions to
   the observed session
2. **Reduced overfitting to drift**: partial fine-tuning updates only the last
   encoder blocks (`--unfreeze-last-n 2` by default) instead of rewriting the
   entire pretrained representation
3. **Combines pretraining and multitask supervision**: retains the stronger
   Phase 1 prior while still learning positions and velocities jointly

- **Input:** `[sbp_z (96) | dropout_ind (96) | zeros (4)]` = 196 dims
- **Context:** `[dropout_ind (96) | raw_mean (96) | raw_std (96)]` = 288 dims
- **Output:** multitask heads for 2 positions + 2 velocities
- **Architecture:** Phase 1 MAE encoder + context-conditioned decode trunk
- **Init:** best Phase 1 checkpoint (default:
  `phase1/masked_autoencoder/sweep_3325164_5/best_model.pt`)
- **Fine-tuning:** partial by default (`--unfreeze-last-n 2`)
- **GPU:** Yes (1× H200/L40S)
- **Key params:** `--epochs 50 --encoder-lr 1e-5 --head-lr 4e-4 --position-blend 0.8 --extra-dropout-max 6`

**Running:**
```bash
export PHASE1_CKPT=/scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder/sweep_3325164_5/best_model.pt
sbatch mae_context_partial_ft/mae_context_partial_ft.sbatch
```

---

## Usage

```bash
# Quick local sanity check
cd <method_dir>
python train.py --quick

# Standard single run
sbatch <method>.sbatch

# Experimental new methods
sbatch mae_multitask_decoder/mae_multitask_decoder.sbatch
sbatch transformer_multitask_scratch/transformer_multitask_scratch.sbatch
sbatch mae_context_partial_ft/mae_context_partial_ft.sbatch

# Hyperparameter sweep (8 configs in parallel)
export PHASE1_CKPT=...   # mae_finetune only
sbatch sweep.sbatch

# After sweep: find best config
bash pick_best.sh

# Submit sweep (TTA variants, mae_finetune only)
export MAE_FT_CKPT=/path/to/sweep_X/best_model.pt
sbatch submit_sweep.sbatch
```

---

## Train / Validation Split

- **187 train sessions**, sorted chronologically by D-number
- **Val = last 20 + 8 random** = 28 sessions held out from training
  - Last 20 (hard): largest temporal gap → simulates neural drift challenge
  - 8 random (easy): scattered through timeline
- **159 sessions** used for actual training

---

## Results

### Validation R² (local, held-out train sessions)

| # | Method | Val R² (8-sess balanced) | Full Val R² (28-sess) | Key Config | Notes |
|---|--------|--------------------------|----------------------|------------|-------|
| — | Session mean | 0.0 | 0.0 | — | Trivial baseline |
| — | Ridge Regression | ≈ -4.1 | — | Wiener filter | Provided baseline |
| — | GRU (provided) | ≈ 0.21 | — | hidden=128, 2L | Provided baseline |
| — | Transformer + SSL (provided) | ≈ 0.45 | — | pretrained | Provided baseline |
| 1 | GRU Baseline | 0.704 (4-sess easy only) | 0.498 | h=256, 2L, 30ep | |
| 2 | MAE Finetune | 0.696 (4-sess easy only) | 0.522 | enc_lr=1e-5, hd_lr=5e-4, 40ep | |
| 2+TTA | MAE Finetune + TTA | — | — | + 5ep @ lr=1e-5 | TTA hurts on test |
| 3 | Transformer Decoder | — | — | lr=5e-4, 4L, 50ep | |
| 4a | Ensemble (user+m1+m2+m5) | 0.681 (m2 best member) | ~0.513 avg | 4 models, 50ep, balanced val | **Best** |
| 4b | Ensemble (m1+m2+m5 only) | — | — | 3 teammate models | |
| 5 | MAE Multitask Decoder | 0.729 | 0.608 | pos+vel multitask, overlap inference | Best single model so far |
| 6 | Transformer Multitask Scratch | not run yet | not run yet | scratch multitask control | Kaggle public 0.4188 |
| 7 | MAE Context Partial FT | not run yet | not run yet | context-conditioned partial FT | Kaggle public 0.4989 |

### Kaggle Leaderboard (Public R²)

| # | Method | Public R² | Notes |
|---|--------|-----------|-------|
| 1 | GRU Baseline | 0.3872 | Big val/test gap |
| 2 | MAE Finetune (plain) | 0.4740 | Single model |
| 2+TTA5 | MAE Finetune + TTA (5ep) | 0.4218 | TTA hurts |
| 4b | Ensemble (m1+m2+m5) | 0.5117 | 3-model ensemble |
| **4a** | **Ensemble (user+m1+m2+m5)** | **0.5185** | **Best — 4-model ensemble** |
| 5 | MAE Multitask Decoder | 0.5439 | Best single-model result so far |
| 6 | Transformer Multitask Scratch | 0.4188 | Scratch multitask underperforms pretrained variants |
| 7 | MAE Context Partial FT | 0.4989 | Better than plain MAE, below multitask single model |

---

## Observations and Diagnosis

### Why current scores are lower on Kaggle than subset validation

1. **Biased early stopping**: `mae_finetune/train.py` validates on `val_ids[:4]`, which
   are the 4 easiest sessions (lowest D-numbers, smallest temporal gap from training).
   The model is selected for easy-session performance but tested on hard sessions.
   **Fix (Method 4)**: balanced val subset of 4 easy + 4 hard sessions.

2. **Single Phase 1 init → high prediction variance**: Using one pretrained encoder
   produces highly correlated errors across sessions. Averaging 4 differently-seeded
   Phase 1 initialisations (user's own, m1, m2, m5) reduces this variance.
   **Fix (Method 4)**: ensemble of 4 diverse Phase 1 inits.

3. **TTA hurts Phase 2**: Self-supervised reconstruction TTA adapts the encoder but
   moves it away from position-relevant features → net harm. The Phase 1 TTA worked
   because its TTA objective (reconstruction) matched the test objective (also
   reconstruction). In Phase 2, the test objective is position decoding.
   **Fix (Method 4)**: no TTA by default.

4. **Val R² computed on too few sessions**: `val_ids[:4]` = ~1% of sessions → noisy
   estimate → wrong checkpoint selected as "best". Full val R² (28 sessions) is ~0.2
   lower than the 4-session val R², confirming the bias.

5. **Method 5 confirms the same pattern even after improving the objective**:
   MAE multitask decoding reached **0.7287** on the balanced 8-session validation
   subset and **0.6077** on the full 28-session validation set, but only **0.5439**
   on Kaggle public test. This suggests the new multitask objective helps, but the
   remaining gap is still dominated by chronologically-harder test sessions and the
   lack of explicit cross-session alignment / ensemble variance reduction.
# neuro_p2
