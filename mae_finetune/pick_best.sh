#!/bin/bash
# Find best sweep config and print TTA submission commands
cd /scratch/ml8347/neuroinformatics/project2/phase2/mae_finetune

echo "=== MAE Finetune Sweep Results ==="
printf "%-60s %10s %8s\n" "Config" "Full Val R²" "Best Ep"
printf "%-60s %10s %8s\n" "------" "-----------" "-------"

BEST_R2=-999
BEST_DIR=""

for dir in sweep_*/; do
    cfg="$dir/config.json"
    if [ -f "$cfg" ]; then
        r2=$(python3 -c "import json; print(f\"{json.load(open('$cfg'))['full_val_r2']:.6f}\")" 2>/dev/null)
        epoch=$(python3 -c "import json; print(json.load(open('$cfg'))['best_epoch'])" 2>/dev/null)
        elr=$(python3 -c "import json; print(json.load(open('$cfg'))['encoder_lr'])" 2>/dev/null)
        hlr=$(python3 -c "import json; print(json.load(open('$cfg'))['head_lr'])" 2>/dev/null)
        do_=$(python3 -c "import json; print(json.load(open('$cfg'))['dropout'])" 2>/dev/null)
        ep=$(python3 -c "import json; print(json.load(open('$cfg'))['epochs'])" 2>/dev/null)

        label="${dir%/}  elr=$elr hlr=$hlr do=$do_ ep=$ep"
        printf "%-60s %10s %8s\n" "$label" "$r2" "$epoch"

        is_better=$(python3 -c "print(1 if $r2 > $BEST_R2 else 0)" 2>/dev/null)
        if [ "$is_better" = "1" ]; then
            BEST_R2=$r2
            BEST_DIR=$dir
        fi
    else
        printf "%-60s %10s\n" "${dir%/}" "(not done)"
    fi
done

echo ""
echo "=== BEST: $BEST_DIR  R²=$BEST_R2 ==="
echo ""
echo "Generate submissions (run inside singularity):"
echo ""
echo "  # Plain"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --output sub_best_plain.csv"
echo ""
echo "  # TTA sweep"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 3  --tta-lr 1e-5 --output sub_best_tta3.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5  --tta-lr 1e-5 --output sub_best_tta5.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 10 --tta-lr 1e-5 --output sub_best_tta10.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5  --tta-lr 5e-6 --output sub_best_tta5_lr5e6.csv"
echo ""
echo "  # Or launch array sweep:"
echo "  export MAE_FT_CKPT=${BEST_DIR}best_model.pt"
echo "  sbatch submit_sweep.sbatch"
