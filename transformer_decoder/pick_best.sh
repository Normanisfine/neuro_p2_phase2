#!/bin/bash
# Find best sweep config and print submission commands
cd /scratch/ml8347/neuroinformatics/project2/phase2/transformer_decoder

echo "=== Transformer Decoder Sweep Results ==="
printf "%-55s %10s %8s\n" "Config" "Full Val R²" "Best Ep"
printf "%-55s %10s %8s\n" "------" "-----------" "-------"

BEST_R2=-999
BEST_DIR=""

for dir in sweep_*/; do
    cfg="$dir/config.json"
    if [ -f "$cfg" ]; then
        r2=$(python3 -c "import json; print(f\"{json.load(open('$cfg'))['full_val_r2']:.6f}\")" 2>/dev/null)
        epoch=$(python3 -c "import json; print(json.load(open('$cfg'))['best_epoch'])" 2>/dev/null)
        lr=$(python3 -c "import json; print(json.load(open('$cfg'))['lr'])" 2>/dev/null)
        do_=$(python3 -c "import json; print(json.load(open('$cfg'))['dropout'])" 2>/dev/null)
        ep=$(python3 -c "import json; print(json.load(open('$cfg'))['epochs'])" 2>/dev/null)
        nl=$(python3 -c "import json; print(json.load(open('$cfg'))['num_layers'])" 2>/dev/null)

        label="${dir%/}  lr=$lr do=$do_ L=$nl ep=$ep"
        printf "%-55s %10s %8s\n" "$label" "$r2" "$epoch"

        is_better=$(python3 -c "print(1 if $r2 > $BEST_R2 else 0)" 2>/dev/null)
        if [ "$is_better" = "1" ]; then
            BEST_R2=$r2
            BEST_DIR=$dir
        fi
    else
        printf "%-55s %10s\n" "${dir%/}" "(not done)"
    fi
done

echo ""
echo "=== BEST: $BEST_DIR  R²=$BEST_R2 ==="
echo ""
echo "Generate submission (run inside singularity):"
echo ""
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --output sub_best.csv"
