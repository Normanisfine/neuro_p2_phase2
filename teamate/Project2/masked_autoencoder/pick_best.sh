#!/bin/bash
# Find best sweep config and print TTA submission commands
cd /scratch/ml8347/neuroinformatics/project2/phase1/masked_autoencoder

echo "=== Sweep Results ==="
printf "%-55s %12s %8s\n" "Config" "Full Val NMSE" "Best Ep"
printf "%-55s %12s %8s\n" "------" "-------------" "-------"

BEST_NMSE=999
BEST_DIR=""

for dir in sweep_*/; do
    cfg="$dir/config.json"
    if [ -f "$cfg" ]; then
        nmse=$(python3 -c "import json; print(f\"{json.load(open('$cfg'))['full_val_nmse']:.6f}\")" 2>/dev/null)
        epoch=$(python3 -c "import json; print(json.load(open('$cfg'))['best_epoch'])" 2>/dev/null)
        lr=$(python3 -c "import json; print(json.load(open('$cfg'))['lr'])" 2>/dev/null)
        do=$(python3 -c "import json; print(json.load(open('$cfg'))['dropout'])" 2>/dev/null)
        rw=$(python3 -c "import json; print(json.load(open('$cfg'))['recon_weight'])" 2>/dev/null)
        ep=$(python3 -c "import json; print(json.load(open('$cfg'))['epochs'])" 2>/dev/null)

        label="${dir%/}  lr=$lr do=$do ep=$ep rw=$rw"
        printf "%-55s %12s %8s\n" "$label" "$nmse" "$epoch"

        is_better=$(python3 -c "print(1 if $nmse < $BEST_NMSE else 0)" 2>/dev/null)
        if [ "$is_better" = "1" ]; then
            BEST_NMSE=$nmse
            BEST_DIR=$dir
        fi
    else
        printf "%-55s %12s\n" "${dir%/}" "(not done)"
    fi
done

echo ""
echo "=== BEST: $BEST_DIR  NMSE=$BEST_NMSE ==="
echo ""
echo "Generate submissions (run inside singularity):"
echo ""
echo "  # Plain"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --output sub_best_plain.csv"
echo ""
echo "  # TTA sweep"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 3  --tta-lr 1e-4 --output sub_best_tta3.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5  --tta-lr 1e-4 --output sub_best_tta5.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 10 --tta-lr 1e-4 --output sub_best_tta10.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5  --tta-lr 5e-5 --output sub_best_tta5_lr5e5.csv"
echo ""
echo "  # TTA + ridge"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5 --tta-lr 1e-4 --ridge-blend 0.1 --output sub_best_tta5_r01.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5 --tta-lr 1e-4 --ridge-blend 0.2 --output sub_best_tta5_r02.csv"
echo "  python submit.py --checkpoint ${BEST_DIR}best_model.pt --tta-epochs 5 --tta-lr 1e-4 --ridge-blend 0.3 --output sub_best_tta5_r03.csv"
