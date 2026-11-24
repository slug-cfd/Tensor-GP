#!/bin/sh
# Rerun every script, write output/<script>.txt (and the figures), then compare with reference_output/.
# Usage: ./run_all.sh [python]      (default: python3)
PY=${1:-python3}
cd "$(dirname "$0")" || exit 1
mkdir -p output
for s in thresholds_true cond_curve_hp make_figures precision_matrix lgp_assembled_growth precision_matrix_2d \
         lgp_tensor_identity lgp_cross_check lgp_cross_3d t_outside_range lgp_vs_pure_weights \
         lgp_assembled_solvable lgp_vs_pure lgp_gp7_hp_rates pure_gp_viability thresholds pure_gp_2d3d; do
  printf '%-24s' "$s"
  start=$(date +%s)
  if $PY scripts/$s.py > output/$s.txt 2>&1; then echo "done ($(( $(date +%s) - start )) s)"; else echo "FAILED, see output/$s.txt"; fi
done
$PY compare_outputs.py
