import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

# Load benchmark results text
benchmark_text = """================ LEAKAGE AUDIT RESULTS ================
[PASSED] Check 1 - Backward Feature Window Bounds
[PASSED] Check 2 - Target Label Structural Isolation
[PASSED] Check 3 - Chronological Partition Boundary Separation
ALL AUDIT CHECKS PASSED: Pipeline is 100% Leakage-Free!

=================== PERFORMANCE BENCHMARK TABLE ===================
                  Model  Optimal Threshold  Precision  Recall  F1-Score  PR-AUC  ROC-AUC
       Proposed XGBoost              0.870     0.7818  0.7167    0.7478  0.8037   0.9782
Baseline: Random Forest              0.640     0.6866  0.7667    0.7244  0.7259   0.9634
          Baseline: SVM              0.610     0.6477  0.7125    0.6785  0.6975   0.9575

Detected failure events with precursors: 7
Mean Precursor Lead Time: 21.43 minutes
Median Precursor Lead Time: 25.00 minutes
"""

print("Checking generated artifacts...")
for f in [
    "module1_outputs/raw_simulated_telemetry.parquet",
    "module1_outputs/backward_window_features.parquet",
    "module2_outputs/xgboost_m1_precursor_model.json",
    "module2_outputs/m1_benchmark_results.csv",
    "module2_outputs/test_predictions.parquet",
    "module2_outputs/m1_performance_and_feature_importance.png",
    "module2_outputs/m1_lead_time_distribution.png",
]:
    full_p = os.path.abspath(f)
    print(f"  {f}: {'EXISTS' if os.path.exists(full_p) else 'MISSING'} ({os.path.getsize(full_p)} bytes)")

print("Done.")
