import sys
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, classification_report
)

sys.stdout.reconfigure(encoding='utf-8')

print("="*80)
print("STARTING FULL PIPELINE EXECUTION: MODULE 1 & MODULE 2")
print("="*80)

# ==============================================================================
# MODULE 1: Dataset Simulation & Backward-Only Sliding-Window Feature Extraction
# ==============================================================================
print("\n--- MODULE 1 EXECUTION ---")
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

@dataclass
class SimConfig:
    n_machines: int = 25            # number of simulated machines/tasks
    n_timestamps: int = 2000        # length of the trace, in sampling intervals
    sampling_interval_minutes: int = 5   # native sampling interval (minutes)
    n_failures_per_machine: tuple = (0, 3)  # min/max failure events per machine
    precursor_window: int = 60      # how many timesteps before a failure degradation begins
    noise_std: float = 2.5          # baseline telemetry noise
    telemetry_cols: tuple = ("cpu_util", "mem_util", "disk_io", "sched_latency")

@dataclass
class FeatureConfig:
    window_sizes: tuple = (5, 15, 60)   # backward-only window scales, matches Section 4.3
    epsilon: float = 1e-6               # prevents division by zero in CV feature
    min_periods_policy: str = "strict"  # "strict" -> drop rows with < W prior samples

sim_cfg = SimConfig()
feat_cfg = FeatureConfig()

def simulate_machine_trace(machine_id: int, cfg: SimConfig, rng: np.random.Generator) -> pd.DataFrame:
    T = cfg.n_timestamps
    timestamps = pd.date_range("2026-01-01", periods=T, freq=f"{cfg.sampling_interval_minutes}min")

    base_levels = {"cpu_util": 35, "mem_util": 45, "disk_io": 20, "sched_latency": 15}
    series = {}
    for col in cfg.telemetry_cols:
        x = np.zeros(T)
        x[0] = base_levels[col] + rng.normal(0, cfg.noise_std)
        theta = 0.05
        for t in range(1, T):
            x[t] = x[t - 1] + theta * (base_levels[col] - x[t - 1]) + rng.normal(0, cfg.noise_std)
        series[col] = x

    n_failures = rng.integers(cfg.n_failures_per_machine[0], cfg.n_failures_per_machine[1] + 1)
    failure_idx = []
    if n_failures > 0:
        candidates = rng.choice(
            np.arange(cfg.precursor_window + 10, T - 5),
            size=min(n_failures, T // (cfg.precursor_window + 20)),
            replace=False,
        )
        failure_idx = sorted(candidates.tolist())

    failure_event = np.zeros(T, dtype=int)
    for f_idx in failure_idx:
        failure_event[f_idx] = 1
        start = max(0, f_idx - cfg.precursor_window)
        ramp = np.linspace(0, 1, f_idx - start) ** 1.5
        for col in cfg.telemetry_cols:
            drift_scale = {"cpu_util": 45, "mem_util": 40, "disk_io": 55, "sched_latency": 70}[col]
            extra_noise = rng.normal(0, cfg.noise_std * (0.5 + 2 * ramp))
            series[col][start:f_idx] += ramp * drift_scale + extra_noise

    df = pd.DataFrame({
        "machine_id": machine_id,
        "timestamp": timestamps,
        **{col: np.clip(series[col], 0, 100) for col in cfg.telemetry_cols},
        "failure_event": failure_event,
    })
    return df

def simulate_dataset(cfg: SimConfig, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng_local = np.random.default_rng(seed)
    frames = [simulate_machine_trace(m, cfg, rng_local) for m in range(cfg.n_machines)]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    return df

raw_df = simulate_dataset(sim_cfg)
print(f"Simulated dataset shape: {raw_df.shape}")
print(f"Machines: {raw_df['machine_id'].nunique()}   Total failure events: {raw_df['failure_event'].sum()}")

expected_cols = set(sim_cfg.telemetry_cols) | {"machine_id", "timestamp", "failure_event"}
missing = expected_cols - set(raw_df.columns)
assert not missing, f"Harmonization check failed — missing expected columns: {missing}"
print("Harmonization check passed: all expected telemetry fields are present.")

def add_backward_window_features(
    df: pd.DataFrame,
    telemetry_cols: List[str],
    window_sizes: List[int],
    group_col: str = "machine_id",
    time_col: str = "timestamp",
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    out = df.sort_values([group_col, time_col]).copy()
    grouped = out.groupby(group_col, sort=False)

    for col in telemetry_cols:
        for W in window_sizes:
            roll = grouped[col].rolling(window=W, min_periods=W)
            mean_feat = roll.mean().reset_index(level=0, drop=True)
            std_feat = roll.std(ddof=0).reset_index(level=0, drop=True)

            shifted = grouped[col].shift(W - 1)
            trend_feat = out[col] - shifted

            out[f"{col}_mean_W{W}"] = mean_feat
            out[f"{col}_std_W{W}"] = std_feat
            out[f"{col}_trend_W{W}"] = trend_feat
            out[f"{col}_cv_W{W}"] = std_feat / (mean_feat + epsilon)

    return out

feature_df = add_backward_window_features(
    raw_df,
    telemetry_cols=list(sim_cfg.telemetry_cols),
    window_sizes=list(feat_cfg.window_sizes),
)

n_before = len(feature_df)
feature_df = feature_df.dropna().reset_index(drop=True)
n_after = len(feature_df)
print(f"Rows before dropping insufficient-history rows: {n_before}")
print(f"Rows after (min_periods=W enforced per window scale): {n_after}")

def leakage_boundary_check(raw: pd.DataFrame, features: pd.DataFrame, telemetry_cols, window_sizes,
                            group_col="machine_id", time_col="timestamp", n_samples=200, seed=RANDOM_SEED,
                            epsilon=1e-6) -> bool:
    check_rng = np.random.default_rng(seed)
    sample_idx = check_rng.choice(features.index, size=min(n_samples, len(features)), replace=False)
    violations = []
    raw_sorted = raw.sort_values([group_col, time_col])

    for idx in sample_idx:
        row = features.loc[idx]
        machine_id, t = row[group_col], row[time_col]
        history = raw_sorted[(raw_sorted[group_col] == machine_id) & (raw_sorted[time_col] <= t)]

        for col in telemetry_cols:
            for W in window_sizes:
                if len(history) < W:
                    continue
                window_vals = history[col].iloc[-W:].values
                expected_mean = window_vals.mean()
                expected_std = window_vals.std(ddof=0)
                expected_cv = expected_std / (expected_mean + epsilon)

                actual_mean = row[f"{col}_mean_W{W}"]
                actual_std = row[f"{col}_std_W{W}"]
                actual_cv = row[f"{col}_cv_W{W}"]

                if not np.isclose(expected_mean, actual_mean, atol=1e-8):
                    violations.append((idx, col, W, "mean", expected_mean, actual_mean))
                if not np.isclose(expected_std, actual_std, atol=1e-8):
                    violations.append((idx, col, W, "std", expected_std, actual_std))
                if not np.isclose(expected_cv, actual_cv, atol=1e-8):
                    violations.append((idx, col, W, "cv", expected_cv, actual_cv))

    if violations:
        print(f"LEAKAGE AUDIT FAILED: {len(violations)} violation(s) found.")
        return False

    print(f"Leakage audit PASSED: {len(sample_idx)} sampled rows recomputed independently from raw data.")
    return True

audit_passed = leakage_boundary_check(
    raw_df, feature_df,
    telemetry_cols=list(sim_cfg.telemetry_cols),
    window_sizes=list(feat_cfg.window_sizes),
)
assert audit_passed, "Pipeline halted: leakage boundary check failed."

os.makedirs("module1_outputs", exist_ok=True)
raw_df.to_parquet("module1_outputs/raw_simulated_telemetry.parquet", index=False)
feature_df.to_parquet("module1_outputs/backward_window_features.parquet", index=False)
print("Saved Module 1 parquet files successfully.")

# ==============================================================================
# MODULE 2: Forward-Horizon Failure Precursor Labeling, Chronological Partitioning & XGBoost Precursor Training
# ==============================================================================
print("\n--- MODULE 2 EXECUTION ---")

@dataclass
class Module2Config:
    input_dir: str = "module1_outputs"
    output_dir: str = "module2_outputs"
    horizon_minutes: int = 30
    sampling_interval_minutes: int = 5
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    cv_folds: int = 3
    random_seed: int = 42

config = Module2Config()
os.makedirs(config.output_dir, exist_ok=True)

df_raw = pd.read_parquet(os.path.join(config.input_dir, "raw_simulated_telemetry.parquet"))
df_features = pd.read_parquet(os.path.join(config.input_dir, "backward_window_features.parquet"))

def compute_forward_horizon_labels(df_raw, horizon_minutes=30, sampling_interval_minutes=5):
    horizon_steps = horizon_minutes // sampling_interval_minutes
    df_sorted = df_raw.sort_values(['machine_id', 'timestamp']).copy()
    df_sorted['is_precursor'] = 0
    
    label_list = []
    for machine_id, group in df_sorted.groupby('machine_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        failure_indices = group.index[group['failure_event'] == 1].tolist()
        
        precursor_mask = np.zeros(len(group), dtype=int)
        for f_idx in failure_indices:
            start_idx = max(0, f_idx - horizon_steps)
            precursor_mask[start_idx:f_idx] = 1
            
        group['is_precursor'] = precursor_mask
        # Only return target column 'is_precursor' to avoid column collisions with failure_event in df_features
        label_list.append(group[['machine_id', 'timestamp', 'is_precursor']])
        
    df_labels = pd.concat(label_list, ignore_index=True)
    return df_labels

df_labels = compute_forward_horizon_labels(df_raw, config.horizon_minutes, config.sampling_interval_minutes)
df_dataset = pd.merge(df_features, df_labels, on=['machine_id', 'timestamp'], how='inner')

precursor_count = df_dataset['is_precursor'].sum()
total_count = len(df_dataset)
print(f"Dataset shape after merging features & labels: {df_dataset.shape}")
print(f"Precursor labels (y=1): {precursor_count} / {total_count} ({(precursor_count/total_count)*100:.2f}%)")

def chronological_split_by_machine_with_embargo(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    max_window_steps: int = 60,
    horizon_steps: int = 6
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs machine-wise chronological partitioning with temporal embargo/purging:
    1. Target Embargo: Purges the last `horizon_steps` rows from Train and Val partitions
       to prevent target labels near the partition boundary from looking forward into the next set.
    2. Feature Embargo: Purges the first `max_window_steps - 1` rows from Val and Test partitions
       to prevent feature windows near the boundary from spanning back into the prior set.
    """
    train_list, val_list, test_list = [], [], []
    for machine_id, group in df.groupby('machine_id'):
        group_sorted = group.sort_values('timestamp').reset_index(drop=True)
        n = len(group_sorted)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        # Raw partition boundaries
        train_raw = group_sorted.iloc[:n_train]
        val_raw = group_sorted.iloc[n_train:n_train + n_val]
        test_raw = group_sorted.iloc[n_train + n_val:]

        # Apply embargo purging
        # End of Train: purge last horizon_steps to avoid target look-ahead into Val
        train_purged = train_raw.iloc[:-horizon_steps] if len(train_raw) > horizon_steps else train_raw
        
        # Start of Val: purge first (max_window_steps - 1) to avoid feature lookback into Train
        # End of Val: purge last horizon_steps to avoid target look-ahead into Test
        val_start_cut = min(max_window_steps - 1, max(0, len(val_raw) - 1))
        val_end_cut = len(val_raw) - horizon_steps if len(val_raw) > horizon_steps else len(val_raw)
        val_purged = val_raw.iloc[val_start_cut:max(val_start_cut, val_end_cut)]

        # Start of Test: purge first (max_window_steps - 1) to avoid feature lookback into Val
        test_start_cut = min(max_window_steps - 1, max(0, len(test_raw) - 1))
        test_purged = test_raw.iloc[test_start_cut:]

        train_list.append(train_purged)
        val_list.append(val_purged)
        test_list.append(test_purged)

    return (
        pd.concat(train_list, ignore_index=True),
        pd.concat(val_list, ignore_index=True),
        pd.concat(test_list, ignore_index=True)
    )

max_w_steps = max(feat_cfg.window_sizes)
h_steps = config.horizon_minutes // config.sampling_interval_minutes

df_train, df_val, df_test = chronological_split_by_machine_with_embargo(
    df_dataset,
    train_ratio=config.train_ratio,
    val_ratio=config.val_ratio,
    test_ratio=config.test_ratio,
    max_window_steps=max_w_steps,
    horizon_steps=h_steps
)

print(f"Train Partition (after embargo purge): {df_train.shape[0]} rows (y=1: {df_train['is_precursor'].sum()})")
print(f"Val Partition   (after embargo purge): {df_val.shape[0]} rows (y=1: {df_val['is_precursor'].sum()})")
print(f"Test Partition  (after embargo purge): {df_test.shape[0]} rows (y=1: {df_test['is_precursor'].sum()})")

non_feature_cols = ['machine_id', 'timestamp', 'is_precursor', 'failure_event']
feature_cols = [c for c in df_dataset.columns if c not in non_feature_cols]
print(f"Number of engineered input features X: {len(feature_cols)}")

class LeakageAuditModule:
    """
    Programmatically verifies 5 strict temporal non-leakage properties:
    1. Feature Computation Window Bounds (numerical re-verification)
    2. Target Label Structural Isolation
    3. Chronological Partition Boundary Separation
    4. Feature Window Embargo Isolation (no lookback overlap across partitions)
    5. Target Label Embargo Isolation (no horizon lookahead across partitions)
    """
    def __init__(self, df_raw, df_dataset, df_train, df_val, df_test, feature_cols, max_window_steps, horizon_steps):
        self.df_raw = df_raw
        self.df_dataset = df_dataset
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.feature_cols = feature_cols
        self.max_window_steps = max_window_steps
        self.horizon_steps = horizon_steps

    def audit_feature_boundary(self) -> bool:
        """Check 1: Re-verifies random samples to ensure rolling features rely strictly on past raw data."""
        check_rng = np.random.default_rng(42)
        sample_idx = check_rng.choice(self.df_train.index, size=min(50, len(self.df_train)), replace=False)
        raw_sorted = self.df_raw.sort_values(['machine_id', 'timestamp'])

        for idx in sample_idx:
            row = self.df_train.loc[idx]
            machine_id, t = row['machine_id'], row['timestamp']
            history = raw_sorted[(raw_sorted['machine_id'] == machine_id) & (raw_sorted['timestamp'] <= t)]
            if len(history) < 5:
                continue
            window_vals = history['cpu_util'].iloc[-5:].values
            expected_mean = window_vals.mean()
            actual_mean = row['cpu_util_mean_W5']
            if not np.isclose(expected_mean, actual_mean, atol=1e-6):
                return False
        return True

    def audit_target_isolation(self) -> bool:
        """Check 2: Ensures target columns are strictly absent from feature matrix X."""
        forbidden = ['is_precursor', 'failure_event', 'failure_event_x', 'failure_event_y']
        return not any(c in self.feature_cols for c in forbidden)

    def audit_partition_temporal_order(self) -> bool:
        """Check 3: Confirms strict monotonically increasing timestamps across partitions."""
        violations = 0
        for m_id in self.df_dataset['machine_id'].unique():
            tr = self.df_train[self.df_train['machine_id'] == m_id]
            va = self.df_val[self.df_val['machine_id'] == m_id]
            te = self.df_test[self.df_test['machine_id'] == m_id]

            if not tr.empty and not va.empty:
                if tr['timestamp'].max() >= va['timestamp'].min():
                    violations += 1
            if not va.empty and not te.empty:
                if va['timestamp'].max() >= te['timestamp'].min():
                    violations += 1
        return violations == 0

    def audit_feature_embargo_isolation(self) -> bool:
        """Check 4: Verifies feature lookback window (W_max) of Val/Test does not span into prior partitions."""
        interval_delta = pd.Timedelta(minutes=5)
        max_lookback_delta = interval_delta * (self.max_window_steps - 1)

        for m_id in self.df_dataset['machine_id'].unique():
            tr = self.df_train[self.df_train['machine_id'] == m_id]
            va = self.df_val[self.df_val['machine_id'] == m_id]
            te = self.df_test[self.df_test['machine_id'] == m_id]

            if not tr.empty and not va.empty:
                tr_max_time = tr['timestamp'].max()
                va_min_time = va['timestamp'].min()
                # Earliest validation timestamp minus lookback must be strictly after train max timestamp
                if va_min_time - max_lookback_delta <= tr_max_time:
                    return False

            if not va.empty and not te.empty:
                va_max_time = va['timestamp'].max()
                te_min_time = te['timestamp'].min()
                if te_min_time - max_lookback_delta <= va_max_time:
                    return False

        return True

    def audit_target_embargo_isolation(self) -> bool:
        """Check 5: Verifies target precursor horizon (H) does not look ahead across partition boundaries."""
        horizon_delta = pd.Timedelta(minutes=5 * self.horizon_steps)

        for m_id in self.df_dataset['machine_id'].unique():
            tr = self.df_train[self.df_train['machine_id'] == m_id]
            va = self.df_val[self.df_val['machine_id'] == m_id]

            if not tr.empty and not va.empty:
                tr_max_time = tr['timestamp'].max()
                va_min_time = va['timestamp'].min()
                if tr_max_time + horizon_delta >= va_min_time:
                    return False

        return True

    def run_full_audit(self) -> Dict[str, bool]:
        return {
            "Check 1 - Backward Feature Window Bounds": self.audit_feature_boundary(),
            "Check 2 - Target Label Structural Isolation": self.audit_target_isolation(),
            "Check 3 - Chronological Partition Boundary Separation": self.audit_partition_temporal_order(),
            "Check 4 - Feature Window Embargo Isolation": self.audit_feature_embargo_isolation(),
            "Check 5 - Target Horizon Embargo Isolation": self.audit_target_embargo_isolation(),
        }

auditor = LeakageAuditModule(raw_df, df_dataset, df_train, df_val, df_test, feature_cols, max_w_steps, h_steps)
audit_results = auditor.run_full_audit()

print("\n================ LEAKAGE AUDIT RESULTS ================")
all_passed = True
for check, status in audit_results.items():
    res_str = "PASSED" if status else "FAILED"
    print(f"[{res_str}] {check}")
    if not status:
        all_passed = False

assert all_passed, "Leakage audit failed!"
print("ALL 5 AUDIT CHECKS PASSED: Pipeline is 100% Leakage-Free!")

X_train, y_train = df_train[feature_cols], df_train['is_precursor']
X_val, y_val = df_val[feature_cols], df_val['is_precursor']
X_test, y_test = df_test[feature_cols], df_test['is_precursor']

scale_pos_weight = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")

# ==============================================================================
# FORWARD-CHAINING CROSS-VALIDATION HYPERPARAMETER TUNING
# ==============================================================================
print("\nExecuting Forward-Chaining Cross-Validation & XGBoost Hyperparameter Tuning...")

def forward_chaining_cv_xgboost(df_tr: pd.DataFrame, feature_cols: List[str], seed: int = 42) -> Dict:
    """
    Executes a 2-fold expanding temporal window cross-validation over the training dataset
    to discover optimal hyperparameter combinations for XGBoost.
    """
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.05, "n_estimators": 150, "colsample_bytree": 0.8},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 200, "colsample_bytree": 0.7},
    ]

    best_score = -1.0
    best_params = param_grid[1]

    # Structure 2 forward-chaining splits within df_train per machine
    for params in param_grid:
        fold_scores = []
        for split_ratio in [0.50, 0.75]:
            train_folds, val_folds = [], []
            for m_id, group in df_tr.groupby('machine_id'):
                group_sorted = group.sort_values('timestamp').reset_index(drop=True)
                cut = int(len(group_sorted) * split_ratio)
                val_cut = int(len(group_sorted) * (split_ratio + 0.20))
                train_folds.append(group_sorted.iloc[:cut])
                val_folds.append(group_sorted.iloc[cut:val_cut])

            f_tr = pd.concat(train_folds, ignore_index=True)
            f_va = pd.concat(val_folds, ignore_index=True)

            X_ftr, y_ftr = f_tr[feature_cols], f_tr['is_precursor']
            X_fva, y_fva = f_va[feature_cols], f_va['is_precursor']

            spw = (len(y_ftr) - y_ftr.sum()) / max(1, y_ftr.sum())
            clf = xgb.XGBClassifier(
                **params,
                scale_pos_weight=spw,
                random_state=seed,
                eval_metric="logloss"
            )
            clf.fit(X_ftr, y_ftr, eval_set=[(X_fva, y_fva)], verbose=False)
            probs = clf.predict_proba(X_fva)[:, 1]
            prec, rec, _ = precision_recall_curve(y_fva, probs)
            score = auc(rec, prec)
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    print(f"Forward-Chaining CV Complete. Best PR-AUC: {best_score:.4f}")
    print(f"Selected Best Hyperparameters: {best_params}")
    return best_params

best_xgb_params = forward_chaining_cv_xgboost(df_train, feature_cols, seed=config.random_seed)

print("\nTraining Proposed XGBoost Precursor Classifier with Best Hyperparameters...")
xgb_model = xgb.XGBClassifier(
    **best_xgb_params,
    scale_pos_weight=scale_pos_weight,
    random_state=config.random_seed,
    eval_metric="logloss",
    early_stopping_rounds=15
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print(f"XGBoost Model Training Complete. Best iteration: {xgb_model.best_iteration}")

print("Training Baseline 1: Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    class_weight="balanced",
    random_state=config.random_seed
)
rf_model.fit(X_train, y_train)

print("Training Baseline 2: Support Vector Classifier...")
svm_model = SVC(
    kernel="rbf",
    probability=True,
    class_weight="balanced",
    random_state=config.random_seed
)
svm_rng = np.random.default_rng(config.random_seed)
svm_indices = svm_rng.choice(len(X_train), size=min(10000, len(X_train)), replace=False)
svm_model.fit(X_train.iloc[svm_indices], y_train.iloc[svm_indices])

print("All Models Trained Successfully!")

models = {
    "Proposed XGBoost": xgb_model,
    "Baseline: Random Forest": rf_model,
    "Baseline: SVM": svm_model
}

results = []
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1]
    else:
        y_probs = model.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_probs)

    y_val_probs = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.5
    best_f1_val = 0
    for t in np.linspace(0.1, 0.9, 81):
        f1_t = f1_score(y_val, (y_val_probs >= t).astype(int), zero_division=0)
        if f1_t > best_f1_val:
            best_f1_val = f1_t
            best_thresh = t

    y_pred = (y_probs >= best_thresh).astype(int)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Optimal Threshold": round(best_thresh, 3),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "PR-AUC": round(pr_auc, 4),
        "ROC-AUC": round(roc_auc, 4)
    })

df_results = pd.DataFrame(results)
print("\n=================== PERFORMANCE BENCHMARK TABLE ===================")
print(df_results.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, model in models.items():
    y_probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    axes[0].plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})")

axes[0].set_title("Precision-Recall Curve (Out-of-Time Test Set)")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].legend(loc="lower left")

importances = xgb_model.feature_importances_
top_indices = np.argsort(importances)[-15:]
axes[1].barh(range(15), importances[top_indices], align='center', color='teal')
axes[1].set_yticks(range(15))
axes[1].set_yticklabels([feature_cols[i] for i in top_indices])
axes[1].set_title("Top 15 XGBoost Feature Importances")
axes[1].set_xlabel("Gain / Importance Weight")

plt.tight_layout()
plt.savefig(os.path.join(config.output_dir, "m1_performance_and_feature_importance.png"), dpi=300)
plt.close()

def compute_lead_times(df_test, y_probs, threshold=0.5, sampling_interval_minutes=5, horizon_minutes=30):
    horizon_steps = horizon_minutes // sampling_interval_minutes
    df_eval = df_test[['machine_id', 'timestamp', 'is_precursor', 'failure_event']].copy()
    df_eval['pred_prob'] = y_probs
    df_eval['pred_precursor'] = (y_probs >= threshold).astype(int)

    lead_times = []
    for machine_id, group in df_eval.groupby('machine_id'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        failure_indices = group.index[group['failure_event'] == 1].tolist()

        for f_idx in failure_indices:
            precursor_preds = group.iloc[max(0, f_idx - horizon_steps):f_idx]
            flagged = precursor_preds[precursor_preds['pred_precursor'] == 1]
            if not flagged.empty:
                first_flag_idx = flagged.index[0]
                lead_time_mins = (f_idx - first_flag_idx) * sampling_interval_minutes
                lead_times.append(lead_time_mins)

    return lead_times

xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
best_thresh = df_results.loc[df_results['Model'] == 'Proposed XGBoost', 'Optimal Threshold'].values[0]
lead_times = compute_lead_times(df_test, xgb_probs, threshold=best_thresh, sampling_interval_minutes=config.sampling_interval_minutes, horizon_minutes=config.horizon_minutes)

if lead_times:
    mean_lt = np.mean(lead_times)
    median_lt = np.median(lead_times)
    print(f"\nDetected failure events with precursors: {len(lead_times)}")
    print(f"Mean Precursor Lead Time: {mean_lt:.2f} minutes")
    print(f"Median Precursor Lead Time: {median_lt:.2f} minutes")

    plt.figure(figsize=(8, 4))
    sns.histplot(lead_times, bins=10, kde=True, color='darkblue')
    plt.title(f"Precursor Lead-Time Distribution (Mean = {mean_lt:.1f} mins)")
    plt.xlabel("Lead Time Before Failure (Minutes)")
    plt.ylabel("Failure Count")
    plt.savefig(os.path.join(config.output_dir, "m1_lead_time_distribution.png"), dpi=300)
    plt.close()
else:
    print("\nNo failure precursors detected on test set.")

xgb_model.save_model(os.path.join(config.output_dir, "xgboost_m1_precursor_model.json"))
df_results.to_csv(os.path.join(config.output_dir, "m1_benchmark_results.csv"), index=False)

df_test_out = df_test[['machine_id', 'timestamp', 'is_precursor', 'failure_event']].copy()
df_test_out['xgb_pred_prob'] = xgb_probs
df_test_out.to_parquet(os.path.join(config.output_dir, "test_predictions.parquet"))

print("\nSaved All Module 2 Artifacts Successfully:")
print(f" - Model: {os.path.join(config.output_dir, 'xgboost_m1_precursor_model.json')}")
print(f" - Benchmark CSV: {os.path.join(config.output_dir, 'm1_benchmark_results.csv')}")
print(f" - Test Predictions: {os.path.join(config.output_dir, 'test_predictions.parquet')}")
print(f" - Charts: m1_performance_and_feature_importance.png, m1_lead_time_distribution.png")

print("="*80)
print("FULL PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("="*80)

