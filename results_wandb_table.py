import wandb
import pandas as pd
import re
import numpy as np

# Connect to W&B
api = wandb.Api()
project_name = "SMALA_embedding_expts"
project = f"saranga7/{project_name}"
runs = api.runs(project)

# Collect results
records = []
for run in runs:
    # fetch the metrics you're interested in
    acc = run.summary.get("test_initial/accuracy_weighted")
    acc_slides = run.summary.get("test_initial/accuracy_weighted_slides")
    f1_slides = run.summary.get("test_initial/f1_weighted_slides")
    specificity_slides = run.summary.get("test_initial/specificity_weighted_slides")
    sensitivity_slides = run.summary.get("test_initial/recall_slides")

    # skip incomplete runs
    if acc is None or acc_slides is None:
        continue

    name = run.name or ""

    # Normalize group name by removing any _fold<digits> and _seed<digits> anywhere in the name.
    # Example: "dinov3_vitb16_vote_fold3_seed1234" -> "dinov3_vitb16_vote"
    group = re.sub(r'(_fold\d+|_seed\d+)', '', name)

    # try to extract fold/seed values (optional; keep for inspection)
    fold_match = re.search(r'_fold(\d+)', name)
    seed_match = re.search(r'_seed(\d+)', name)
    fold = int(fold_match.group(1)) if fold_match else None
    seed = int(seed_match.group(1)) if seed_match else None

    records.append({
        "run": name,
        "group": group,
        "fold": fold,
        "seed": seed,
        "accuracy_imgs": float(acc),                 # raw floats; rounding later
        "accuracy": float(acc_slides),
        "f1": float(f1_slides) if f1_slides is not None else np.nan,
        "specificity": float(specificity_slides) if specificity_slides is not None else np.nan,
        "sensitivity": float(sensitivity_slides) if sensitivity_slides is not None else np.nan,
    })

# Build DataFrame of per-run results
df_runs = pd.DataFrame(records)

if df_runs.empty:
    raise SystemExit("No complete runs found. Check metric names / availability.")

# Aggregate across all runs in the same 'group' (this averages across seeds & folds)
agg_funcs = {
    "accuracy_imgs": ["mean", "std"],
    "accuracy": ["mean", "std"],
    "f1": ["mean", "std"],
    "specificity": ["mean", "std"],
    "sensitivity": ["mean", "std"],
}

grouped = df_runs.groupby("group").agg(agg_funcs)
# flatten multiindex columns
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
grouped = grouped.reset_index()

# Round numbers to 3 decimals for presentation
for c in grouped.columns:
    if grouped[c].dtype.kind in 'f':
        grouped[c] = grouped[c].round(3)

# Prepare human-friendly "mean ± std" columns (handle NaNs)
def fmt_mean_std(mean, std):
    if pd.isna(mean):
        return ""
    std_str = f"{std:.3f}" if not pd.isna(std) else "nan"
    return f"{mean:.3f} ± {std_str}"

grouped["Accuracy (Images)"] = grouped.apply(lambda r: fmt_mean_std(r["accuracy_imgs_mean"], r["accuracy_imgs_std"]), axis=1)
grouped["Accuracy"] = grouped.apply(lambda r: fmt_mean_std(r["accuracy_mean"], r["accuracy_std"]), axis=1)
grouped["F1"] = grouped.apply(lambda r: fmt_mean_std(r["f1_mean"], r["f1_std"]), axis=1)
grouped["Specificity"] = grouped.apply(lambda r: fmt_mean_std(r["specificity_mean"], r["specificity_std"]), axis=1)
grouped["Sensitivity"] = grouped.apply(lambda r: fmt_mean_std(r["sensitivity_mean"], r["sensitivity_std"]), axis=1)

# Final table: keep only nice columns in desired order
table = grouped[[
    "group",
    "Accuracy (Images)",
    "Accuracy",
    "F1",
    "Specificity",
    "Sensitivity"
]].sort_values("group").reset_index(drop=True)

# Print to console
print("Per-run results (first 10 rows):")
print(df_runs.head(10).to_string(index=False))
print("\nGrouped results (mean ± std):")
print(table.to_string(index=False))

# Save outputs
df_runs.to_csv(f"wandb_runs_per_run_{project_name}.csv", index=False, float_format="%.6f")
table.to_csv(f"wandb_results_grouped_{project_name}.csv", index=False)
print(f"\nSaved per-run CSV: wandb_runs_per_run_{project_name}.csv")
print(f"Saved grouped CSV: wandb_results_grouped_{project_name}.csv")
