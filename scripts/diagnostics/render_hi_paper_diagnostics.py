from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams.update({
    "figure.dpi": 180,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})

HOPE2_TOKEN = "hi_lewm_p2_train_hope2_22253175"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render paper-ready figures from saved hi diagnostics.")
    p.add_argument("--offline-log-root", type=Path, required=True)
    p.add_argument("--acting-log-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--matrix-csv", type=Path, required=True)
    p.add_argument("--baseline-md", type=Path, required=True, help="Baseline markdown snapshot or generated baseline summary CSV.")
    return p.parse_args()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="	")


def filter_hope2(df: pd.DataFrame) -> pd.DataFrame:
    if "policy" in df.columns:
        return df[df["policy"].astype(str).str.contains(HOPE2_TOKEN, na=False)].copy()
    return df.copy()


def save_fig(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix('.png'), bbox_inches='tight')
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def baseline_d50_best(path: Path) -> float | None:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "goal_offset_steps" not in df.columns or "success_rate_mean" not in df.columns:
            return None
        rows = df[df["goal_offset_steps"].astype(str) == "50"]
        if rows.empty:
            return None
        values = pd.to_numeric(rows["success_rate_mean"], errors="coerce").dropna()
        return float(values.max()) if not values.empty else None

    text = path.read_text()
    m = re.search(r"\| `D50` \| 6 \| `([0-9.]+)` \| `([0-9.]+)` \| `([0-9.]+)` \|", text)
    return float(m.group(2)) if m else None


def render_teacher_vs_open_loop(df: pd.DataFrame, out_dir: Path) -> None:
    df = filter_hope2(df)
    df = df.sort_values(["goal_offset_steps", "high_horizon"])
    labels, teacher, open_true, open_cem = [], [], [], []
    for _, row in df.iterrows():
        labels.append(f"D{int(row['goal_offset_steps'])} H{int(row['high_horizon'])}")
        teacher.append(float(row["teacher_forced_mse_mean"]))
        open_true.append(float(row["open_loop_true_mse_mean"]))
        open_cem.append(float(row["open_loop_cem_mse_mean"]))
    x = range(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - w for i in x], teacher, width=w, label="Teacher-forced")
    ax.bar(list(x), open_true, width=w, label="Open-loop true macro")
    ax.bar([i + w for i in x], open_cem, width=w, label="Open-loop CEM macro")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latent MSE")
    ax.set_title("High-Level Teacher vs Open-Loop Error")
    ax.legend(frameon=False)
    save_fig(fig, out_dir / 'teacher_vs_open_loop')


def render_failure_ladder(oracle: pd.DataFrame, generated: pd.DataFrame, online: pd.DataFrame, baseline_best: float | None, out_dir: Path) -> None:
    oracle_row = filter_hope2(oracle).query("goal_offset_steps == 50 and high_horizon == 2 and low_horizon == 2 and low_receding_horizon == 1").iloc[0]
    generated_row = filter_hope2(generated).query("goal_offset_steps == 50 and high_horizon == 2 and low_horizon == 2 and low_receding_horizon == 1").iloc[0]
    online_row = filter_hope2(online).query("goal_offset_steps == 50 and high_horizon == 2 and low_horizon == 2 and low_receding_horizon == 1").iloc[0]
    labels = ["Oracle subgoal acting", "Generated subgoal acting", "Online hierarchical"]
    values = [float(oracle_row['success_rate']), float(generated_row['success_rate']), float(online_row['success_rate'])]
    if baseline_best is not None:
        labels.append("Baseline best")
        values.append(float(baseline_best))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=["#457b9d", "#e76f51", "#264653", "#2a9d8f"][:len(labels)])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, v + 1.0, f"{v:.0f}", ha='center', va='bottom')
    ax.set_ylim(0, max(values) + 15)
    ax.set_ylabel("Success rate (%)")
    ax.set_title("Failure Ladder at D50")
    save_fig(fig, out_dir / 'failure_ladder_d50')


def render_low_level_panels(oracle: pd.DataFrame, gap: pd.DataFrame, out_dir: Path) -> None:
    oracle = filter_hope2(oracle).query("goal_offset_steps == 50 and high_horizon == 2 and low_receding_horizon == 1").sort_values("low_horizon")
    gap = filter_hope2(gap).query("goal_offset_steps == 50 and low_receding_horizon == 1").sort_values("low_horizon")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(oracle['low_horizon'], oracle['success_rate'], marker='o')
    axes[0].set_title('Oracle Subgoal Acting')
    axes[0].set_xlabel('Low horizon')
    axes[0].set_ylabel('Success rate (%)')
    axes[1].plot(gap['low_horizon'], gap['overall_actual_error_mean'], marker='o', label='Actual error')
    axes[1].plot(gap['low_horizon'], gap['overall_reality_gap_mean'], marker='s', label='Reality gap')
    axes[1].set_title('Low-Level Reality Gap')
    axes[1].set_xlabel('Low horizon')
    axes[1].set_ylabel('Latent error')
    axes[1].legend(frameon=False)
    save_fig(fig, out_dir / 'low_level_horizon_sweep')


def render_temporal_validity(generated: pd.DataFrame, out_dir: Path) -> None:
    rows = filter_hope2(generated).query("goal_offset_steps == 50 and low_horizon == 2 and low_receding_horizon == 1").sort_values('high_horizon')
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(rows))
    step1 = rows['step1_offset_error_token_mean'].astype(float).tolist()
    step2 = [float(v) if pd.notna(v) else 0.0 for v in rows['step2_offset_error_token_mean']]
    w = 0.35
    ax.bar([i - w/2 for i in x], step1, width=w, label='Stage 1')
    ax.bar([i + w/2 for i in x], step2, width=w, label='Stage 2')
    ax.axhline(0.0, color='black', linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"H{int(h)}" for h in rows['high_horizon']])
    ax.set_ylabel('Offset error (tokens)')
    ax.set_title('Generated Subgoal Temporal Validity')
    ax.legend(frameon=False)
    save_fig(fig, out_dir / 'generated_subgoal_offset_error')


def render_online_churn(online: pd.DataFrame, out_dir: Path) -> None:
    rows = filter_hope2(online).query("goal_offset_steps == 50").copy()
    rows['label'] = rows.apply(lambda r: f"H{int(r['high_horizon'])}/L{int(r['low_horizon'])}/R{int(r['low_receding_horizon'])}", axis=1)
    rows = rows.sort_values(['high_horizon', 'low_horizon', 'low_receding_horizon'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(rows['label'], rows['success_rate'].astype(float), color='#457b9d')
    axes[0].set_ylabel('Success rate (%)')
    axes[0].set_title('Online Success')
    axes[0].tick_params(axis='x', rotation=30)
    axes[1].bar(rows['label'], rows['mean_subgoal_churn_mse'].astype(float), color='#e76f51')
    axes[1].set_ylabel('Subgoal churn MSE')
    axes[1].set_title('Replanning Instability')
    axes[1].tick_params(axis='x', rotation=30)
    save_fig(fig, out_dir / 'online_churn_and_success')


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / 'figures'
    table_dir = args.output_dir / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    teacher = read_tsv(args.offline_log_root / 'summary_teacher_vs_open_loop.tsv')
    oracle = read_tsv(args.acting_log_root / 'summary_oracle_subgoal_acting.tsv')
    gap = read_tsv(args.acting_log_root / 'summary_low_level_reality_gap.tsv')
    generated = read_tsv(args.acting_log_root / 'summary_generated_subgoal_acting.tsv')
    online = read_tsv(args.acting_log_root / 'summary_online_hierarchical_logging.tsv')

    teacher.to_csv(table_dir / 'teacher_vs_open_loop.csv', index=False)
    oracle.to_csv(table_dir / 'oracle_subgoal_acting.csv', index=False)
    gap.to_csv(table_dir / 'low_level_reality_gap.csv', index=False)
    generated.to_csv(table_dir / 'generated_subgoal_acting.csv', index=False)
    online.to_csv(table_dir / 'online_hierarchical_logging.csv', index=False)

    render_teacher_vs_open_loop(teacher, fig_dir)
    render_failure_ladder(oracle, generated, online, baseline_d50_best(args.baseline_md), fig_dir)
    render_low_level_panels(oracle, gap, fig_dir)
    render_temporal_validity(generated, fig_dir)
    render_online_churn(online, fig_dir)

    manifest = {
        'offline_log_root': str(args.offline_log_root),
        'acting_log_root': str(args.acting_log_root),
        'matrix_csv': str(args.matrix_csv),
        'baseline_md': str(args.baseline_md),
        'figures': sorted(str(p.name) for p in fig_dir.iterdir()),
        'tables': sorted(str(p.name) for p in table_dir.iterdir()),
    }
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
