import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta

# Small demo project:
# analyze synthetic operations data
# and generate simple risk insights

OUTPUT_DIR = Path("operations_risk_output")
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_data(days=90, seed=42):
    rng = np.random.default_rng(seed)
    start_date = datetime.today().date() - timedelta(days=days - 1)

    lines = ["Lime Line", "Heat Unit", "Process Unit"]
    records = []

    for i in range(days):
        date = start_date + timedelta(days=i)

        for line in lines:
            base_output = {
                "Lime Line": 108,
                "Heat Unit": 98,
                "Process Unit": 120,
            }[line]

            base_energy = {
                "Lime Line": 540,
                "Heat Unit": 515,
                "Process Unit": 565,
            }[line]

            base_downtime = {
                "Lime Line": 1.3,
                "Heat Unit": 1.7,
                "Process Unit": 1.1,
            }[line]

            output_tons = base_output + rng.normal(0, 5)
            energy_mwh = base_energy + rng.normal(0, 18)
            downtime_hours = max(0, base_downtime + rng.normal(0, 0.6))
            defect_rate = max(0, 1.8 + rng.normal(0, 0.35))

            if rng.random() < 0.08:
                energy_mwh += rng.uniform(35, 90)
                downtime_hours += rng.uniform(1.5, 4.0)
                output_tons -= rng.uniform(8, 20)
                defect_rate += rng.uniform(0.8, 2.0)

            records.append(
                {
                    "date": date,
                    "production_line": line,
                    "output_tons": round(output_tons, 2),
                    "energy_mwh": round(energy_mwh, 2),
                    "downtime_hours": round(downtime_hours, 2),
                    "defect_rate_pct": round(defect_rate, 2),
                }
            )

    df = pd.DataFrame(records)

    missing_indices = rng.choice(df.index, size=max(3, len(df) // 40), replace=False)
    half = len(missing_indices) // 2

    for idx in missing_indices[:half]:
        df.loc[idx, "energy_mwh"] = np.nan

    for idx in missing_indices[half:]:
        df.loc[idx, "downtime_hours"] = np.nan

    return df


def clean_data(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["production_line", "date"]).reset_index(drop=True)

    df["energy_mwh"] = df.groupby("production_line")["energy_mwh"].transform(
        lambda s: s.fillna(s.median())
    )
    df["downtime_hours"] = df.groupby("production_line")["downtime_hours"].transform(
        lambda s: s.fillna(s.median())
    )

    df["energy_per_ton"] = df["energy_mwh"] / df["output_tons"]

    # Basic KPIs
    df["high_downtime_flag"] = df["downtime_hours"] > df["downtime_hours"].quantile(0.90)
    df["high_defect_flag"] = df["defect_rate_pct"] > df["defect_rate_pct"].quantile(0.90)

    return df


def flag_risk_patterns(df):
    df = df.copy()

    df["rolling_energy_per_ton"] = df.groupby("production_line")["energy_per_ton"].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )
    df["rolling_output"] = df.groupby("production_line")["output_tons"].transform(
        lambda s: s.rolling(window=7, min_periods=3).mean()
    )

    df["energy_spike_flag"] = df["energy_per_ton"] > df["rolling_energy_per_ton"] * 1.10
    df["output_drop_flag"] = df["output_tons"] < df["rolling_output"] * 0.92

    df["risk_score"] = (
        df["energy_spike_flag"].astype(int)
        + df["output_drop_flag"].astype(int)
        + df["high_downtime_flag"].astype(int)
        + df["high_defect_flag"].astype(int)
    )

    df["risk_level"] = pd.cut(
        df["risk_score"],
        bins=[-1, 0, 1, 2, 4],
        labels=["Low", "Moderate", "High", "Critical"],
    )

    return df


def create_summary_table(df):
    summary = (
        df.groupby("production_line")
        .agg(
            avg_output_tons=("output_tons", "mean"),
            avg_energy_per_ton=("energy_per_ton", "mean"),
            avg_downtime_hours=("downtime_hours", "mean"),
            avg_defect_rate_pct=("defect_rate_pct", "mean"),
            critical_days=("risk_level", lambda s: (s == "Critical").sum()),
            high_or_critical_days=("risk_level", lambda s: s.isin(["High", "Critical"]).sum()),
        )
        .reset_index()
    )

    return summary.sort_values("high_or_critical_days", ascending=False)


def generate_recommendations(summary):
    recommendations = []

    top_risk_line = summary.iloc[0]
    worst_energy_line = summary.sort_values("avg_energy_per_ton", ascending=False).iloc[0]
    worst_downtime_line = summary.sort_values("avg_downtime_hours", ascending=False).iloc[0]

    recommendations.append(
        f"{top_risk_line['production_line']} should be reviewed first, since it had the most high-risk days during the period."
    )
    recommendations.append(
        f"{worst_energy_line['production_line']} appears to be the least energy-efficient line and may need closer investigation."
    )
    recommendations.append(
        f"{worst_downtime_line['production_line']} shows the highest average downtime, which may point to maintenance or process issues."
    )

    return recommendations


def save_outputs(raw_df, analyzed_df, summary_df, recommendations):
    raw_df.to_csv(OUTPUT_DIR / "01_raw_operations_data.csv", index=False)
    analyzed_df.to_csv(OUTPUT_DIR / "02_analyzed_operations_data.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "03_summary_table.csv", index=False)

    high_risk_days = analyzed_df.loc[
        analyzed_df["risk_level"].isin(["High", "Critical"]),
        [
            "date",
            "production_line",
            "output_tons",
            "energy_per_ton",
            "downtime_hours",
            "defect_rate_pct",
            "risk_score",
            "risk_level",
        ],
    ].sort_values(["risk_score", "date"], ascending=[False, False])

    high_risk_days.to_csv(OUTPUT_DIR / "04_high_risk_days.csv", index=False)

    with open(OUTPUT_DIR / "05_recommendations.txt", "w", encoding="utf-8") as f:
        f.write("Recommendations\n")
        f.write("=" * 16 + "\n\n")
        for i, item in enumerate(recommendations, start=1):
            f.write(f"{i}. {item}\n")


def save_chart(summary_df):
    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["production_line"], summary_df["high_or_critical_days"])
    plt.title("High or Critical Risk Days by Production Line")
    plt.xlabel("Production Line")
    plt.ylabel("Days")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_risk_days_chart.png")
    plt.close()


if __name__ == "__main__":
    raw_data = generate_data()
    cleaned_data = clean_data(raw_data)
    analyzed_data = flag_risk_patterns(cleaned_data)
    summary_table = create_summary_table(analyzed_data)
    recommendations = generate_recommendations(summary_table)

    save_outputs(raw_data, analyzed_data, summary_table, recommendations)
    save_chart(summary_table)

    print("Project files created in:", OUTPUT_DIR.resolve())
    print("\nSummary table:")
    print(summary_table.round(2).to_string(index=False))
    print("\nRecommendations:")
    for item in recommendations:
        print("-", item)