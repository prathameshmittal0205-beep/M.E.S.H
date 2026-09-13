"""Data quality validation module for MESH.

Implements rigorous, traceable validation checks for:
- Missing, NaN, and Inf values
- Duplicate records
- Timestamp / cycle sequence monotonicity and gaps
- Physical boundary and range violations
- Label and target consistency
- Sensor availability summaries

Conforms to docs/data/DATA_PIPELINE.md and docs/NO_HALLUCINATION_POLICY.md.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from pipelines.data.schema import DataQualityReport


# ==============================================================================
# UNVERIFIED ENGINEERING ESTIMATES
# These are engineering approximations based on general aerospace / machining domain
# knowledge. They are NOT documented ground truth in the official dataset papers or
# readme files. Per MESH policy, they are strictly informational and DO NOT gate
# the QA pass/fail decision.
# ==============================================================================

CMAPSS_ENGINEERING_ESTIMATES: Dict[str, Tuple[float, float]] = {
    "s_1": (400.0, 700.0),    # T2: Fan inlet temp [°R]
    "s_2": (500.0, 800.0),    # T24: LPC outlet temp [°R]
    "s_3": (1200.0, 1800.0),  # T30: HPC outlet temp [°R]
    "s_4": (1000.0, 1600.0),  # T50: LPT outlet temp [°R]
    "s_5": (5.0, 30.0),       # P2: Fan inlet pressure [psia]
    "s_6": (10.0, 40.0),      # P15: Bypass duct pressure [psia]
    "s_7": (200.0, 700.0),    # P30: HPC outlet pressure [psia]
    "s_8": (1500.0, 3000.0),  # Nf: Physical fan speed [rpm]
    "s_9": (7000.0, 11000.0), # Nc: Physical core speed [rpm]
    "s_10": (0.5, 2.5),       # epr: Engine pressure ratio
    "s_11": (30.0, 60.0),     # Ps30: Static pressure [psia]
    "s_12": (100.0, 800.0),   # phi: Fuel flow ratio [pps/psi]
    "s_13": (2000.0, 3000.0), # NRf: Corrected fan speed [rpm]
    "s_14": (7000.0, 10000.0),# NRc: Corrected core speed [rpm]
    "s_15": (6.0, 12.0),      # BPR: Bypass ratio
    "s_16": (0.01, 0.05),     # farB: Burner fuel-air ratio
    "s_17": (300.0, 450.0),   # htBleed: Bleed enthalpy
    "s_18": (2000.0, 3000.0), # Nf_dmd: Demanded fan speed [rpm]
    "s_19": (80.0, 120.0),    # PCNfR_dmd: Demanded corrected speed [rpm]
    "s_20": (20.0, 50.0),     # W31: HPT coolant bleed [lbm/s]
    "s_21": (10.0, 35.0),     # W32: LPT coolant bleed [lbm/s]
}

AI4I_ENGINEERING_ESTIMATES: Dict[str, Tuple[float, float]] = {
    "Air temperature [K]": (270.0, 350.0),
    "Process temperature [K]": (270.0, 370.0),
    "Rotational speed [rpm]": (500.0, 3500.0),
    "Torque [Nm]": (0.0, 120.0),
    "Tool wear [min]": (0.0, 350.0),
}


class DataQualityValidator:
    """Validates raw and parsed dataframes for structural and domain integrity."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id

    def compute_empirical_stats(self, df: pd.DataFrame, numeric_cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Computes empirical statistics directly and traceably from the observed dataset."""
        stats = {}
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col]
                stats[col] = {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                }
        return stats

    def compute_empirical_bounds(
        self,
        df: pd.DataFrame,
        numeric_cols: Sequence[str],
        sigma_multiplier: float = 3.0,
    ) -> Dict[str, Dict[str, Any]]:
        """Derives empirical bounds traceably from the observed training distribution.

        Formula: [max(0.0, mean - k*std), mean + k*std] for physically non-negative sensors,
        supplemented by the actual observed min and max.
        """
        bounds = {}
        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                series = df[col]
                mean_val = float(series.mean())
                std_val = float(series.std())
                obs_min = float(series.min())
                obs_max = float(series.max())

                lower_3sigma = max(0.0, mean_val - sigma_multiplier * std_val)
                upper_3sigma = mean_val + sigma_multiplier * std_val

                bounds[col] = {
                    "observed_min": obs_min,
                    "observed_max": obs_max,
                    "mean": round(mean_val, 4),
                    "std": round(std_val, 4),
                    "empirical_lower_3sigma": round(lower_3sigma, 4),
                    "empirical_upper_3sigma": round(upper_3sigma, 4),
                    "margin_rule": f"mean +/- {sigma_multiplier}*std (lower bounded at 0.0 for non-negative quantities)",
                }
        return bounds

    def validate_cmapss(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
    ) -> Tuple[DataQualityReport, Dict[str, Any]]:
        """Runs full quality audit on NASA C-MAPSS dataframe."""
        notes: List[str] = []
        flagged_records: Dict[str, Any] = {}

        total_records = len(df)
        total_runs = df["unit_number"].nunique() if "unit_number" in df.columns else 0

        # 1. Missing / NaN / Inf values (Hard QA gate)
        nan_counts = df.isna().sum().to_dict()
        inf_counts = {col: int(np.isinf(df[col]).sum()) for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
        total_missing = sum(nan_counts.values()) + sum(inf_counts.values())
        if total_missing > 0:
            notes.append(f"Found {total_missing} missing or infinite values across columns.")

        # 2. Duplicate records (unit_number + time_cycles) (Hard QA gate)
        duplicate_count = 0
        if "unit_number" in df.columns and "time_cycles" in df.columns:
            duplicate_mask = df.duplicated(subset=["unit_number", "time_cycles"])
            duplicate_count = int(duplicate_mask.sum())
            if duplicate_count > 0:
                notes.append(f"Found {duplicate_count} duplicate (unit_number, time_cycles) records.")
                flagged_records["duplicates"] = df[duplicate_mask].index.tolist()

        # 3. Cycle monotonicity and sequence continuity per engine (Hard QA gate)
        monotonicity_violations = 0
        gap_count = 0
        if "unit_number" in df.columns and "time_cycles" in df.columns:
            for unit_id, group in df.groupby("unit_number"):
                cycles = group["time_cycles"].values
                diffs = np.diff(cycles)
                non_pos = np.where(diffs <= 0)[0]
                gaps = np.where(diffs > 1)[0]
                if len(non_pos) > 0:
                    monotonicity_violations += len(non_pos)
                if len(gaps) > 0:
                    gap_count += len(gaps)

            if monotonicity_violations > 0:
                notes.append(f"Found {monotonicity_violations} cycle monotonicity violations.")
            if gap_count > 0:
                notes.append(f"Found {gap_count} cycle gaps.")

        # 4. Physical feasibility (Hard QA gate: absolute temperatures, speeds, pressures must be > 0)
        impossible_reading_count = 0
        for sensor_col in [f"s_{i}" for i in range(1, 22)]:
            if sensor_col in df.columns:
                series = df[sensor_col]
                neg_mask = series < 0.0
                neg_num = int(neg_mask.sum())
                if neg_num > 0:
                    impossible_reading_count += neg_num
                    notes.append(f"Sensor {sensor_col} has {neg_num} physically impossible negative readings.")

        # 5. Informational engineering estimate screening (NOT a hard QA gate; marked explicitly)
        heuristic_oob_details: Dict[str, int] = {}
        for sensor_col, (low, high) in CMAPSS_ENGINEERING_ESTIMATES.items():
            if sensor_col in df.columns:
                series = df[sensor_col]
                oob_mask = (series < low) | (series > high)
                oob_num = int(oob_mask.sum())
                if oob_num > 0:
                    heuristic_oob_details[sensor_col] = oob_num

        # 6. Empirical bounds and distribution summary
        sensor_cols = [f"s_{i}" for i in range(1, 22)] + ["setting_1", "setting_2", "setting_3"]
        empirical_bounds = self.compute_empirical_bounds(df, sensor_cols, sigma_multiplier=3.0)
        empirical_stats = self.compute_empirical_stats(df, sensor_cols)

        # 7. Sensor availability summary
        availability: Dict[str, float] = {}
        for col in df.columns:
            valid_ratio = float((~df[col].isna()).mean())
            availability[col] = round(valid_ratio, 4)

        # 8. Target validation (if target_rul is present)
        if "target_rul" in df.columns:
            neg_rul = int((df["target_rul"] < 0).sum())
            if neg_rul > 0:
                notes.append(f"Found {neg_rul} negative RUL values.")
                monotonicity_violations += neg_rul

        passed = (
            total_missing == 0
            and duplicate_count == 0
            and monotonicity_violations == 0
            and gap_count == 0
            and impossible_reading_count == 0
        )

        notes.append("Hard QA gate evaluated on verifiable structural rules: missing values, duplicates, cycle continuity, and physical non-negativity.")
        notes.append(f"Informational engineering estimate screening: {len(heuristic_oob_details)} sensors with out-of-estimate readings (bounds_source: engineering_estimate_unverified).")

        report = DataQualityReport(
            dataset_id=self.dataset_id,
            total_records=total_records,
            total_runs=total_runs,
            missing_value_count=total_missing,
            duplicate_records_count=duplicate_count,
            monotonicity_violations=monotonicity_violations + gap_count,
            invalid_sensor_readings=impossible_reading_count,
            sensor_availability_summary=availability,
            passed_qa=passed,
            notes=notes,
        )

        details = {
            "dataset_id": self.dataset_id,
            "total_records": total_records,
            "total_runs": total_runs,
            "missing_values": total_missing,
            "duplicates": duplicate_count,
            "monotonicity_violations": monotonicity_violations,
            "cycle_gaps": gap_count,
            "physically_impossible_readings": impossible_reading_count,
            "bounds_source": "engineering_estimate_unverified",
            "engineering_estimate_out_of_bounds": heuristic_oob_details,
            "empirical_bounds_source": "empirical_training_distribution_3sigma",
            "empirical_sensor_bounds": empirical_bounds,
            "empirical_statistics": empirical_stats,
            "dropped_records_count": 0,
            "passed_qa": passed,
            "notes": notes,
        }

        return report, details

    def validate_ai4i(
        self,
        df: pd.DataFrame,
    ) -> Tuple[DataQualityReport, Dict[str, Any]]:
        """Runs full quality audit on AI4I 2020 synthetic benchmark dataframe."""
        notes: List[str] = []
        total_records = len(df)
        total_runs = df["Product ID"].nunique() if "Product ID" in df.columns else total_records

        # 1. Missing / NaN / Inf values (Hard QA gate)
        nan_counts = df.isna().sum().to_dict()
        inf_counts = {col: int(np.isinf(df[col]).sum()) for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
        total_missing = sum(nan_counts.values()) + sum(inf_counts.values())
        if total_missing > 0:
            notes.append(f"Found {total_missing} missing or infinite values.")

        # 2. Duplicate UDI or Product ID (Hard QA gate)
        duplicate_udi = int(df.duplicated(subset=["UDI"]).sum()) if "UDI" in df.columns else 0
        duplicate_prod = int(df.duplicated(subset=["Product ID"]).sum()) if "Product ID" in df.columns else 0
        duplicate_count = duplicate_udi + duplicate_prod
        if duplicate_count > 0:
            notes.append(f"Found {duplicate_udi} duplicate UDIs and {duplicate_prod} duplicate Product IDs.")

        # 3. UDI Monotonicity (Hard QA gate)
        monotonicity_violations = 0
        if "UDI" in df.columns:
            udis = df["UDI"].values
            diffs = np.diff(udis)
            non_monotonic = int((diffs != 1).sum())
            if non_monotonic > 0:
                monotonicity_violations = non_monotonic
                notes.append(f"Found {non_monotonic} non-consecutive UDI sequence steps.")

        # 4. Physical non-negativity (Hard QA gate: Kelvin temps, speed, torque, wear must be >= 0)
        impossible_reading_count = 0
        sensor_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        for sensor_col in sensor_cols:
            if sensor_col in df.columns:
                series = df[sensor_col]
                neg_num = int((series < 0.0).sum())
                if neg_num > 0:
                    impossible_reading_count += neg_num
                    notes.append(f"Column '{sensor_col}' has {neg_num} physically impossible negative values.")

        # 5. Informational engineering estimate screening (NOT a hard QA gate; marked explicitly)
        heuristic_oob_details: Dict[str, int] = {}
        for sensor_col, (low, high) in AI4I_ENGINEERING_ESTIMATES.items():
            if sensor_col in df.columns:
                series = df[sensor_col]
                oob_mask = (series < low) | (series > high)
                oob_num = int(oob_mask.sum())
                if oob_num > 0:
                    heuristic_oob_details[sensor_col] = oob_num

        # 6. Empirical bounds and distribution summary
        empirical_bounds = self.compute_empirical_bounds(df, sensor_cols, sigma_multiplier=3.0)
        empirical_stats = self.compute_empirical_stats(df, sensor_cols)

        # 7. Label consistency check (Hard QA gate)
        label_inconsistencies = 0
        for mode in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
            if mode in df.columns:
                invalid_flags = int((~df[mode].isin([0, 1])).sum())
                if invalid_flags > 0:
                    label_inconsistencies += invalid_flags
                    notes.append(f"Found {invalid_flags} non-binary values in target flag '{mode}'.")

        # Check process temperature vs air temperature
        if "Process temperature [K]" in df.columns and "Air temperature [K]" in df.columns:
            temp_diff = df["Process temperature [K]"] - df["Air temperature [K]"]
            inverted_temps = int((temp_diff < 0).sum())
            if inverted_temps > 0:
                notes.append(f"Note: {inverted_temps} records have Process temp < Air temp.")

        # 8. Sensor availability summary
        availability: Dict[str, float] = {}
        for col in df.columns:
            valid_ratio = float((~df[col].isna()).mean())
            availability[col] = round(valid_ratio, 4)

        passed = (
            total_missing == 0
            and duplicate_count == 0
            and monotonicity_violations == 0
            and impossible_reading_count == 0
            and label_inconsistencies == 0
        )

        notes.append("Hard QA gate evaluated on verifiable structural rules: missing values, duplicates, UDI continuity, and non-negativity.")
        notes.append(f"Informational engineering estimate screening: {len(heuristic_oob_details)} sensors with out-of-estimate readings (bounds_source: engineering_estimate_unverified).")

        report = DataQualityReport(
            dataset_id=self.dataset_id,
            total_records=total_records,
            total_runs=total_runs,
            missing_value_count=total_missing,
            duplicate_records_count=duplicate_count,
            monotonicity_violations=monotonicity_violations,
            invalid_sensor_readings=impossible_reading_count,
            sensor_availability_summary=availability,
            passed_qa=passed,
            notes=notes,
        )

        details = {
            "dataset_id": self.dataset_id,
            "total_records": total_records,
            "total_runs": total_runs,
            "missing_values": total_missing,
            "duplicates": duplicate_count,
            "monotonicity_violations": monotonicity_violations,
            "physically_impossible_readings": impossible_reading_count,
            "bounds_source": "engineering_estimate_unverified",
            "engineering_estimate_out_of_bounds": heuristic_oob_details,
            "empirical_bounds_source": "empirical_training_distribution_3sigma",
            "empirical_sensor_bounds": empirical_bounds,
            "empirical_statistics": empirical_stats,
            "dropped_records_count": 0,
            "passed_qa": passed,
            "notes": notes,
        }

        return report, details


def save_quality_report(report_details: Dict[str, Any], output_path: Path) -> None:
    """Serializes quality audit report to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_details, f, indent=2)


def run_validation_from_config(
    config: Dict[str, Any],
    root_dir: Optional[Path] = None,
) -> Tuple[DataQualityReport, Dict[str, Any]]:
    """Runs data quality validation audit driven by configuration."""
    if root_dir is None:
        root_dir = Path.cwd()

    ds_cfg = config["dataset"]
    paths_cfg = config["paths"]

    dataset_name = ds_cfg["name"]
    dataset_id = ds_cfg["dataset_id"]
    output_path = root_dir / paths_cfg["qa_report_output"]

    validator = DataQualityValidator(dataset_id=dataset_id)

    if dataset_name == "cmapss_fd001":
        from pipelines.data.adapters.cmapss import CMAPSSAdapter
        subdataset = ds_cfg.get("subdataset", "FD001")
        drop_invariant = ds_cfg.get("drop_invariant_sensors", False)
        adapter = CMAPSSAdapter(subdataset=subdataset, drop_invariant_sensors=drop_invariant)

        train_path = root_dir / paths_cfg["train_file"]
        if not train_path.exists():
            raise FileNotFoundError(f"C-MAPSS raw training file not found: {train_path}")

        df = adapter.load_raw(train_path)
        report, details = validator.validate_cmapss(df, is_train=True)
        save_quality_report(details, output_path)
        return report, details

    elif dataset_name == "ai4i2020":
        from pipelines.data.adapters.ai4i import AI4IAdapter
        adapter = AI4IAdapter(dataset_version=ds_cfg.get("dataset_version", "1.0.0"))

        raw_path = root_dir / paths_cfg["train_file"]
        if not raw_path.exists():
            raise FileNotFoundError(f"AI4I raw file not found: {raw_path}")

        df = adapter.load_raw(raw_path)
        report, details = validator.validate_ai4i(df)
        save_quality_report(details, output_path)
        return report, details

    else:
        raise ValueError(f"Unsupported dataset for validation: '{dataset_name}'")


def run_full_validation_audit() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Runs data quality validation audit on both C-MAPSS FD001 and AI4I 2020."""
    root_dir = Path.cwd()

    import yaml
    c_cfg_path = root_dir / "configs" / "data" / "cmapss_fd001.yaml"
    a_cfg_path = root_dir / "configs" / "data" / "ai4i2020.yaml"

    if c_cfg_path.exists() and a_cfg_path.exists():
        with open(c_cfg_path, "r", encoding="utf-8") as f:
            c_cfg = yaml.safe_load(f)
        with open(a_cfg_path, "r", encoding="utf-8") as f:
            a_cfg = yaml.safe_load(f)
        _, cmapss_details = run_validation_from_config(c_cfg, root_dir=root_dir)
        _, ai4i_details = run_validation_from_config(a_cfg, root_dir=root_dir)
        return cmapss_details, ai4i_details

    from pipelines.data.adapters.cmapss import CMAPSSAdapter
    from pipelines.data.adapters.ai4i import AI4IAdapter

    # 1. Audit C-MAPSS FD001
    cmapss_adapter = CMAPSSAdapter(subdataset="FD001")
    cmapss_path = root_dir / "data" / "raw" / "cmapss" / "train_FD001.txt"
    cmapss_df = cmapss_adapter.load_raw(cmapss_path)
    cmapss_val = DataQualityValidator(dataset_id="nasa_cmapss_fd001")
    _, cmapss_details = cmapss_val.validate_cmapss(cmapss_df, is_train=True)
    cmapss_out = root_dir / "data" / "interim" / "qa_reports" / "cmapss_fd001_qa.json"
    save_quality_report(cmapss_details, cmapss_out)

    # 2. Audit AI4I 2020
    ai4i_adapter = AI4IAdapter()
    ai4i_path = root_dir / "data" / "raw" / "ai4i2020" / "ai4i2020.csv"
    ai4i_df = ai4i_adapter.load_raw(ai4i_path)
    ai4i_val = DataQualityValidator(dataset_id="ai4i2020")
    _, ai4i_details = ai4i_val.validate_ai4i(ai4i_df)
    ai4i_out = root_dir / "data" / "interim" / "qa_reports" / "ai4i2020_qa.json"
    save_quality_report(ai4i_details, ai4i_out)

    return cmapss_details, ai4i_details


if __name__ == "__main__":
    cmapss_rep, ai4i_rep = run_full_validation_audit()
    print("Audit re-run complete.")
    print("C-MAPSS passed QA:", cmapss_rep["passed_qa"], "Out of estimate:", cmapss_rep["engineering_estimate_out_of_bounds"])
    print("AI4I 2020 passed QA:", ai4i_rep["passed_qa"], "Out of estimate:", ai4i_rep["engineering_estimate_out_of_bounds"])
