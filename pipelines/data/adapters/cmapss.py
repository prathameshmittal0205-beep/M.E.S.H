"""NASA C-MAPSS Turbofan Engine Dataset Adapter for MESH.

Parses run-to-failure multivariate degradation data from C-MAPSS simulation runs
(FD001 to FD004) per docs/data/DATA_PIPELINE.md and Saxena et al. (PHM08).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

from pipelines.data.adapters.base import BaseDatasetAdapter
from pipelines.data.schema import (
    CanonicalSample,
    ModalityMetadata,
    Targets,
    WindowMetadata,
)

# 26 standard C-MAPSS columns as documented in readme.txt
CMAPSS_RAW_COLUMNS = [
    "unit_number",
    "time_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
    "s_1", "s_2", "s_3", "s_4", "s_5",
    "s_6", "s_7", "s_8", "s_9", "s_10",
    "s_11", "s_12", "s_13", "s_14", "s_15",
    "s_16", "s_17", "s_18", "s_19", "s_20", "s_21",
]

# Physical sensor descriptions and units from Damage Propagation Modeling.pdf Table 2
CMAPSS_SENSOR_METADATA = {
    "s_1": ("Total temperature at fan inlet (T2)", "°R"),
    "s_2": ("Total temperature at LPC outlet (T24)", "°R"),
    "s_3": ("Total temperature at HPC outlet (T30)", "°R"),
    "s_4": ("Total temperature at LPT outlet (T50)", "°R"),
    "s_5": ("Pressure at fan inlet (P2)", "psia"),
    "s_6": ("Total pressure in bypass-duct (P15)", "psia"),
    "s_7": ("Total pressure at HPC outlet (P30)", "psia"),
    "s_8": ("Physical fan speed (Nf)", "rpm"),
    "s_9": ("Physical core speed (Nc)", "rpm"),
    "s_10": ("Engine pressure ratio P50/P2 (epr)", "--"),
    "s_11": ("Static pressure at HPC outlet (Ps30)", "psia"),
    "s_12": ("Ratio of fuel flow to Ps30 (phi)", "pps/psi"),
    "s_13": ("Corrected fan speed (NRf)", "rpm"),
    "s_14": ("Corrected core speed (NRc)", "rpm"),
    "s_15": ("Bypass Ratio (BPR)", "--"),
    "s_16": ("Burner fuel-air ratio (farB)", "--"),
    "s_17": ("Bleed Enthalpy (htBleed)", "--"),
    "s_18": ("Demanded fan speed (Nf_dmd)", "rpm"),
    "s_19": ("Demanded corrected fan speed (PCNfR_dmd)", "rpm"),
    "s_20": ("HPT coolant bleed (W31)", "lbm/s"),
    "s_21": ("LPT coolant bleed (W32)", "lbm/s"),
}

# Constant channels at sea-level condition (FD001 / FD003)
INVARIANT_SEA_LEVEL_SENSORS = ["s_1", "s_5", "s_10", "s_16", "s_18", "s_19"]


class CMAPSSAdapter(BaseDatasetAdapter):
    """Adapter for NASA C-MAPSS Turbofan Engine Degradation Simulation data."""

    def __init__(
        self,
        subdataset: str = "FD001",
        dataset_version: str = "1.0.0",
        drop_invariant_sensors: bool = False,
    ):
        super().__init__(dataset_id=f"nasa_cmapss_{subdataset.lower()}", dataset_version=dataset_version)
        self.subdataset = subdataset.upper()
        self.drop_invariant_sensors = drop_invariant_sensors

    @property
    def supported_modalities(self) -> Dict[str, ModalityMetadata]:
        """Returns native physical sensor modality groupings."""
        if self.drop_invariant_sensors and self.subdataset in ("FD001", "FD003"):
            temp_channels = ["s_2", "s_3", "s_4"]
            press_channels = ["s_6", "s_7", "s_11"]
            speed_channels = ["s_8", "s_9", "s_13", "s_14"]
            gas_channels = ["s_12", "s_15", "s_17", "s_20", "s_21"]
        else:
            temp_channels = ["s_1", "s_2", "s_3", "s_4"]
            press_channels = ["s_5", "s_6", "s_7", "s_10", "s_11"]
            speed_channels = ["s_8", "s_9", "s_13", "s_14", "s_18", "s_19"]
            gas_channels = ["s_12", "s_15", "s_16", "s_17", "s_20", "s_21"]

        return {
            "temperatures": ModalityMetadata(
                name="temperatures",
                channels=temp_channels,
                units=[CMAPSS_SENSOR_METADATA[c][1] for c in temp_channels],
                description="Turbofan gas-path total temperatures",
            ),
            "pressures": ModalityMetadata(
                name="pressures",
                channels=press_channels,
                units=[CMAPSS_SENSOR_METADATA[c][1] for c in press_channels],
                description="Turbofan compressor/bypass pressures",
            ),
            "speeds": ModalityMetadata(
                name="speeds",
                channels=speed_channels,
                units=[CMAPSS_SENSOR_METADATA[c][1] for c in speed_channels],
                description="Physical and corrected spool rotational speeds",
            ),
            "gas_flow": ModalityMetadata(
                name="gas_flow",
                channels=gas_channels,
                units=[CMAPSS_SENSOR_METADATA[c][1] for c in gas_channels],
                description="Fuel ratio, bypass ratio, bleed enthalpy, and coolant bleeds",
            ),
            "operational_settings": ModalityMetadata(
                name="operational_settings",
                channels=["setting_1", "setting_2", "setting_3"],
                units=["altitude", "mach", "TRA"],
                description="Altitude, Mach number, and Throttle Resolver Angle",
            ),
        }

    @property
    def supported_targets(self) -> List[str]:
        return ["target_rul"]

    def load_raw(self, raw_path: Path, **kwargs: Any) -> pd.DataFrame:
        """Loads a raw space-delimited C-MAPSS file."""
        if not raw_path.exists():
            raise FileNotFoundError(f"C-MAPSS raw file not found: {raw_path}")

        df = pd.read_csv(
            raw_path,
            sep=r"\s+",
            header=None,
            names=CMAPSS_RAW_COLUMNS,
            engine="python",
        )
        return df

    def parse_runs(
        self,
        df: pd.DataFrame,
        is_train: bool = True,
        rul_file_path: Optional[Path] = None,
        **kwargs: Any,
    ) -> Dict[str, pd.DataFrame]:
        """Groups records by engine unit_number and attaches ground-truth RUL.

        For training trajectories: RUL(t) = max_cycle - t.
        For test trajectories: RUL(t) = RUL_final + (max_cycle - t).
        """
        runs = {}
        grouped = df.groupby("unit_number")

        test_ruls: Dict[int, float] = {}
        if not is_train:
            if rul_file_path is not None and rul_file_path.exists():
                rul_df = pd.read_csv(rul_file_path, sep=r"\s+", header=None)
                for idx, val in enumerate(rul_df[0].values):
                    test_ruls[idx + 1] = float(val)

        for unit_id, group in grouped:
            run_df = group.sort_values("time_cycles").copy().reset_index(drop=True)
            max_cycle = run_df["time_cycles"].max()

            if is_train:
                run_df["target_rul"] = (max_cycle - run_df["time_cycles"]).astype(float)
            else:
                base_remaining = test_ruls.get(int(unit_id), 0.0)
                run_df["target_rul"] = (base_remaining + (max_cycle - run_df["time_cycles"])).astype(float)

            runs[f"unit_{int(unit_id):03d}"] = run_df

        return runs

    def extract_windows(
        self,
        run_data: Dict[str, pd.DataFrame],
        window_size: int = 20,
        stride: int = 5,
        preprocessor_version: str = "raw_unscaled",
        scaler_version: str = "none",
    ) -> List[CanonicalSample]:
        """Generates canonical sliding windows from parsed run trajectories.

        Adheres strictly to the 20-step / stride-5 window configuration
        without cross-run data bleed.
        """
        modalities = self.supported_modalities
        canonical_samples: List[CanonicalSample] = []

        for run_id, run_df in run_data.items():
            n_rows = len(run_df)
            if n_rows < window_size:
                continue

            for start_idx in range(0, n_rows - window_size + 1, stride):
                end_idx = start_idx + window_size
                window_slice = run_df.iloc[start_idx:end_idx]

                start_cycle = int(window_slice["time_cycles"].iloc[0])
                end_cycle = int(window_slice["time_cycles"].iloc[-1])
                target_rul_val = float(window_slice["target_rul"].iloc[-1])

                # Extract per-modality numpy arrays of shape (T=20, C_m)
                modality_values: Dict[str, np.ndarray] = {}
                modality_mask: Dict[str, int] = {}

                for mod_name, mod_meta in modalities.items():
                    mod_array = window_slice[mod_meta.channels].to_numpy(dtype=np.float32)
                    modality_values[mod_name] = mod_array
                    modality_mask[mod_name] = 1  # Available

                sample_id = f"{self.dataset_id}_{run_id}_c{start_cycle:04d}_c{end_cycle:04d}"
                metadata = WindowMetadata(
                    dataset_id=self.dataset_id,
                    dataset_version=self.dataset_version,
                    run_id=run_id,
                    asset_id=run_id,
                    window_start=start_cycle,
                    window_end=end_cycle,
                    sampling_interval_seconds=None,
                    operational_condition=self.subdataset,
                )

                targets = Targets(
                    target_rul=target_rul_val,
                    target_fault_class=None,
                    target_binary_failure=None,
                    target_degradation=None,
                )

                sample = CanonicalSample(
                    sample_id=sample_id,
                    metadata=metadata,
                    modality_values=modality_values,
                    modality_mask=modality_mask,
                    targets=targets,
                    preprocessor_version=preprocessor_version,
                    scaler_version=scaler_version,
                )
                sample.validate(expected_window_length=window_size)
                canonical_samples.append(sample)

        return canonical_samples
