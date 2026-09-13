"""Dataset provenance recording and verification module for MESH.

Tracks SHA-256 hashes, source URLs, licensing, physical system descriptions,
and acquisition metadata per docs/data/DATA_ACQUISITION.md and docs/data/DATA_PROVENANCE.md.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class FileChecksum:
    relative_path: str
    file_size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetProvenanceRecord:
    dataset_id: str
    dataset_version: str
    display_name: str
    source_url: str
    download_url: str
    download_timestamp: str
    license_or_terms: str
    machine_or_equipment_type: str
    sensor_list: List[str]
    units: Dict[str, str]
    sampling_information: str
    run_identifier_definition: str
    targets_available: List[str]
    natural_missingness: str
    notes_and_limitations: str
    files: List[FileChecksum]


def compute_file_sha256(filepath: Path) -> str:
    """Computes the SHA-256 hash of a file using buffered reading."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def scan_dataset_files(directory: Path) -> List[FileChecksum]:
    """Scans all files within a directory and computes their SHA-256 checksums."""
    checksums = []
    if not directory.exists():
        return checksums

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            rel_path = str(path.relative_to(directory.parent.parent)).replace("\\", "/")
            checksums.append(
                FileChecksum(
                    relative_path=rel_path,
                    file_size_bytes=path.stat().st_size,
                    sha256=compute_file_sha256(path),
                )
            )
    return checksums


def generate_provenance_catalog(
    raw_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Builds and serializes a provenance catalog for all raw datasets present in data/raw/."""
    cmapss_dir = raw_dir / "cmapss"
    ai4i_dir = raw_dir / "ai4i2020"

    cmapss_files = scan_dataset_files(cmapss_dir)
    ai4i_files = scan_dataset_files(ai4i_dir)

    # C-MAPSS metadata provenance:
    # - Sensors & units: Directly verified from 'Damage Propagation Modeling.pdf' Table 2 (Page 3).
    # - Trajectory & scenario: Directly verified from 'readme.txt'.
    # - Source URL & paper citation: From 'readme.txt' and paper Reference [1].
    # - License: Unverified in raw txt files; noted as open PHM challenge research data.
    cmapss_record = DatasetProvenanceRecord(
        dataset_id="nasa_cmapss",
        dataset_version="1.0.0",
        display_name="NASA C-MAPSS Turbofan Engine Degradation Simulation Dataset",
        source_url="http://ti.arc.nasa.gov/projects/data_prognostics (Paper Ref [1]); cataloged at https://data.nasa.gov/dataset/C-MAPSS-Aircraft-Engine-Simulator-Data/xaut-bemq",
        download_url="https://phm-datasets.s3.amazonaws.com/NASA/CMAPSSData.zip (Workspace download)",
        download_timestamp="2026-09-13T21:00:51Z",
        license_or_terms="Unverified in raw text files. Cited as public PHM08 Challenge benchmark data created under NASA IVHM program (Saxena et al., 2008).",
        machine_or_equipment_type="Commercial Modular Aero-Propulsion System Simulation (90,000 lb thrust class turbofan engine)",
        sensor_list=[
            "s_1 (T2: Total temperature at fan inlet) [°R]",
            "s_2 (T24: Total temperature at LPC outlet) [°R]",
            "s_3 (T30: Total temperature at HPC outlet) [°R]",
            "s_4 (T50: Total temperature at LPT outlet) [°R]",
            "s_5 (P2: Pressure at fan inlet) [psia]",
            "s_6 (P15: Total pressure in bypass-duct) [psia]",
            "s_7 (P30: Total pressure at HPC outlet) [psia]",
            "s_8 (Nf: Physical fan speed) [rpm]",
            "s_9 (Nc: Physical core speed) [rpm]",
            "s_10 (epr: Engine pressure ratio P50/P2) [--]",
            "s_11 (Ps30: Static pressure at HPC outlet) [psia]",
            "s_12 (phi: Ratio of fuel flow to Ps30) [pps/psi]",
            "s_13 (NRf: Corrected fan speed) [rpm]",
            "s_14 (NRc: Corrected core speed) [rpm]",
            "s_15 (BPR: Bypass Ratio) [--]",
            "s_16 (farB: Burner fuel-air ratio) [--]",
            "s_17 (htBleed: Bleed Enthalpy) [--]",
            "s_18 (Nf_dmd: Demanded fan speed) [rpm]",
            "s_19 (PCNfR_dmd: Demanded corrected fan speed) [rpm]",
            "s_20 (W31: HPT coolant bleed) [lbm/s]",
            "s_21 (W32: LPT coolant bleed) [lbm/s]",
        ],
        units={
            "temperatures (T2, T24, T30, T50)": "°R (Rankine)",
            "pressures (P2, P15, P30, Ps30)": "psia",
            "speeds (Nf, Nc, NRf, NRc, Nf_dmd, PCNfR_dmd)": "rpm",
            "fuel_flow_ratio (phi)": "pps/psi",
            "ratios (epr, BPR, farB, htBleed)": "dimensionless",
            "coolant_bleeds (W31, W32)": "lbm/s",
        },
        sampling_information="Discrete operational flight cycles (1 snapshot per flight cycle at cruise steady-state per paper section V)",
        run_identifier_definition="unit_number (identifies individual engine asset in fleet, column 1 in txt files)",
        targets_available=["RUL (Remaining Useful Life in operational flight cycles until failure threshold)"],
        natural_missingness="None (all 21 sensor columns and 3 setting columns present in all rows; sensors 1, 5, 10, 16, 18, 19 constant in FD001 due to sea-level condition)",
        notes_and_limitations="Physics-inspired thermo-dynamical simulation model output, not physical flight recorder telemetry (verified from Damage Propagation Modeling.pdf Section III-A). Sea-level sets (FD001, FD003) contain constant sensor columns.",
        files=cmapss_files,
    )

    # AI4I 2020 metadata provenance:
    # - Columns: Verified directly from raw 'ai4i2020.csv' header.
    # - Source URL: Cited directly from repo doc 'docs/data/DATASET_STRATEGY.md' line 59.
    # - License: Unverified in CSV file; UCI website lists CC BY 4.0, but no license text is bundled in the download.
    ai4i_record = DatasetProvenanceRecord(
        dataset_id="ai4i2020",
        dataset_version="1.0.0",
        display_name="AI4I 2020 Predictive Maintenance Dataset (Synthetic Benchmark)",
        source_url="https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance (cited in docs/data/DATASET_STRATEGY.md:L59)",
        download_url="https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
        download_timestamp="2026-09-13T20:59:51Z",
        license_or_terms="Unverified in raw CSV (UCI web catalog lists CC BY 4.0, but no license file is included in raw archive).",
        machine_or_equipment_type="Synthetic milling machine process model (10,000 generated data points)",
        sensor_list=[
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ],
        units={
            "Air temperature": "K",
            "Process temperature": "K",
            "Rotational speed": "rpm",
            "Torque": "Nm",
            "Tool wear": "min",
        },
        sampling_information="10,000 synthetic rows representing simulated milling machine snapshots",
        run_identifier_definition="Product ID / UDI sequence (single continuous synthetic record sequence)",
        targets_available=[
            "Machine failure (binary, column 9)",
            "TWF (Tool Wear Failure, column 10)",
            "HDF (Heat Dissipation Failure, column 11)",
            "PWF (Power Failure, column 12)",
            "OSF (Overstrain Failure, column 13)",
            "RNF (Random Failure, column 14)",
        ],
        natural_missingness="None (all columns populated across 10,000 rows)",
        notes_and_limitations="Explicitly documented as synthetic in MESH docs (README.md Section 5, docs/data/DATASET_STRATEGY.md:L17, docs/MESH_DATASET_DECISION_MATRIX.md:L11). In-scope strictly for data loader verification, API sanity testing, and classical baseline comparisons. Must not be claimed as real industrial evidence.",
        files=ai4i_files,
    )

    catalog = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            "nasa_cmapss": asdict(cmapss_record),
            "ai4i2020": asdict(ai4i_record),
        },
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2)

    return catalog


def run_provenance_from_config(
    config: Dict[str, Any],
    root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Scans raw data files and generates provenance tracking catalog per config."""
    if root_dir is None:
        root_dir = Path.cwd()

    paths_cfg = config["paths"]
    raw_dir = root_dir / paths_cfg["raw_dir"]
    output_path = root_dir / paths_cfg["provenance_output"]

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    # Verify primary configured raw files exist and have non-zero bytes
    train_file = root_dir / paths_cfg["train_file"]
    if not train_file.exists():
        raise FileNotFoundError(f"Required raw data file missing: {train_file}")
    if train_file.stat().st_size == 0:
        raise ValueError(f"Raw data file is empty (0 bytes): {train_file}")

    if paths_cfg.get("test_file"):
        test_file = root_dir / paths_cfg["test_file"]
        if not test_file.exists():
            raise FileNotFoundError(f"Required raw test file missing: {test_file}")
        if test_file.stat().st_size == 0:
            raise ValueError(f"Raw test file is empty (0 bytes): {test_file}")

    if paths_cfg.get("rul_file"):
        rul_file = root_dir / paths_cfg["rul_file"]
        if not rul_file.exists():
            raise FileNotFoundError(f"Required raw RUL file missing: {rul_file}")
        if rul_file.stat().st_size == 0:
            raise ValueError(f"Raw RUL file is empty (0 bytes): {rul_file}")

    # Generate full raw directory catalog
    raw_root = root_dir / "data" / "raw"
    return generate_provenance_catalog(raw_root, output_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_directory = base_dir / "data" / "raw"
    catalog_path = base_dir / "data" / "raw" / "provenance.json"
    catalog_result = generate_provenance_catalog(raw_directory, catalog_path)
    print(f"Provenance catalog generated with {len(catalog_result['datasets'])} datasets.")
    for ds_id, data in catalog_result["datasets"].items():
        print(f"  - {ds_id}: {len(data['files'])} files tracked")
