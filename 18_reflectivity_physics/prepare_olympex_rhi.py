import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyart


SOURCE_URL = (
    "https://pmm-gv.gsfc.nasa.gov/pub/NPOL/OLYMPEX/rhi_a/2015/1113/"
    "np1151113020003.RAWD75S.gz"
)
OUTPUT_PATH = Path(__file__).parent / "static" / "olympex_rhi_20151113_020003_az210.npz"
MIN_REFLECTIVITY_DBZ = 10.0
MIN_NCP = 0.3
MIN_RHOHV = 0.8


def prepare_rhi(source_path, output_path=OUTPUT_PATH):
    radar = pyart.io.read_sigmet(source_path)
    if radar.scan_type != "rhi":
        raise ValueError("Expected a native RHI volume")
    sweep = int(np.argmin(np.abs(radar.fixed_angle["data"] - 210)))
    ray_slice = radar.get_slice(sweep)
    elevation_deg = radar.elevation["data"][ray_slice]
    if np.any(np.diff(elevation_deg) <= 0):
        raise ValueError("Expected ascending native elevation rays")
    ranges_km = radar.range["data"] / 1000
    gate_count = int(np.searchsorted(ranges_km, np.hypot(32.8, 8))) + 1
    reflectivity = radar.fields["reflectivity"]["data"][ray_slice, :gate_count]
    gate_filter = pyart.filters.moment_based_gate_filter(
        radar.extract_sweeps([sweep]),
        min_ncp=MIN_NCP,
        min_rhv=MIN_RHOHV,
        min_refl=MIN_REFLECTIVITY_DBZ,
        max_refl=None,
    )
    x, y, z = radar.get_gate_x_y_z(sweep, edges=True)
    with source_path.open("rb") as source:
        source_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        reflectivity_dbz=reflectivity.filled(np.nan).astype(np.float32),
        valid_mask=gate_filter.gate_included[:, :gate_count],
        gate_filter_min_reflectivity_dbz=MIN_REFLECTIVITY_DBZ,
        gate_filter_min_ncp=MIN_NCP,
        gate_filter_min_rhohv=MIN_RHOHV,
        ranges_km=ranges_km[:gate_count].astype(np.float32),
        elevation_deg=elevation_deg.astype(np.float32),
        ground_edges_km=(np.hypot(x, y)[:, :gate_count + 1] / 1000).astype(np.float32),
        height_edges_km=(z[:, :gate_count + 1] / 1000).astype(np.float32),
        azimuth_deg=float(radar.fixed_angle["data"][sweep]),
        radar_altitude_m=float(radar.altitude["data"][0]),
        source_url=SOURCE_URL,
        source_sha256=source_sha256,
        scan_type=radar.scan_type,
        sweep_index=sweep,
        time_units=radar.time["units"],
        ray_time_seconds=radar.time["data"][ray_slice],
        display_range_km=32.8,
        display_top_km=8.0,
        vmin=10.0,
        vmax=50.0,
        cmap_name="NWSRef",
    )
    print(f"Saved native RHI sweep {sweep}: {reflectivity.shape} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    prepare_rhi(args.source_path, args.output)
