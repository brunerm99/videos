# props/nexrad.py

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from manim import DEGREES, RIGHT, UP, ManimColor, rgb_to_color
from scipy import signal
from scipy.interpolate import interp1d

from .colormaps import get_colormap

NEXRAD_CASE_S3_KEY = "2024/05/07/KTLX/KTLX20240507_023811_V06"
NEXRAD_VOLUME_DIR = Path(__file__).resolve().parent / "static" / "nexrad"
LEGACY_NEXRAD_VOLUME_DIR = (
    Path(__file__).resolve().parents[1] / "16_dual_pol" / "static" / "nexrad"
)
NEXRAD_PPI_CACHE_VERSION = 1
NEXRAD_PRODUCT_SPECS = {
    "reflectivity": {
        "field_name": "reflectivity",
        "source_field_name": "reflectivity",
        "field_key": "reflectivity_dbz",
        "aliases": ("z", "dbz", "ref"),
    },
    "velocity": {
        "field_name": "velocity",
        "source_field_name": "velocity",
        "field_key": "velocity_ms",
        "aliases": ("v", "vel"),
    },
    "zdr": {
        "field_name": "differential_reflectivity",
        "source_field_name": "differential_reflectivity",
        "field_key": "zdr_db",
        "aliases": ("z_dr", "differential_reflectivity"),
    },
    "rhohv": {
        "field_name": "cross_correlation_ratio",
        "source_field_name": "cross_correlation_ratio",
        "field_key": "rhohv",
        "aliases": ("rho_hv", "rho", "cross_correlation_ratio"),
    },
    "phidp": {
        "field_name": "differential_phase",
        "source_field_name": "differential_phase",
        "field_key": "phidp_deg",
        "aliases": ("phi_dp", "phi", "differential_phase"),
    },
    "kdp": {
        "field_name": "specific_differential_phase",
        "source_field_name": "differential_phase",
        "field_key": "kdp_deg_per_km",
        "aliases": ("k_dp", "specific_differential_phase"),
        "derived": "kdp_from_phidp",
        "units": "deg/km",
        "vmin": -2.0,
        "vmax": 8.0,
    },
}
NEXRAD_PRODUCT_ALIASES = {
    alias: product_key
    for product_key, spec in NEXRAD_PRODUCT_SPECS.items()
    for alias in (product_key, *spec.get("aliases", ()))
}
__all__ = [
    "NEXRAD_CASE_S3_KEY",
    "val_to_manim_color",
    "get_available_nexrad_products",
    "get_nexrad_product_ppi_data",
]


def _get_nexrad_archive_volume(s3_key=NEXRAD_CASE_S3_KEY):
    import urllib.request

    volume_name = Path(s3_key).name
    raw_path = NEXRAD_VOLUME_DIR / volume_name
    if raw_path.exists():
        return raw_path

    legacy_raw_path = LEGACY_NEXRAD_VOLUME_DIR / volume_name
    if legacy_raw_path.exists():
        return legacy_raw_path

    NEXRAD_VOLUME_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        f"https://unidata-nexrad-level2.s3.amazonaws.com/{s3_key}", raw_path
    )
    return raw_path


@lru_cache(maxsize=4)
def _get_nexrad_radar_cached(s3_key=NEXRAD_CASE_S3_KEY):
    import pyart

    raw_path = _get_nexrad_archive_volume(s3_key=s3_key)
    return pyart.io.read_nexrad_archive(str(raw_path))


def _safe_cache_token(value):
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in str(value)
    )


def _format_cache_number(value):
    return f"{float(value):g}".replace(".", "p")


def _get_nexrad_ppi_cache_paths(
    s3_key,
    field_name,
    sweep,
    max_range_km,
    resolution,
):
    volume_name = _safe_cache_token(Path(s3_key).name)
    field_token = _safe_cache_token(field_name)
    range_token = _format_cache_number(max_range_km)
    cache_stem = (
        f"{volume_name}_{field_token}_sw{int(sweep)}_"
        f"{range_token}km_{int(resolution)}px_v{NEXRAD_PPI_CACHE_VERSION}"
    )
    cache_base = NEXRAD_VOLUME_DIR / cache_stem
    return cache_base.with_suffix(".json"), cache_base.with_suffix(".npz")


def _get_nexrad_field_names_cache_path(s3_key):
    volume_name = _safe_cache_token(Path(s3_key).name)
    return (
        NEXRAD_VOLUME_DIR
        / f"{volume_name}_field_names_v{NEXRAD_PPI_CACHE_VERSION}.json"
    )


def _load_nexrad_field_names_cache(s3_key):
    cache_path = _get_nexrad_field_names_cache_path(s3_key)
    if not cache_path.exists():
        return None

    with cache_path.open() as cache_file:
        cache_data = json.load(cache_file)
    if cache_data.get("cache_version") != NEXRAD_PPI_CACHE_VERSION:
        return None
    return tuple(cache_data["field_names"])


def _write_nexrad_field_names_cache(s3_key, field_names):
    cache_path = _get_nexrad_field_names_cache_path(s3_key)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w") as cache_file:
        json.dump(
            {
                "cache_version": NEXRAD_PPI_CACHE_VERSION,
                "field_names": list(field_names),
            },
            cache_file,
            indent=2,
            sort_keys=True,
        )
        cache_file.write("\n")
    tmp_path.replace(cache_path)


def _get_nexrad_product_cache_key(product_key):
    return f"product_{_safe_cache_token(product_key)}"


def _load_nexrad_ppi_cache(s3_key, field_name, sweep, max_range_km, resolution):
    metadata_path, arrays_path = _get_nexrad_ppi_cache_paths(
        s3_key,
        field_name,
        sweep,
        max_range_km,
        resolution,
    )
    if not metadata_path.exists() or not arrays_path.exists():
        return None

    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    if metadata.get("cache_version") != NEXRAD_PPI_CACHE_VERSION:
        return None

    with np.load(arrays_path) as arrays:
        data = {
            "field_data": arrays["field_data"].copy(),
            "valid_mask": arrays["valid_mask"].astype(bool, copy=True),
            "azimuth_grid_deg": arrays["azimuth_grid_deg"].copy(),
            "range_grid_km": arrays["range_grid_km"].copy(),
            "x_coords_km": arrays["x_coords_km"].copy(),
            "y_coords_km": arrays["y_coords_km"].copy(),
        }

    metadata.pop("cache_version", None)
    data.update(metadata)
    return data


def _write_nexrad_ppi_cache(s3_key, field_name, sweep, max_range_km, resolution, data):
    metadata_path, arrays_path = _get_nexrad_ppi_cache_paths(
        s3_key,
        field_name,
        sweep,
        max_range_km,
        resolution,
    )
    tmp_metadata_path = metadata_path.with_name(metadata_path.name + ".tmp")
    tmp_arrays_path = arrays_path.with_name(arrays_path.name + ".tmp")

    metadata = {
        key: value for key, value in data.items() if not isinstance(value, np.ndarray)
    }
    metadata["cache_version"] = NEXRAD_PPI_CACHE_VERSION

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_arrays_path.open("wb") as arrays_file:
        np.savez(
            arrays_file,
            field_data=data["field_data"],
            valid_mask=data["valid_mask"],
            azimuth_grid_deg=data["azimuth_grid_deg"],
            range_grid_km=data["range_grid_km"],
            x_coords_km=data["x_coords_km"],
            y_coords_km=data["y_coords_km"],
        )
    with tmp_metadata_path.open("w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)
        metadata_file.write("\n")
    tmp_arrays_path.replace(arrays_path)
    tmp_metadata_path.replace(metadata_path)


def _nexrad_azimuth_delta_deg(azimuth_deg, target_deg):
    return ((azimuth_deg - target_deg + 180.0) % 360.0) - 180.0


def _nexrad_native_gate_value(
    radar, field_name, azimuth_deg, range_km, requested_sweep
):
    if field_name not in radar.fields:
        return np.nan, None

    ranges_km = radar.range["data"].astype(np.float32) / 1000.0
    range_gate = int(np.abs(ranges_km - range_km).argmin())
    requested_sweep_valid = 0 <= requested_sweep < radar.nsweeps
    requested_angle = (
        float(radar.fixed_angle["data"][requested_sweep])
        if requested_sweep_valid
        else None
    )

    candidates = []
    for sweep in range(radar.nsweeps):
        sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
        sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
        azimuths = radar.azimuth["data"][sweep_start:sweep_end].astype(np.float32)
        if azimuths.size == 0:
            continue

        azimuth_error = np.abs(_nexrad_azimuth_delta_deg(azimuths, azimuth_deg))
        ray_offset = int(np.argmin(azimuth_error))
        ray_index = sweep_start + ray_offset
        value = float(
            np.ma.filled(
                radar.fields[field_name]["data"][ray_index, range_gate], np.nan
            )
        )
        if not np.isfinite(value):
            continue

        elevation_delta = (
            abs(float(radar.fixed_angle["data"][sweep]) - requested_angle)
            if requested_angle is not None
            else float(sweep)
        )
        candidates.append(
            (
                float(azimuth_error[ray_offset]),
                elevation_delta,
                abs(sweep - requested_sweep) if requested_sweep_valid else sweep,
                value,
                {
                    "sweep": int(sweep),
                    "ray_index": int(ray_index),
                    "gate_index": int(range_gate),
                    "azimuth_deg": float(azimuths[ray_offset]),
                    "range_km": float(radar.range["data"][range_gate] / 1000.0),
                    "elevation_deg": float(radar.fixed_angle["data"][sweep]),
                },
            )
        )

    if not candidates:
        return np.nan, None

    _, _, _, value, location = min(candidates, key=lambda item: item[:3])
    return value, location


@lru_cache(maxsize=16)
def _get_nexrad_dual_pol_gate_moments_cached(
    s3_key=NEXRAD_CASE_S3_KEY,
    requested_sweep=0,
    azimuth_deg=135.0,
    range_km=88.198,
):
    from datetime import datetime

    radar = _get_nexrad_radar_cached(s3_key=s3_key)
    required_fields = (
        "reflectivity",
        "differential_reflectivity",
        "cross_correlation_ratio",
        "differential_phase",
    )
    ranges_km = radar.range["data"].astype(np.float32) / 1000.0
    range_gate = int(np.abs(ranges_km - range_km).argmin())
    requested_sweep_valid = 0 <= requested_sweep < radar.nsweeps
    requested_angle = (
        float(radar.fixed_angle["data"][requested_sweep])
        if requested_sweep_valid
        else None
    )

    candidates = []
    for sweep in range(radar.nsweeps):
        sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
        sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
        azimuths = radar.azimuth["data"][sweep_start:sweep_end].astype(np.float32)
        if azimuths.size == 0:
            continue

        azimuth_error = np.abs(_nexrad_azimuth_delta_deg(azimuths, azimuth_deg))
        ray_offset = int(np.argmin(azimuth_error))
        ray_index = sweep_start + ray_offset
        values = {
            field_name: float(
                np.ma.filled(
                    radar.fields[field_name]["data"][ray_index, range_gate], np.nan
                )
            )
            for field_name in required_fields
            if field_name in radar.fields
        }
        missing_count = sum(
            not np.isfinite(values.get(field_name, np.nan))
            for field_name in required_fields
        )
        elevation_delta = (
            abs(float(radar.fixed_angle["data"][sweep]) - requested_angle)
            if requested_angle is not None
            else float(sweep)
        )
        candidates.append(
            (
                missing_count,
                float(azimuth_error[ray_offset]),
                elevation_delta,
                abs(sweep - requested_sweep) if requested_sweep_valid else sweep,
                sweep,
                ray_index,
                values,
                {
                    "sweep": int(sweep),
                    "ray_index": int(ray_index),
                    "gate_index": int(range_gate),
                    "azimuth_deg": float(azimuths[ray_offset]),
                    "range_km": float(radar.range["data"][range_gate] / 1000.0),
                    "elevation_deg": float(radar.fixed_angle["data"][sweep]),
                },
            )
        )

    if not candidates:
        raise RuntimeError("No native NEXRAD rays were found for the requested gate")

    *_, values, location = min(candidates, key=lambda item: item[:4])
    if any(
        not np.isfinite(values.get(field_name, np.nan))
        for field_name in required_fields
    ):
        raise RuntimeError(
            "No native NEXRAD gate had all required dual-pol moments near "
            f"{azimuth_deg:.1f} deg, {range_km:.1f} km"
        )

    velocity_ms, velocity_location = _nexrad_native_gate_value(
        radar, "velocity", azimuth_deg, range_km, requested_sweep
    )
    spectrum_width_ms, _ = _nexrad_native_gate_value(
        radar, "spectrum_width", azimuth_deg, range_km, requested_sweep
    )
    nyquist_velocity_ms = np.nan
    if velocity_location is not None:
        instrument_parameters = radar.instrument_parameters or {}
        nyquist_parameter = instrument_parameters.get("nyquist_velocity")
        if nyquist_parameter is not None:
            nyquist_data = np.ma.filled(nyquist_parameter["data"], np.nan).astype(
                np.float32
            )
            velocity_sweep = velocity_location["sweep"]
            sweep_start = int(radar.sweep_start_ray_index["data"][velocity_sweep])
            sweep_end = int(radar.sweep_end_ray_index["data"][velocity_sweep]) + 1
            if len(nyquist_data) >= sweep_end:
                nyquist_velocity_ms = float(
                    np.nanmedian(nyquist_data[sweep_start:sweep_end])
                )
            elif len(nyquist_data) > velocity_sweep:
                nyquist_velocity_ms = float(nyquist_data[velocity_sweep])

    scan_time = datetime.strptime(
        radar.time["units"].split("since ", 1)[1], "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "station": str(
            radar.metadata.get("instrument_name", s3_key.split("/")[-1][:4])
        ),
        "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": str(raw_path),
        "reflectivity_dbz": values["reflectivity"],
        "zdr_db": values["differential_reflectivity"],
        "rhohv": values["cross_correlation_ratio"],
        "phidp_deg": values["differential_phase"],
        "velocity_ms": velocity_ms,
        "spectrum_width_ms": spectrum_width_ms,
        "nyquist_velocity_ms": nyquist_velocity_ms,
        **location,
    }


def _colored_complex_noise(rng, count, alpha):
    white = (
        rng.normal(size=count).astype(np.float32)
        + 1j * rng.normal(size=count).astype(np.float32)
    ) / np.sqrt(2.0)
    out = np.empty(count, dtype=np.complex64)
    out[0] = white[0]
    innovation = np.sqrt(max(0.0, 1.0 - alpha * alpha))
    for idx in range(1, count):
        out[idx] = alpha * out[idx - 1] + innovation * white[idx]
    power = np.sqrt(np.mean(np.abs(out) ** 2))
    return out / power if power > 0 else out


@lru_cache(maxsize=16)
def _build_nexrad_rhohv_timeseries_cached(
    s3_key=NEXRAD_CASE_S3_KEY,
    requested_sweep=0,
    azimuth_deg=135.0,
    range_km=88.198,
    sample_count=128,
    seed=7,
):
    moments = _get_nexrad_dual_pol_gate_moments_cached(
        s3_key=s3_key,
        requested_sweep=requested_sweep,
        azimuth_deg=azimuth_deg,
        range_km=range_km,
    )
    rng = np.random.default_rng(seed)

    rhohv = float(np.clip(moments["rhohv"], 0.0, 0.995))
    zdr_linear = 10 ** (float(np.clip(moments["zdr_db"], -6.0, 6.0)) / 10.0)
    h_power = 1.0
    v_power = 1.0 / max(zdr_linear, 1e-3)

    spectrum_width_ms = moments["spectrum_width_ms"]
    if np.isfinite(spectrum_width_ms):
        alpha = float(np.clip(0.94 - 0.055 * spectrum_width_ms, 0.35, 0.92))
    else:
        alpha = 0.82

    common = _colored_complex_noise(rng, sample_count, alpha)
    independent = _colored_complex_noise(rng, sample_count, alpha)
    pulse_index = np.arange(sample_count, dtype=np.float32)

    velocity_ms = moments["velocity_ms"]
    nyquist_velocity_ms = moments["nyquist_velocity_ms"]
    if (
        np.isfinite(velocity_ms)
        and np.isfinite(nyquist_velocity_ms)
        and nyquist_velocity_ms > 0
    ):
        phase_step = np.pi * float(
            np.clip(velocity_ms / nyquist_velocity_ms, -1.0, 1.0)
        )
    else:
        phase_step = 0.0
    doppler_phase = np.exp(1j * phase_step * pulse_index)
    differential_phase = np.exp(1j * np.deg2rad(moments["phidp_deg"]))

    h_iq = np.sqrt(h_power) * common * doppler_phase
    v_iq = (
        np.sqrt(v_power)
        * (rhohv * common + np.sqrt(max(0.0, 1.0 - rhohv * rhohv)) * independent)
        * doppler_phase
        * differential_phase
    )

    h_signal = np.real(h_iq).astype(np.float32)
    v_signal = np.real(v_iq).astype(np.float32)
    scale = max(
        float(np.max(np.abs(h_signal))),
        float(np.max(np.abs(v_signal))),
        1e-6,
    )
    h_signal = 0.88 * h_signal / scale
    v_signal = 0.88 * v_signal / scale

    return {
        "pulse_index": pulse_index,
        "h": h_signal,
        "v": v_signal,
        "moments": moments,
    }


def _get_nexrad_field_display_limits(radar, sweep, field_name, field_data):
    if field_name == "reflectivity":
        return -20.0, 75.0, "dBZ"

    if field_name == "velocity":
        nyquist_velocity = np.nan
        instrument_parameters = radar.instrument_parameters or {}
        nyquist_parameter = instrument_parameters.get("nyquist_velocity")
        if nyquist_parameter is not None:
            nyquist_data = np.ma.filled(nyquist_parameter["data"], np.nan).astype(
                np.float32
            )
            sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
            sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
            if len(nyquist_data) >= sweep_end:
                nyquist_velocity = float(
                    np.nanmedian(nyquist_data[sweep_start:sweep_end])
                )
            elif len(nyquist_data) > sweep:
                nyquist_velocity = float(nyquist_data[sweep])

        if not np.isfinite(nyquist_velocity) or nyquist_velocity <= 0:
            finite = np.abs(field_data[np.isfinite(field_data)])
            nyquist_velocity = (
                float(np.nanpercentile(finite, 99)) if finite.size else 35.0
            )

        nyquist_velocity = float(np.ceil(max(5.0, nyquist_velocity) / 5.0) * 5.0)
        return -nyquist_velocity, nyquist_velocity, "m/s"

    finite = field_data[np.isfinite(field_data)]
    if finite.size == 0:
        return 0.0, 1.0, str(radar.fields[field_name].get("units", ""))

    return (
        float(np.nanmin(finite)),
        float(np.nanmax(finite)),
        str(radar.fields[field_name].get("units", "")),
    )


@lru_cache(maxsize=16)
def _get_nexrad_ppi_data_cached(
    s3_key=NEXRAD_CASE_S3_KEY,
    sweep=6,
    max_range_km=150.0,
    resolution=1200,
    field_name="reflectivity",
):
    from datetime import datetime

    if sweep < 0:
        sweep = _get_nexrad_best_sweep_for_field(
            s3_key=s3_key,
            requested_sweep=sweep,
            field_name=field_name,
        )
    cached_data = _load_nexrad_ppi_cache(
        s3_key,
        field_name,
        sweep,
        max_range_km,
        resolution,
    )
    if cached_data is not None:
        return cached_data

    radar = _get_nexrad_radar_cached(s3_key=s3_key)
    if sweep < 0 or sweep >= radar.nsweeps:
        sweep = _get_nexrad_best_sweep_for_field(
            s3_key=s3_key,
            requested_sweep=sweep,
            field_name=field_name,
        )
        cached_data = _load_nexrad_ppi_cache(
            s3_key,
            field_name,
            sweep,
            max_range_km,
            resolution,
        )
        if cached_data is not None:
            return cached_data
    field_data = np.ma.filled(
        radar.get_field(sweep, field_name).astype(np.float32), np.nan
    )
    sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
    sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
    azimuths = radar.azimuth["data"][sweep_start:sweep_end].astype(np.float32)
    ranges_km = radar.range["data"].astype(np.float32) / 1000.0

    order = np.argsort(azimuths)
    azimuths = azimuths[order]
    field_data = field_data[order]

    gate_mask = ranges_km <= max_range_km
    ranges_km = ranges_km[gate_mask]
    field_data = field_data[:, gate_mask]

    radial_step_km = float(np.median(np.diff(ranges_km)))
    x_coords = np.linspace(-max_range_km, max_range_km, resolution, dtype=np.float32)
    y_coords = np.linspace(max_range_km, -max_range_km, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    range_grid_km = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    azimuth_grid_deg = (np.degrees(np.arctan2(grid_x, grid_y)) + 360.0) % 360.0

    azimuth_ext = np.concatenate([azimuths, [azimuths[0] + 360.0]])
    azimuth_edges = (azimuth_ext[:-1] + azimuth_ext[1:]) / 2.0
    first_edge = azimuth_edges[-1] - 360.0
    wrapped_azimuth = azimuth_grid_deg.copy()
    wrapped_azimuth[wrapped_azimuth < first_edge] += 360.0
    ray_index = np.searchsorted(azimuth_edges, wrapped_azimuth, side="right") % len(
        azimuths
    )

    range_edges = np.concatenate(
        [
            [max(0.0, ranges_km[0] - radial_step_km / 2.0)],
            ranges_km[:-1] + radial_step_km / 2.0,
            [ranges_km[-1] + radial_step_km / 2.0],
        ]
    )
    gate_index = np.searchsorted(range_edges, range_grid_km, side="right") - 1
    valid_mask = (
        (range_grid_km >= range_edges[0])
        & (range_grid_km <= max_range_km)
        & (gate_index >= 0)
        & (gate_index < len(ranges_km))
    )

    cartesian_field = np.full(range_grid_km.shape, np.nan, dtype=np.float32)
    cartesian_field[valid_mask] = field_data[
        ray_index[valid_mask], gate_index[valid_mask]
    ]
    valid_mask &= np.isfinite(cartesian_field)

    vmin, vmax, units = _get_nexrad_field_display_limits(
        radar, sweep, field_name, cartesian_field
    )

    scan_time = datetime.strptime(
        radar.time["units"].split("since ", 1)[1], "%Y-%m-%dT%H:%M:%SZ"
    )
    data = {
        "station": str(
            radar.metadata.get("instrument_name", s3_key.split("/")[-1][:4])
        ),
        "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "elevation_deg": round(float(radar.fixed_angle["data"][sweep]), 2),
        "vcp_pattern": int(radar.metadata.get("vcp_pattern", -1)),
        "sweep": int(sweep),
        "max_range_km": float(max_range_km),
        "field_name": field_name,
        "units": units,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "source": "Unidata NEXRAD Level II archive on AWS S3",
        "field_data": cartesian_field,
        "valid_mask": valid_mask,
        "azimuth_grid_deg": azimuth_grid_deg,
        "range_grid_km": range_grid_km,
        "x_coords_km": x_coords,
        "y_coords_km": y_coords,
    }
    _write_nexrad_ppi_cache(s3_key, field_name, sweep, max_range_km, resolution, data)
    return data


@lru_cache(maxsize=16)
def _get_nexrad_fixed_angles_cached(
    s3_key=NEXRAD_CASE_S3_KEY,
):
    radar = _get_nexrad_radar_cached(s3_key=s3_key)
    return tuple(float(angle) for angle in radar.fixed_angle["data"])


@lru_cache(maxsize=32)
def _get_nexrad_valid_sweeps_cached(
    s3_key=NEXRAD_CASE_S3_KEY,
    field_name="reflectivity",
):
    radar = _get_nexrad_radar_cached(s3_key=s3_key)

    if field_name not in radar.fields:
        return tuple()

    valid_sweeps = []
    for candidate_sweep in range(radar.nsweeps):
        try:
            field_data = np.ma.filled(
                radar.get_field(candidate_sweep, field_name).astype(np.float32), np.nan
            )
        except Exception:
            continue

        finite_fraction = float(np.isfinite(field_data).mean())
        if finite_fraction > 0:
            valid_sweeps.append(
                (
                    int(candidate_sweep),
                    float(radar.fixed_angle["data"][candidate_sweep]),
                    finite_fraction,
                )
            )

    return tuple(valid_sweeps)


@lru_cache(maxsize=16)
def _get_nexrad_field_names_cached(s3_key=NEXRAD_CASE_S3_KEY):
    cached_field_names = _load_nexrad_field_names_cache(s3_key)
    if cached_field_names is not None:
        return cached_field_names

    radar = _get_nexrad_radar_cached(s3_key=s3_key)
    field_names = tuple(sorted(radar.fields.keys()))
    _write_nexrad_field_names_cache(s3_key, field_names)
    return field_names


def _normalize_nexrad_product(product):
    product_key = str(product).strip().lower().replace("-", "_")
    try:
        return NEXRAD_PRODUCT_ALIASES[product_key]
    except KeyError as exc:
        available = ", ".join(sorted(NEXRAD_PRODUCT_SPECS))
        raise ValueError(
            f"Unknown NEXRAD product '{product}'. Available products: {available}"
        ) from exc


def _get_nexrad_best_sweep_for_field(
    s3_key=NEXRAD_CASE_S3_KEY,
    requested_sweep=6,
    field_name="reflectivity",
):
    valid_sweeps = _get_nexrad_valid_sweeps_cached(
        s3_key=s3_key,
        field_name=field_name,
    )
    if not valid_sweeps:
        raise RuntimeError(f"No valid sweeps found for NEXRAD field '{field_name}'")

    fixed_angles = _get_nexrad_fixed_angles_cached(s3_key=s3_key)
    requested_angle = None
    if 0 <= requested_sweep < len(fixed_angles):
        requested_angle = fixed_angles[requested_sweep]

    if requested_angle is None:
        best_sweep, _, _ = min(
            valid_sweeps,
            key=lambda item: (item[1], item[0], -item[2]),
        )
    else:
        best_sweep, _, _ = min(
            valid_sweeps,
            key=lambda item: (
                abs(item[1] - requested_angle),
                abs(item[0] - requested_sweep),
                -item[2],
            ),
        )
    return best_sweep


def _copy_metadata(data):
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }


def get_available_nexrad_products(s3_key=NEXRAD_CASE_S3_KEY):
    field_names = set(_get_nexrad_field_names_cached(s3_key=s3_key))
    available = {}
    for product_key, spec in NEXRAD_PRODUCT_SPECS.items():
        source_field_name = spec["source_field_name"]
        if source_field_name not in field_names:
            continue

        available[product_key] = {
            "field_name": spec["field_name"],
            "source_field_name": source_field_name,
            "field_key": spec["field_key"],
            "aliases": spec.get("aliases", ()),
            "derived": bool(spec.get("derived")),
        }
    return available


def get_nexrad_product_ppi_data(
    product,
    s3_key=NEXRAD_CASE_S3_KEY,
    sweep=6,
    max_range_km=150.0,
    resolution=1200,
):
    product_key = _normalize_nexrad_product(product)
    spec = NEXRAD_PRODUCT_SPECS[product_key]
    field_key = spec["field_key"]
    product_cache_key = _get_nexrad_product_cache_key(product_key)
    cached_data = _load_nexrad_ppi_cache(
        s3_key,
        product_cache_key,
        sweep,
        max_range_km,
        resolution,
    )
    if cached_data is not None:
        cached_data[field_key] = cached_data.pop("field_data")
        cached_data["product"] = product_key
        cached_data["field_key"] = field_key
        return cached_data

    available_products = get_available_nexrad_products(s3_key=s3_key)
    if product_key not in available_products:
        available = ", ".join(sorted(available_products)) or "none"
        raise RuntimeError(
            f"NEXRAD product '{product_key}' is not available in this volume. "
            f"Available products: {available}"
        )

    if spec.get("derived") == "kdp_from_phidp":
        data_copy = _get_nexrad_kdp_ppi_data_cached(
            s3_key=s3_key,
            sweep=sweep,
            max_range_km=max_range_km,
            resolution=resolution,
            product_key=product_key,
            field_key=field_key,
        )
        cache_data = data_copy.copy()
        cache_data["field_data"] = cache_data[field_key]
        _write_nexrad_ppi_cache(
            s3_key,
            product_cache_key,
            sweep,
            max_range_km,
            resolution,
            cache_data,
        )
        return data_copy

    source_field_name = spec["source_field_name"]
    selected_sweep = _get_nexrad_best_sweep_for_field(
        s3_key=s3_key,
        requested_sweep=sweep,
        field_name=source_field_name,
    )
    data_copy = _copy_metadata(
        _get_nexrad_ppi_data_cached(
            s3_key=s3_key,
            sweep=selected_sweep,
            max_range_km=max_range_km,
            resolution=resolution,
            field_name=source_field_name,
        )
    )
    data_copy[field_key] = data_copy.pop("field_data")
    data_copy["product"] = product_key
    data_copy["field_key"] = field_key
    cache_data = data_copy.copy()
    cache_data["field_data"] = cache_data[field_key]
    _write_nexrad_ppi_cache(
        s3_key,
        product_cache_key,
        sweep,
        max_range_km,
        resolution,
        cache_data,
    )
    return data_copy


def _derive_kdp_from_phidp_sweep(phidp_deg, ranges_km):
    kdp_deg_per_km = np.full(phidp_deg.shape, np.nan, dtype=np.float32)
    for ray_idx, phidp_ray in enumerate(phidp_deg):
        finite_gate_idx = np.flatnonzero(np.isfinite(phidp_ray))
        if finite_gate_idx.size < 5:
            continue

        gate_runs = np.split(
            finite_gate_idx,
            np.flatnonzero(np.diff(finite_gate_idx) > 1) + 1,
        )
        for gate_idx in gate_runs:
            if gate_idx.size < 5:
                continue

            phidp_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(phidp_ray[gate_idx])))
            window_len = min(
                31,
                gate_idx.size if gate_idx.size % 2 else gate_idx.size - 1,
            )
            if window_len >= 5:
                phidp_unwrapped = signal.savgol_filter(
                    phidp_unwrapped,
                    window_length=window_len,
                    polyorder=2,
                    mode="interp",
                )

            dphidp_dr = np.gradient(phidp_unwrapped, ranges_km[gate_idx])
            if dphidp_dr.size >= 9:
                dphidp_dr = signal.medfilt(dphidp_dr, kernel_size=9)
            elif dphidp_dr.size >= 5:
                dphidp_dr = signal.medfilt(dphidp_dr, kernel_size=5)
            kdp_deg_per_km[ray_idx, gate_idx] = 0.5 * dphidp_dr

    return kdp_deg_per_km


@lru_cache(maxsize=16)
def _get_nexrad_kdp_ppi_data_cached(
    s3_key=NEXRAD_CASE_S3_KEY,
    sweep=6,
    max_range_km=150.0,
    resolution=1200,
    product_key="kdp",
    field_key="kdp_deg_per_km",
):
    from datetime import datetime

    field_name = "differential_phase"
    sweep = _get_nexrad_best_sweep_for_field(
        s3_key=s3_key,
        requested_sweep=sweep,
        field_name=field_name,
    )
    cached_data = _load_nexrad_ppi_cache(
        s3_key,
        product_key,
        sweep,
        max_range_km,
        resolution,
    )
    if cached_data is not None:
        cached_data[field_key] = cached_data.pop("field_data")
        cached_data["product"] = product_key
        cached_data["field_key"] = field_key
        return cached_data

    radar = _get_nexrad_radar_cached(s3_key=s3_key)
    phidp_data = np.ma.filled(
        radar.get_field(sweep, field_name).astype(np.float32), np.nan
    )
    sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
    sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
    azimuths = radar.azimuth["data"][sweep_start:sweep_end].astype(np.float32)
    ranges_km = radar.range["data"].astype(np.float32) / 1000.0

    order = np.argsort(azimuths)
    azimuths = azimuths[order]
    phidp_data = phidp_data[order]

    gate_mask = ranges_km <= max_range_km
    ranges_km = ranges_km[gate_mask]
    phidp_data = phidp_data[:, gate_mask]
    kdp_data = _derive_kdp_from_phidp_sweep(phidp_data, ranges_km)

    radial_step_km = float(np.median(np.diff(ranges_km)))
    x_coords = np.linspace(-max_range_km, max_range_km, resolution, dtype=np.float32)
    y_coords = np.linspace(max_range_km, -max_range_km, resolution, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    range_grid_km = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    azimuth_grid_deg = (np.degrees(np.arctan2(grid_x, grid_y)) + 360.0) % 360.0

    azimuth_ext = np.concatenate([azimuths, [azimuths[0] + 360.0]])
    azimuth_edges = (azimuth_ext[:-1] + azimuth_ext[1:]) / 2.0
    first_edge = azimuth_edges[-1] - 360.0
    wrapped_azimuth = azimuth_grid_deg.copy()
    wrapped_azimuth[wrapped_azimuth < first_edge] += 360.0
    ray_index = np.searchsorted(azimuth_edges, wrapped_azimuth, side="right") % len(
        azimuths
    )

    range_edges = np.concatenate(
        [
            [max(0.0, ranges_km[0] - radial_step_km / 2.0)],
            ranges_km[:-1] + radial_step_km / 2.0,
            [ranges_km[-1] + radial_step_km / 2.0],
        ]
    )
    gate_index = np.searchsorted(range_edges, range_grid_km, side="right") - 1
    valid_mask = (
        (range_grid_km >= range_edges[0])
        & (range_grid_km <= max_range_km)
        & (gate_index >= 0)
        & (gate_index < len(ranges_km))
    )

    cartesian_field = np.full(range_grid_km.shape, np.nan, dtype=np.float32)
    cartesian_field[valid_mask] = kdp_data[
        ray_index[valid_mask], gate_index[valid_mask]
    ]
    valid_mask &= np.isfinite(cartesian_field)

    scan_time = datetime.strptime(
        radar.time["units"].split("since ", 1)[1], "%Y-%m-%dT%H:%M:%SZ"
    )
    data = {
        "station": str(
            radar.metadata.get("instrument_name", s3_key.split("/")[-1][:4])
        ),
        "scan_time": scan_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "elevation_deg": round(float(radar.fixed_angle["data"][sweep]), 2),
        "vcp_pattern": int(radar.metadata.get("vcp_pattern", -1)),
        "sweep": int(sweep),
        "max_range_km": float(max_range_km),
        "field_name": "specific_differential_phase",
        "units": "deg/km",
        "vmin": -2.0,
        "vmax": 8.0,
        "source": "Unidata NEXRAD Level II archive on AWS S3",
        "product": product_key,
        "field_key": field_key,
        "field_data": cartesian_field,
        "valid_mask": valid_mask,
        "azimuth_grid_deg": azimuth_grid_deg,
        "range_grid_km": range_grid_km,
        "x_coords_km": x_coords,
        "y_coords_km": y_coords,
    }
    _write_nexrad_ppi_cache(s3_key, product_key, sweep, max_range_km, resolution, data)
    data[field_key] = data.pop("field_data")
    return data


def _get_nexrad_colormap(cmap_name, fallback_name):
    return get_colormap(cmap_name, fallback_name)


def val_to_manim_color(
    val,
    vmin=0.0,
    vmax=75.0,
    cmap_name="NWSRef",
    fallback_cmap_name="viridis",
    nan_color: ManimColor | None = None,
    nan_val=np.nan,
):
    val = float(np.asarray(val).squeeze())
    if np.isnan(val) or val == nan_val:
        if nan_color is not None:
            return nan_color
        return None
    if not np.isfinite(val):
        val = vmin
    normalized = np.clip((float(val) - vmin) / (vmax - vmin), 0.0, 1.0)
    rgba = _get_nexrad_colormap(cmap_name, fallback_cmap_name)(normalized)
    return rgb_to_color((np.asarray(rgba[:3]) * 255).astype(np.uint8))


def _reflectivity_dbz_to_rgba(
    reflectivity_dbz,
    valid_mask=None,
    vmin=-20.0,
    vmax=75.0,
    cmap_name="NWSRef",
    min_alpha=0.45,
):
    if valid_mask is None:
        valid_mask = np.isfinite(reflectivity_dbz)

    normalized = np.clip((reflectivity_dbz - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_nexrad_colormap(cmap_name, "turbo")
    rgba = (cmap(normalized) * 255).astype(np.uint8)

    alpha = np.zeros(reflectivity_dbz.shape, dtype=np.uint8)
    if np.any(valid_mask):
        alpha_strength = min_alpha + (1.0 - min_alpha) * np.clip(
            (reflectivity_dbz[valid_mask] - 2.0) / 30.0, 0.0, 1.0
        )
        alpha[valid_mask] = np.uint8(255 * alpha_strength)
    rgba[..., 3] = alpha
    return rgba


def _velocity_ms_to_rgba(
    velocity_ms,
    valid_mask=None,
    vmin=-35.0,
    vmax=35.0,
    cmap_name="NWSVel",
    min_alpha=0.55,
):
    if valid_mask is None:
        valid_mask = np.isfinite(velocity_ms)

    normalized = np.clip((velocity_ms - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_nexrad_colormap(cmap_name, "coolwarm")
    rgba = (cmap(normalized) * 255).astype(np.uint8)

    alpha = np.zeros(velocity_ms.shape, dtype=np.uint8)
    if np.any(valid_mask):
        max_velocity = max(abs(vmin), abs(vmax), 1.0)
        alpha_strength = min_alpha + (1.0 - min_alpha) * np.clip(
            np.abs(velocity_ms[valid_mask]) / max_velocity, 0.0, 1.0
        )
        alpha[valid_mask] = np.uint8(255 * alpha_strength)
    rgba[..., 3] = alpha
    return rgba


def _nexrad_field_to_rgba(
    field_data,
    valid_mask=None,
    vmin=0.0,
    vmax=1.0,
    cmap_name="viridis",
    fallback_cmap_name="viridis",
    min_alpha=0.72,
):
    if valid_mask is None:
        valid_mask = np.isfinite(field_data)

    normalized = np.clip((field_data - vmin) / (vmax - vmin), 0.0, 1.0)
    cmap = _get_nexrad_colormap(cmap_name, fallback_cmap_name)
    rgba = (cmap(normalized) * 255).astype(np.uint8)

    alpha = np.zeros(field_data.shape, dtype=np.uint8)
    alpha[valid_mask] = np.uint8(255 * min_alpha)
    rgba[..., 3] = alpha
    return rgba


def _nexrad_scan_reveal_mask(azimuth_grid_deg, sweep_azimuth_deg, trail_deg=45.0):
    azimuth_lag_deg = (sweep_azimuth_deg - azimuth_grid_deg) % 360.0
    return azimuth_lag_deg <= trail_deg


def _nexrad_sweep_progress_mask(azimuth_grid_deg, sweep_azimuth_deg):
    return azimuth_grid_deg <= sweep_azimuth_deg


def _blend_radar_rgba(base_rgba, next_rgba, update_mask):
    return np.where(update_mask[..., None], next_rgba, base_rgba)


def _build_nexrad_colorbar_image(
    height_px=900,
    width_px=56,
    vmin=-20.0,
    vmax=75.0,
    cmap_name="NWSRef",
    fallback_cmap_name="turbo",
):
    gradient = np.linspace(vmax, vmin, height_px, dtype=np.float32)[:, None]
    cmap = _get_nexrad_colormap(cmap_name, fallback_cmap_name)
    rgba = (cmap((gradient - vmin) / (vmax - vmin)) * 255).astype(np.uint8)
    rgba = np.repeat(rgba, width_px, axis=1)
    rgba[..., 3] = 255
    return rgba


def _polar_ppi_point(center, plot_radius, max_range_km, azimuth_deg, range_km):
    scaled_radius = plot_radius * (range_km / max_range_km)
    theta = azimuth_deg * DEGREES
    return (
        center
        + RIGHT * (scaled_radius * np.sin(theta))
        + UP * (scaled_radius * np.cos(theta))
    )


def slice_updater(metadata, scan_progress, beam_width_deg):
    max_range = metadata["max_range_km"]
    r = np.linspace(0, max_range, 1200)
    x = r * np.sin(np.deg2rad(~scan_progress - beam_width_deg / 2))
    y = r * np.cos(np.deg2rad(~scan_progress - beam_width_deg / 2))
    x_idx = np.abs(metadata["x_coords_km"][None, :] - x[:, None]).argmin(axis=1)
    y_idx = np.abs(metadata["y_coords_km"][None, :] - y[:, None]).argmin(axis=1)
    z = metadata["reflectivity_dbz"][y_idx, x_idx]
    z = np.nan_to_num(z, nan=0)
    order = np.argsort(r)
    r = r[order]
    z = z[order]

    return interp1d(r, z, fill_value="extrapolate")


def get_ppi_slice(metadata, ax, scan_progress, beam_width_deg, color, opacity):
    def updater():
        z_func_scan = slice_updater(metadata, scan_progress, beam_width_deg)

        nexrad_scan_plot = ax.plot(
            z_func_scan,
            x_range=[2.5, 150, 1 / 200],
            color=color,
            stroke_opacity=~opacity,
        )
        return nexrad_scan_plot

    return updater
