from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .props.ppi import polar_to_cartesian


NEXRAD_ARCHIVE_URL = "https://unidata-nexrad-level2.s3.amazonaws.com"
NEXRAD_CASE_S3_KEY = "2024/05/07/KTLX/KTLX20240507_023811_V06"
NEXRAD_CACHE_VERSION = 3
NEXRAD_PRODUCT_SPECS = {
    "reflectivity": {
        "field_name": "reflectivity",
        "field_key": "reflectivity_dbz",
        "aliases": ("z", "dbz", "ref"),
        "vmin": -20.0,
        "vmax": 75.0,
        "units": "dBZ",
        "cmap_name": "NWSRef",
        "fallback_cmap_name": "turbo",
    },
    "velocity": {
        "field_name": "velocity",
        "field_key": "velocity_ms",
        "aliases": ("v", "vel"),
        "units": "m/s",
        "cmap_name": "NWSVel",
        "fallback_cmap_name": "coolwarm",
    },
    "zdr": {
        "field_name": "differential_reflectivity",
        "field_key": "zdr_db",
        "aliases": ("z_dr", "differential_reflectivity"),
        "vmin": -8.0,
        "vmax": 8.0,
        "units": "dB",
        "cmap_name": "Spectral_r",
        "fallback_cmap_name": "coolwarm",
    },
    "rhohv": {
        "field_name": "cross_correlation_ratio",
        "field_key": "rhohv",
        "aliases": ("rho_hv", "rho", "cross_correlation_ratio"),
        "vmin": 0.7,
        "vmax": 1.05,
        "units": "",
        "cmap_name": "NWS_CC",
        "fallback_cmap_name": "viridis",
    },
    "phidp": {
        "field_name": "differential_phase",
        "field_key": "phidp_deg",
        "aliases": ("phi_dp", "phi", "differential_phase"),
        "vmin": 0.0,
        "vmax": 360.0,
        "units": "degrees",
        "cmap_name": "twilight",
        "fallback_cmap_name": "hsv",
    },
    "kdp": {
        "field_name": "specific_differential_phase",
        "source_field_name": "differential_phase",
        "field_key": "kdp_deg_per_km",
        "aliases": ("k_dp", "specific_differential_phase"),
        "derived": "kdp_from_phidp",
        "vmin": -2.0,
        "vmax": 8.0,
        "units": "deg/km",
        "cmap_name": "NWSRef",
        "fallback_cmap_name": "turbo",
    },
}
NEXRAD_PRODUCT_ALIASES = {
    alias: product
    for product, spec in NEXRAD_PRODUCT_SPECS.items()
    for alias in (product, *spec["aliases"])
}


__all__ = [
    "NEXRAD_ARCHIVE_URL",
    "NEXRAD_CASE_S3_KEY",
    "NEXRAD_PRODUCT_SPECS",
    "fetch_nexrad_volume",
    "get_available_nexrad_products",
    "get_nexrad_product_ppi_data",
    "nexrad_ppi_from_radar",
    "read_nexrad_volume",
]


def _default_cache_dir():
    configured = os.getenv("MELP_NEXRAD_CACHE")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "melp" / "nexrad"


def _cache_dir(cache_dir=None):
    if cache_dir is None:
        return _default_cache_dir()
    return Path(cache_dir).expanduser()


def _validate_s3_key(s3_key):
    s3_key = str(s3_key).strip().lstrip("/")
    parts = Path(s3_key).parts
    if not s3_key or ".." in parts or Path(s3_key).name in ("", "."):
        raise ValueError(f"Invalid NEXRAD S3 key: {s3_key!r}")
    return s3_key


def fetch_nexrad_volume(
    s3_key=NEXRAD_CASE_S3_KEY,
    *,
    cache_dir=None,
    force=False,
):
    s3_key = _validate_s3_key(s3_key)
    raw_dir = _cache_dir(cache_dir) / "raw"
    raw_path = raw_dir.joinpath(*Path(s3_key).parts)
    if raw_path.is_file() and raw_path.stat().st_size > 0 and not force:
        return raw_path

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = raw_path.with_name(
        f"{raw_path.name}.{os.getpid()}.{uuid.uuid4().hex}.part"
    )
    url = f"{NEXRAD_ARCHIVE_URL}/{s3_key}"
    try:
        urllib.request.urlretrieve(url, partial_path)
        if not partial_path.is_file() or partial_path.stat().st_size == 0:
            raise RuntimeError(f"NEXRAD download from {url} produced an empty file.")
        partial_path.replace(raw_path)
    finally:
        partial_path.unlink(missing_ok=True)
    return raw_path


def _import_pyart():
    try:
        import pyart
    except ImportError as exc:
        raise ImportError(
            "Reading NEXRAD Level-II files requires Py-ART. Install the "
            "optional dependency with `uv pip install arm-pyart` in a "
            "Python version supported by Py-ART."
        ) from exc
    return pyart


def _read_nexrad_path(volume_path):
    return _import_pyart().io.read_nexrad_archive(
        volume_path,
        linear_interp=False,
    )


def read_nexrad_volume(
    volume_path=None,
    *,
    s3_key=NEXRAD_CASE_S3_KEY,
    cache_dir=None,
    force_download=False,
):
    if volume_path is None:
        volume_path = fetch_nexrad_volume(
            s3_key,
            cache_dir=cache_dir,
            force=force_download,
        )
    volume_path = Path(volume_path).expanduser()
    if not volume_path.is_file():
        raise FileNotFoundError(f"NEXRAD volume not found: {volume_path}")
    return _read_nexrad_path(str(volume_path.resolve()))


def _normalize_product(product):
    key = str(product).strip().lower().replace("-", "_")
    try:
        return NEXRAD_PRODUCT_ALIASES[key]
    except KeyError as exc:
        available = ", ".join(sorted(NEXRAD_PRODUCT_SPECS))
        raise ValueError(
            f"Unknown NEXRAD product {product!r}. Available products: {available}."
        ) from exc


def _resolve_product_source(spec, radar_fields):
    field_name = spec["field_name"]
    if field_name in radar_fields:
        return field_name, None
    source_field_name = spec.get("source_field_name")
    if source_field_name in radar_fields:
        return source_field_name, spec.get("derived")
    return None, None


def _available_sweeps(radar, field_name):
    if field_name not in radar.fields:
        return []

    valid = []
    for sweep in range(int(radar.nsweeps)):
        try:
            values = np.ma.filled(
                radar.get_field(sweep, field_name),
                np.nan,
            ).astype(np.float32)
        except Exception:
            continue
        finite_fraction = float(np.isfinite(values).mean())
        if finite_fraction > 0:
            angle = float(radar.fixed_angle["data"][sweep])
            valid.append((sweep, angle, finite_fraction))
    return valid


def _select_sweep(radar, field_name, requested_sweep):
    available = _available_sweeps(radar, field_name)
    if not available:
        raise RuntimeError(f"No valid sweeps contain NEXRAD field {field_name!r}.")

    if requested_sweep is None or str(requested_sweep).lower() in {
        "auto",
        "lowest",
    }:
        return min(available, key=lambda item: (item[1], item[0], -item[2]))[0]

    try:
        sweep_number = int(requested_sweep)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sweep must be an integer, 'lowest', 'auto', or None."
        ) from exc
    if isinstance(requested_sweep, (float, np.floating)):
        if not float(requested_sweep).is_integer():
            raise ValueError(
                "sweep must be an integer, 'lowest', 'auto', or None."
            )
    requested_sweep = sweep_number

    for sweep, _, _ in available:
        if sweep == requested_sweep:
            return sweep

    choices = ", ".join(
        f"{sweep} ({angle:g} deg)" for sweep, angle, _ in available
    )
    raise ValueError(
        f"Sweep {requested_sweep} does not contain valid {field_name!r} data. "
        f"Available sweeps: {choices}."
    )


def _velocity_limit(radar, sweep, field_data):
    nyquist_velocity = np.nan
    instrument_parameters = getattr(radar, "instrument_parameters", None) or {}
    parameter = instrument_parameters.get("nyquist_velocity")
    if parameter is not None:
        data = np.ma.filled(parameter["data"], np.nan).astype(np.float32)
        sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
        sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
        if len(data) >= sweep_end:
            nyquist_velocity = float(np.nanmedian(data[sweep_start:sweep_end]))
        elif len(data) > sweep:
            nyquist_velocity = float(data[sweep])

    if not np.isfinite(nyquist_velocity) or nyquist_velocity <= 0:
        finite = np.abs(field_data[np.isfinite(field_data)])
        nyquist_velocity = float(np.nanpercentile(finite, 99)) if finite.size else 35.0
    return float(np.ceil(max(5.0, nyquist_velocity) / 5.0) * 5.0)


def _derive_kdp_from_phidp(phidp_deg, ranges_km):
    from scipy import signal

    phidp_deg = np.ma.filled(
        np.ma.asarray(phidp_deg, dtype=np.float32),
        np.nan,
    )
    kdp_deg_per_km = np.full(phidp_deg.shape, np.nan, dtype=np.float32)
    for ray_index, phidp_ray in enumerate(phidp_deg):
        finite_gate_indices = np.flatnonzero(np.isfinite(phidp_ray))
        if finite_gate_indices.size < 5:
            continue

        gate_runs = np.split(
            finite_gate_indices,
            np.flatnonzero(np.diff(finite_gate_indices) > 1) + 1,
        )
        for gate_indices in gate_runs:
            if gate_indices.size < 5:
                continue

            unwrapped = np.rad2deg(
                np.unwrap(np.deg2rad(phidp_ray[gate_indices]))
            )
            window_length = min(
                31,
                gate_indices.size
                if gate_indices.size % 2
                else gate_indices.size - 1,
            )
            if window_length >= 5:
                unwrapped = signal.savgol_filter(
                    unwrapped,
                    window_length=window_length,
                    polyorder=2,
                    mode="interp",
                )

            derivative = np.gradient(unwrapped, ranges_km[gate_indices])
            if derivative.size >= 9:
                derivative = signal.medfilt(derivative, kernel_size=9)
            elif derivative.size >= 5:
                derivative = signal.medfilt(derivative, kernel_size=5)
            kdp_deg_per_km[ray_index, gate_indices] = 0.5 * derivative

    return np.ma.masked_invalid(kdp_deg_per_km)


def _scan_time(radar, ray_index=0):
    units = str(getattr(radar, "time", {}).get("units", ""))
    if "since " not in units:
        return ""
    value = units.split("since ", 1)[1].strip().replace("Z", "+00:00")
    try:
        base = datetime.fromisoformat(value)
    except ValueError:
        return value
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)

    offsets = np.asarray(getattr(radar, "time", {}).get("data", [0.0]), dtype=float)
    ray_index = int(ray_index)
    if 0 <= ray_index < offsets.size and np.isfinite(offsets[ray_index]):
        offset = float(offsets[ray_index])
    else:
        offset = 0.0
    return (base + timedelta(seconds=offset)).astimezone(timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def _metadata_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def nexrad_ppi_from_radar(
    radar,
    product="reflectivity",
    *,
    sweep="lowest",
    max_range_km=150.0,
    resolution=600,
    source="NEXRAD Level II",
):
    product = _normalize_product(product)
    spec = NEXRAD_PRODUCT_SPECS[product]
    field_name = spec["field_name"]
    source_field_name, derived = _resolve_product_source(spec, radar.fields)
    if source_field_name is None:
        available = ", ".join(sorted(radar.fields)) or "none"
        required = [field_name]
        fallback_field = spec.get("source_field_name")
        if fallback_field and fallback_field != field_name:
            required.append(fallback_field)
        required_fields = " or ".join(repr(name) for name in required)
        raise RuntimeError(
            f"NEXRAD product {product!r} needs {required_fields}, which is not "
            f"present. Available fields: {available}."
        )

    sweep = _select_sweep(radar, source_field_name, sweep)
    native_field = np.ma.asarray(radar.get_field(sweep, source_field_name))
    sweep_start = int(radar.sweep_start_ray_index["data"][sweep])
    sweep_end = int(radar.sweep_end_ray_index["data"][sweep]) + 1
    azimuths_deg = np.asarray(
        radar.azimuth["data"][sweep_start:sweep_end],
        dtype=np.float32,
    )
    ranges_km = np.asarray(radar.range["data"], dtype=np.float32) / 1000.0
    if derived == "kdp_from_phidp":
        native_field = _derive_kdp_from_phidp(native_field, ranges_km)

    data = polar_to_cartesian(
        native_field,
        azimuths_deg,
        ranges_km,
        max_range_km=max_range_km,
        resolution=resolution,
    )
    field_data = data["field_data"]

    if product == "velocity":
        limit = _velocity_limit(radar, sweep, field_data)
        vmin, vmax = -limit, limit
    else:
        valid_values = field_data[data["valid_mask"]]
        vmin = spec.get("vmin")
        vmax = spec.get("vmax")
        if vmin is None:
            vmin = float(np.nanmin(valid_values)) if valid_values.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(valid_values)) if valid_values.size else 1.0

    field_metadata = radar.fields.get(source_field_name, {})
    if derived:
        units = str(spec.get("units", ""))
    else:
        units = str(field_metadata.get("units", spec.get("units", "")))
    station = str(
        getattr(radar, "metadata", {}).get("instrument_name", "")
    )
    elevation_deg = float(radar.fixed_angle["data"][sweep])
    field_key = spec["field_key"]
    data.update(
        station=station,
        scan_time=_scan_time(radar, sweep_start),
        elevation_deg=round(elevation_deg, 2),
        vcp_pattern=_metadata_int(
            getattr(radar, "metadata", {}).get("vcp_pattern", -1)
        ),
        sweep=int(sweep),
        product=product,
        field_name=field_name,
        source_field_name=source_field_name,
        derived=bool(derived),
        field_key=field_key,
        units=units,
        vmin=float(vmin),
        vmax=float(vmax),
        cmap_name=spec["cmap_name"],
        fallback_cmap_name=spec["fallback_cmap_name"],
        source=str(source),
    )
    data[field_key] = field_data
    return data


def get_available_nexrad_products(
    s3_key=NEXRAD_CASE_S3_KEY,
    *,
    cache_dir=None,
    volume_path=None,
    radar=None,
):
    if radar is None:
        radar = read_nexrad_volume(
            volume_path,
            s3_key=s3_key,
            cache_dir=cache_dir,
        )
    available = {}
    for product, spec in NEXRAD_PRODUCT_SPECS.items():
        source_field_name, derived = _resolve_product_source(spec, radar.fields)
        if source_field_name is None:
            continue
        available[product] = {
            "field_name": spec["field_name"],
            "source_field_name": source_field_name,
            "field_key": spec["field_key"],
            "aliases": spec["aliases"],
            "derived": bool(derived),
            "units": spec.get("units", ""),
        }
    return available


def _safe_token(value):
    return "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in str(value)
    )


def _format_number(value):
    return f"{float(value):g}".replace(".", "p")


def _source_cache_identity(s3_key, volume_path):
    if volume_path is None:
        s3_key = _validate_s3_key(s3_key)
        label = Path(s3_key).name
        identity = f"s3:{s3_key}"
        source = f"{NEXRAD_ARCHIVE_URL}/{s3_key}"
        return label, identity, source, None

    volume_path = Path(volume_path).expanduser().resolve()
    if not volume_path.is_file():
        raise FileNotFoundError(f"NEXRAD volume not found: {volume_path}")
    stat = volume_path.stat()
    identity = (
        f"file:{volume_path}:{stat.st_mtime_ns}:{stat.st_size}"
    )
    return volume_path.name, identity, str(volume_path), volume_path


def _volume_cache_id(label, identity):
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{_safe_token(label)}_{digest}"


def _cache_paths(cache_dir, volume_id, product, sweep, max_range_km, resolution):
    sweep_token = _safe_token(sweep if sweep is not None else "auto")
    stem = (
        f"{_safe_token(volume_id)}_{_safe_token(product)}_sw{sweep_token}_"
        f"{_format_number(max_range_km)}km_{int(resolution)}px_"
        f"v{NEXRAD_CACHE_VERSION}"
    )
    base = _cache_dir(cache_dir) / "ppi" / stem
    return base.with_suffix(".json"), base.with_suffix(".npz")


def _copy_ppi_data(data):
    result = {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    field_key = result.get("field_key")
    if field_key and field_key != "field_data":
        result[field_key] = result["field_data"]
    return result


def _load_ppi_cache(paths):
    metadata_path, arrays_path = paths
    if not metadata_path.is_file() or not arrays_path.is_file():
        return None
    try:
        with metadata_path.open() as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.pop("cache_version", None) != NEXRAD_CACHE_VERSION:
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
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    data.update(metadata)
    field_key = data["field_key"]
    data[field_key] = data["field_data"]
    return data


def _write_ppi_cache(paths, data):
    metadata_path, arrays_path = paths
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    metadata_tmp = metadata_path.with_name(metadata_path.name + suffix)
    arrays_tmp = arrays_path.with_name(arrays_path.name + suffix)
    metadata = {
        key: value
        for key, value in data.items()
        if not isinstance(value, np.ndarray)
    }
    metadata["cache_version"] = NEXRAD_CACHE_VERSION
    try:
        with arrays_tmp.open("wb") as arrays_file:
            np.savez_compressed(
                arrays_file,
                field_data=data["field_data"],
                valid_mask=data["valid_mask"],
                azimuth_grid_deg=data["azimuth_grid_deg"],
                range_grid_km=data["range_grid_km"],
                x_coords_km=data["x_coords_km"],
                y_coords_km=data["y_coords_km"],
            )
        with metadata_tmp.open("w") as metadata_file:
            json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            metadata_file.write("\n")
        arrays_tmp.replace(arrays_path)
        metadata_tmp.replace(metadata_path)
    finally:
        arrays_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)


def get_nexrad_product_ppi_data(
    product,
    s3_key=NEXRAD_CASE_S3_KEY,
    sweep="lowest",
    max_range_km=150.0,
    resolution=600,
    *,
    cache_dir=None,
    volume_path=None,
    refresh_cache=False,
    force_download=False,
):
    product = _normalize_product(product)
    if int(resolution) != resolution or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2.")
    resolution = int(resolution)
    max_range_km = float(max_range_km)
    if force_download:
        refresh_cache = True

    label, identity, source, volume_path = _source_cache_identity(
        s3_key,
        volume_path,
    )
    volume_id = _volume_cache_id(label, identity)

    paths = _cache_paths(
        cache_dir,
        volume_id,
        product,
        sweep,
        max_range_km,
        resolution,
    )
    if not refresh_cache:
        cached = _load_ppi_cache(paths)
        if cached is not None:
            return _copy_ppi_data(cached)

    radar = read_nexrad_volume(
        volume_path,
        s3_key=s3_key,
        cache_dir=cache_dir,
        force_download=force_download,
    )
    data = nexrad_ppi_from_radar(
        radar,
        product,
        sweep=sweep,
        max_range_km=max_range_km,
        resolution=resolution,
        source=source,
    )
    _write_ppi_cache(paths, data)
    return _copy_ppi_data(data)
