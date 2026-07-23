# melp

Reusable ManimGL helpers and radar mobjects for this repository.

## Easing functions

`melp` includes the complete easings.net collection as scalar Manim rate
functions. Names use the same snake_case convention as Manim:

```python
from melp import ease_in_out_back, ease_out_bounce

self.play(
    tracker @ 1,
    rate_func=ease_in_out_back,
    run_time=2,
)
```

The collection includes `linear` and the `ease_in_*`, `ease_out_*`, and
`ease_in_out_*` variants of sine, quad, cubic, quart, quint, expo, circ, back,
elastic, and bounce. Back and elastic curves intentionally overshoot the
`[0, 1]` output range. `cubic_bezier(...)`, `bezier_out_in`, `smootherstep`,
and `get_phase_eased_sine(...)` are also available for custom motion.

## Video playback

`VideoMobject` decodes video frames with FFmpeg and updates a ManimGL image
texture. Durations and remaining times are returned in seconds:

```python
from melp import VideoMobject

video = VideoMobject("clip.mp4", speed=1.0)
self.add(video)

video.pause()
video.play()
video.set_speed(2)

duration = video.get_duration()
remaining = video.get_time_remaining()
```

## WSR-88 beams

`WSR88Model.get_beam(...)` returns an extensible `Beam` object whose apex and
axis match the model's current feed-horn pose:

```python
from manimlib import BLUE, DEG, TEAL

from melp.props import get_wsr88_model

radar = get_wsr88_model(height=3.3)
beam = radar.get_beam(
    beamwidth=0.95 * DEG,
    range=8,
    color=BLUE,
    opacity=0.12,
)
self.add(radar, beam)
```

`beamwidth` is the full angular width in radians and `range` is measured in
Manim scene units. The returned beam captures the current radar pose. Use
`always_redraw(lambda: radar.get_beam(...))` when it should follow animated
azimuth or elevation changes.

Use `get_section(start_range, stop_range)` to isolate a range-resolution
volume from the current beam:

```python
volume_bin = beam.get_section(
    4.0,
    5.2,
    color=TEAL,
    opacity=0.3,
    depth_test=False,
)
self.add(volume_bin)
```

Both ranges are measured from the beam origin in scene units. The section
captures the beam's current pose, so wrap the call in `always_redraw(...)` if
it should follow a moving beam. Disabling depth testing avoids z-fighting when
the highlighted section is drawn over the full beam.

Use `get_beamwidth_arc(...)` to place an angular marker at a chosen range. Pass
the current camera frame so the arc uses the cross-beam plane with the largest
visible screen width:

```python
beamwidth_arc = beam.get_beamwidth_arc(
    range=4,
    camera_frame=self.frame,
    color=WHITE,
    stroke_width=5,
)
self.play(ShowCreation(beamwidth_arc))
```

The arc faces the camera position at the time it is built. If the beam or
camera moves afterward, add an updater after its reveal:

```python
beamwidth_arc.add_updater(
    lambda arc: arc.become(
        beam.get_beamwidth_arc(4, self.frame, color=WHITE, stroke_width=5)
    )
)
```

Use `get_section_particles(...)` to create a `DotCloud` sampled uniformly by
volume inside the same range bounds as `get_section(...)`. Set
`max_radius_fraction` below 1 to leave visual space between the particles and
the gate surface:

```python
particles = beam.get_section_particles(
    start_range=4,
    stop_range=5.2,
    count=180,
    max_radius_fraction=0.84,
    seed=8,
    radius=0.04,
    color=BLUE_B,
    glow_factor=0.3,
)
particles.set_color_by_gradient(BLUE_B, BLUE_A, WHITE)
```

The returned cloud captures the beam's current pose. Use a fixed `seed` if the
cloud is rebuilt in an updater so its particles do not jump between frames.

### Highlighting a cloud inside a beam

The visible overlap is the intersection of the cloud with the beam's finite
cone. Bind the beam to a `Cloud`, then animate `beam_highlight` from 0 to 1:

```python
self.play(ShowCreation(beam), run_time=3)

cloud.bind_beam_highlight(
    beam,
    outside_opacity=0.12,
    feather=0.04,
)
self.play(cloud.beam_highlight @ 1, run_time=1.25)
```

The cloud keeps its existing colors and opacity inside the beam while the
outside opacity drops. `inside_opacity` and `outside_opacity` are multipliers
of the cloud's existing alpha. The shader mask follows transformed beams and
clouds automatically. Use `cloud.highlight_in_beam(beam, ...)` for an
immediate highlight and `cloud.clear_beam_highlight()` to remove it.

## WSR-88 scan controls and radome panels

The built-in `az` and `el` controls keep driving the antenna when their
animations are nested inside `LaggedStart` or `AnimationGroup`:

```python
panel_fade = LaggedStart(
    *[panel.animate.set_opacity(0) for panel in radar.upper_dome],
    lag_ratio=0.01,
)
self.play(
    LaggedStart(panel_fade, radar.el @ (20 * DEG), lag_ratio=0.4),
    run_time=2,
)
```

With `indexed_radome_panels=True`, the upper dome's direct children are its 180
authored logical panels, not the 1,401 individual mesh faces. The source face
IDs remain available through `upper_dome.face_polygon_ids` and each panel's
`polygon_ids` attribute.

## PPIGrid

`PPIGrid` renders a north-up Cartesian radar raster in its existing array order
with the meteorological azimuth convention used throughout the videos:

- `0 deg` is north/up.
- `90 deg` is east/right.
- `x = r * sin(theta)` and `y = r * cos(theta)`.

Pass any 2D array directly. Rectangular arrays are fitted to the square radar
extent; NaNs and masked values are transparent, and the azimuth grid is
inferred automatically.

```python
import numpy as np
from manimlib import InteractiveScene

from melp import PPIGrid


class SyntheticPPI(InteractiveScene):
    def construct(self):
        y, x = np.mgrid[-1:1:600j, -1:1:600j]
        values = 45 * np.exp(-((x - 0.25) ** 2 + (y - 0.35) ** 2) / 0.08)
        values[x * x + y * y > 1] = np.nan

        ppi = PPIGrid(
            values,
            max_range_km=150,
            vmin=0,
            vmax=50,
            cmap_name="turbo",
            colorbar_label="value",
            font="Maple Mono CN",
            init_scan_progress=0,
        )
        self.add(ppi)
        self.play(ppi.scan_progress @ 360, run_time=4)
```

`scan_progress` and `scan_start` are `VT` instances measured in degrees.
`data_opacity_tracker` controls the raster opacity. A complete PPI is shown by
default; pass `init_scan_progress=0` when you want a sweep reveal.

Pass `font=...` to set one font for every PPI text element, including cardinal
labels, colorbar tick labels, and the colorbar title. It defaults to the
repository's `FONT` environment variable when that is set.

For native `(ray, gate)` data from Py-ART, xradar, a file, or a proprietary
radar, use `from_polar`:

```python
ppi = PPIGrid.from_polar(
    field_data,       # shape: (number of rays, number of gates)
    azimuths_deg,     # one azimuth per ray
    ranges_km,        # one range per gate
    max_range_km=120,
    resolution=600,
    vmin=-20,
    vmax=75,
    cmap_name="turbo",
)
```

This uses nearest-ray and nearest-gate sampling so the displayed raster does
not invent interpolated measurements. You can convert a radar coordinate back
to the transformed Manim point with:

```python
point = ppi.range_azimuth_to_point(range_km=60, azimuth_deg=135)
```

Mappings are also accepted. Set `field_key` in the mapping, pass it to the
constructor, or provide a `field_data` entry. If your source already has
per-pixel azimuth and range grids, include them so scan reveals follow the
source geometry:

```python
ppi = PPIGrid(
    {
        "field_key": "snr_db",
        "snr_db": snr_image,
        "valid_mask": valid_mask,
        "azimuth_grid_deg": azimuth_grid_deg,
        "range_grid_km": range_grid_km,
        "max_range_km": 80,
        "vmin": -10,
        "vmax": 30,
    },
    cmap_name="viridis",
)
```

## NEXRAD Level-II data

NEXRAD decoding is optional and lazy. Install Py-ART in the ManimGL
environment before using it:

```bash
uv pip install arm-pyart
```

Py-ART currently supports Python 3.11 through 3.13. Use a ManimGL environment
on one of those versions for the NEXRAD decoder; the rest of `melp`, including
`PPIGrid` and raw-volume downloading, does not require Py-ART.

`cmweather` is optional. If it is installed, NEXRAD products use palettes such
as `NWSRef` and `NWSVel`; otherwise `PPIGrid` automatically uses a built-in
Matplotlib fallback.

The returned mapping feeds directly into `PPIGrid`:

```python
from manimlib import InteractiveScene

from melp import PPIGrid, get_nexrad_product_ppi_data


class NexradPPI(InteractiveScene):
    def construct(self):
        data = get_nexrad_product_ppi_data(
            "reflectivity",
            s3_key="2024/05/07/KTLX/KTLX20240507_023811_V06",
            sweep="lowest",
            max_range_km=150,
            resolution=600,
        )
        ppi = PPIGrid(
            data,
            colorbar_label=data["units"],
            font="Maple Mono CN",
            init_scan_progress=0,
        )
        self.add(ppi)
        self.play(ppi.scan_progress @ 360, run_time=4)
```

The first call downloads the public archive object from the Unidata NEXRAD
Level-II S3 bucket, decodes it with Py-ART, and caches both the raw volume and
the Cartesian PPI. Later calls reuse the cache. The default cache is
`~/.cache/melp/nexrad`; pass `cache_dir=...` or set `MELP_NEXRAD_CACHE` to put
it elsewhere.

Pass `sweep="lowest"` (the default) to select the lowest elevation containing
the requested product. An integer requests that exact sweep and raises a clear
error if the product is not available there.

To download without decoding, or to use an existing file:

```python
from melp import fetch_nexrad_volume, get_nexrad_product_ppi_data

s3_key = "2024/05/07/KTLX/KTLX20240507_023811_V06"
path = fetch_nexrad_volume(s3_key, cache_dir="./radar-cache")
data = get_nexrad_product_ppi_data(
    "velocity",
    volume_path=path,
    cache_dir="./radar-cache",
)
```

Supported product names are `reflectivity`, `velocity`, `zdr`, `rhohv`,
`phidp`, and `kdp` when the corresponding source moment exists in the decoded
volume. KDP uses the native field when available and is otherwise derived from
PhiDP. Use
`get_available_nexrad_products(...)` to inspect a particular file.

The NEXRAD helper returns a Cartesian visualization raster. Use the loaded
Py-ART object itself when an animation needs the true native measured rays.
