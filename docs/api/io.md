# I/O Reference

`metrust.io` provides file readers for common meteorological data formats.
Most readers are **native Rust** implementations exposed via PyO3 -- no MetPy
dependency required. A small number of readers are **MetPy shims** that
forward to MetPy when it is installed.

```python
from metrust.io import Level3File, parse_metar, GiniFile
```

---

## Implementation Status

| Reader | Backend | Dependency |
|---|---|---|
| `Level3File` | Native Rust | None |
| `Level2File` | MetPy shim | `pip install metpy` |
| `Metar` / `parse_metar` / `parse_metar_file` | Native Rust | None |
| `StationInfo` / `StationLookup` | Native Rust | None |
| `GiniFile` | Native Rust | None |
| `GempakGrid` / `GempakGridRecord` | Native Rust | None |
| `GempakSounding` / `GempakSoundingStation` / `SoundingData` | Native Rust | None |
| `GempakSurface` / `GempakSurfaceStation` / `SurfaceObs` | Native Rust | None |
| `SurfaceBulletinFeature` / `parse_wpc_surface_bulletin` | Native Rust | None |
| `is_precip_mode` | Native Rust | None |

---

## NEXRAD Level 3 (NIDS)

### `Level3File` -- Native Rust

Parses NEXRAD Level III (NIDS) product files. Supports reading from a file
path or from raw bytes in memory.

```python
from metrust.io import Level3File

f = Level3File.from_file("/path/to/KFWS_N0Q_20230101_0000.nids")

# Or from bytes already in memory
with open("/path/to/product", "rb") as fh:
    f = Level3File.from_bytes(fh.read())
```

#### Static Methods

| Method | Signature | Description |
|---|---|---|
| `from_file` | `(path: str) -> Level3File` | Parse a Level 3 product from a file path. |
| `from_bytes` | `(data: bytes) -> Level3File` | Parse a Level 3 product from raw bytes. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `product_code` | `int` | Numeric product code (e.g., 94 = base reflectivity 0.5 deg). |
| `source_id` | `int` | Source (radar) ID. |
| `latitude` | `float` | Latitude of the radar (degrees). |
| `longitude` | `float` | Longitude of the radar (degrees). |
| `height` | `float` | Height of the radar (feet). |
| `volume_time` | `str` | Volume scan time as `"YYYY-MM-DD HH:MM:SS"`. |
| `data` | `numpy.ndarray` | Flattened data values (float64). Row-major: radial x bin. |
| `num_bins` | `int` | Number of range bins per radial. |
| `num_radials` | `int` | Number of radials in the product. |

To reshape the data into a 2-D array:

```python
import numpy as np
data_2d = f.data.reshape(f.num_radials, f.num_bins)
```

---

## NEXRAD Level 2

### `Level2File` -- MetPy Shim

Level 2 base data reading is not yet implemented in Rust. When MetPy is
installed, `metrust.io.Level2File` transparently forwards to
`metpy.io.Level2File`.

```python
# Requires: pip install metpy
from metrust.io import Level2File

radar = Level2File("/path/to/KFWS20230101_000000_V06")
```

If MetPy is not installed, importing `Level2File` raises an `ImportError`
with a message explaining how to install MetPy for Level 2 support.

---

## METAR Parsing

### `Metar` -- Native Rust

A parsed METAR aviation weather observation.

```python
from metrust.io import Metar

obs = Metar.parse("KATL 011953Z 27010G18KT 10SM FEW250 25/20 A3012")
print(obs.station)       # "KATL"
print(obs.temperature)   # 25.0
print(obs.wind_speed)    # 10.0
print(obs.sky_cover)     # [("FEW", 25000)]
print(obs.weather)       # []
```

#### Static Methods

| Method | Signature | Description |
|---|---|---|
| `parse` | `(text: str) -> Metar` | Parse a single METAR observation string. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `station` | `str` | ICAO station identifier (e.g., `"KATL"`). |
| `time` | `str` | Observation time as `"DDHHMMz"`. |
| `wind_direction` | `float or None` | Wind direction in degrees (`None` for calm/variable). |
| `wind_speed` | `float or None` | Sustained wind speed in knots. |
| `wind_gust` | `float or None` | Gust speed in knots, if reported. |
| `visibility` | `float or None` | Prevailing visibility in statute miles. |
| `temperature` | `float or None` | Temperature in Celsius. |
| `dewpoint` | `float or None` | Dewpoint in Celsius. |
| `altimeter` | `float or None` | Altimeter setting in inches of mercury. |
| `sky_cover` | `list[tuple[str, int or None]]` | Sky condition layers as `(cover_type, height_ft)` tuples. Cover types: `CLR`, `SKC`, `FEW`, `SCT`, `BKN`, `OVC`, `VV`. |
| `weather` | `list[str]` | Present weather phenomena tokens (e.g., `"RA"`, `"+TSRA"`, `"-SN"`). |
| `raw` | `str` | Original raw METAR string. |

### `parse_metar(text)` -- Native Rust

Convenience function. Equivalent to `Metar.parse(text)`.

```python
from metrust.io import parse_metar
obs = parse_metar("KJFK 011953Z 36008KT 10SM SCT250 22/18 A3010")
```

### `parse_metar_file(content)` -- Native Rust

Parse a multi-line string containing one METAR per line. Blank lines and
lines starting with `#` are skipped. Returns all successfully parsed
METARs; unparseable lines are silently ignored.

```python
from metrust.io import parse_metar_file

text = open("metars.txt").read()
observations = parse_metar_file(text)
print(f"Parsed {len(observations)} METARs")
```

**Signature:** `parse_metar_file(content: str) -> list[Metar]`

---

## Station Lookup

### `StationInfo` -- Native Rust

Metadata for a single weather observation station.

```python
from metrust.io import StationInfo

stn = StationInfo(
    id="KATL",
    name="Hartsfield-Jackson Atlanta Intl",
    state="GA",
    country="US",
    latitude=33.6301,
    longitude=-84.4418,
    elevation=315.0,
)
```

#### Constructor

```
StationInfo(id, name, state, country, latitude, longitude, elevation)
```

All arguments are required.

#### Properties

| Property | Type | Description |
|---|---|---|
| `id` | `str` | Station identifier (e.g., `"KATL"`, `"72219"`). |
| `name` | `str` | Human-readable station name. |
| `state` | `str` | State or province code (may be empty for non-US stations). |
| `country` | `str` | Country code (e.g., `"US"`). |
| `latitude` | `float` | Latitude in decimal degrees. |
| `longitude` | `float` | Longitude in decimal degrees. |
| `elevation` | `float` | Elevation in metres above mean sea level. |

### `StationLookup` -- Native Rust

In-memory station database with ID lookup and spatial queries.

```python
from metrust.io import StationLookup, StationInfo

db = StationLookup()
db.add(StationInfo("KATL", "Atlanta", "GA", "US", 33.63, -84.44, 315.0))
db.add(StationInfo("KJFK", "JFK Intl", "NY", "US", 40.64, -73.78, 3.9))

# Lookup by ID (case-insensitive)
stn = db.lookup("katl")

# Find nearest station to a lat/lon
nearest = db.nearest(34.0, -84.0)

# Find all stations within 200 km of a point
nearby = db.within_radius(34.0, -84.0, 200.0)
```

#### Constructor

```
StationLookup()                         # empty table
StationLookup.from_entries(entries)      # from a list of StationInfo
```

#### Methods

| Method | Signature | Description |
|---|---|---|
| `add` | `(station: StationInfo)` | Add a station to the table. |
| `lookup` | `(id: str) -> StationInfo or None` | Look up a station by its ID (case-insensitive). |
| `nearest` | `(lat: float, lon: float) -> StationInfo or None` | Find the station nearest to a given lat/lon. |
| `within_radius` | `(lat: float, lon: float, radius_km: float) -> list[StationInfo]` | Find all stations within a radius (km). Results sorted nearest-first. |
| `len` | `() -> int` | Number of stations in the table. |
| `is_empty` | `() -> bool` | Whether the table is empty. |

`StationLookup` also supports `len()` via Python's `__len__` protocol.

---

## GINI Satellite Images

### `GiniFile` -- Native Rust

Parses GINI-format satellite image files (GOES, POES, etc.).

```python
from metrust.io import GiniFile

img = GiniFile.from_file("/path/to/satellite.gini")
print(img.creating_entity, img.sector, img.channel)
print(f"{img.nx}x{img.ny} pixels, projection: {img.projection}")
```

#### Static Methods

| Method | Signature | Description |
|---|---|---|
| `from_file` | `(path: str) -> GiniFile` | Parse a GINI file from a file path. |
| `from_bytes` | `(data: bytes) -> GiniFile` | Parse a GINI file from raw bytes. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `source` | `str` | Source indicator. |
| `creating_entity` | `str` | Creating entity (satellite platform). |
| `sector` | `str` | Image sector name. |
| `channel` | `str` | Channel / physical element name. |
| `nx` | `int` | Number of pixels per scan line (columns). |
| `ny` | `int` | Number of scan lines (rows). |
| `projection` | `str` | Map projection name. |
| `lat1` | `float` | Latitude of the first grid point (degrees). |
| `lon1` | `float` | Longitude of the first grid point (degrees). |
| `dx` | `float` | Grid spacing in x (km). |
| `dy` | `float` | Grid spacing in y (km). |
| `datetime` | `str` | Image datetime as `"YYYY-MM-DD HH:MM:SS"`. |
| `data` | `numpy.ndarray` | Raw pixel data as a uint8 numpy array. |

---

## GEMPAK Readers

### `GempakGrid` -- Native Rust

Reads GEMPAK gridded data files.

```python
from metrust.io import GempakGrid

grd = GempakGrid.from_file("/path/to/model.gem")
print(f"{grd.nx}x{grd.ny}, projection: {grd.projection}")
print(f"{grd.num_grids} grids available")

# List available grids
for info in grd.grid_info():
    print(info)

# Search for a parameter
temp_grids = grd.find_grids("TMPK")

# Get a specific grid by parameter and level
rec = grd.get_grid("TMPK", 850.0)
if rec:
    data = rec.data.reshape(grd.ny, grd.nx)
```

#### Static Methods

| Method | Signature | Description |
|---|---|---|
| `from_file` | `(path: str) -> GempakGrid` | Open and parse a GEMPAK grid file. |
| `from_bytes` | `(data: bytes) -> GempakGrid` | Parse a GEMPAK grid file from raw bytes. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `source` | `str` | Data source description. |
| `grid_type` | `str` | File type (always `"grid"` for grid files). |
| `nx` | `int` | Number of grid columns (x-dimension). |
| `ny` | `int` | Number of grid rows (y-dimension). |
| `projection` | `str or None` | Projection name from the navigation block (e.g., `"LCC"`, `"CED"`). |
| `lower_left_lat` | `float or None` | Lower-left latitude of the grid domain (degrees). |
| `lower_left_lon` | `float or None` | Lower-left longitude (degrees). |
| `upper_right_lat` | `float or None` | Upper-right latitude (degrees). |
| `upper_right_lon` | `float or None` | Upper-right longitude (degrees). |
| `num_grids` | `int` | Number of grids in the file. |
| `grids` | `list[GempakGridRecord]` | All grid records. |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `grid_info` | `() -> list[str]` | Return a summary string for each grid. |
| `find_grids` | `(parameter: str) -> list[GempakGridRecord]` | Find grids matching a parameter name (case-insensitive). |
| `get_grid` | `(parameter: str, level: float) -> GempakGridRecord or None` | Get a specific grid by parameter name and level. |

### `GempakGridRecord` -- Native Rust

A single grid record from a GEMPAK file.

#### Properties

| Property | Type | Description |
|---|---|---|
| `grid_number` | `int` | Grid number (index within the file). |
| `parameter` | `str` | Parameter name (e.g., `"TMPK"`, `"HGHT"`). |
| `level` | `float` | Primary vertical level value. |
| `level2` | `float` | Secondary vertical level (for layers), or `-1`. |
| `coordinate` | `str` | Vertical coordinate type (e.g., `"PRES"`, `"HGHT"`). |
| `time` | `str` | Valid datetime string. |
| `time2` | `str` | Secondary datetime string. |
| `forecast_type` | `str` | Forecast type (e.g., `"analysis"`, `"forecast"`). |
| `forecast_time` | `str` | Forecast time offset as `"HHH:MM"`. |
| `data` | `numpy.ndarray` | Grid data values as float64 (row-major, `ny * nx` elements). |
| `data_length` | `int` | Number of data values. |

### `GempakSounding` -- Native Rust

Reads GEMPAK sounding (upper air) files.

```python
from metrust.io import GempakSounding

snd = GempakSounding.from_file("/path/to/sounding.gem")
print(f"{snd.num_stations} stations, {snd.num_soundings} soundings")

for s in snd.soundings:
    print(f"{s.station_id} at {s.time}: {s.num_levels} levels")
    p = s.pressure    # numpy array of pressure levels (hPa)
    T = s.temperature # numpy array
```

#### Static Methods

| Method | Signature | Description |
|---|---|---|
| `from_file` | `(path: str) -> GempakSounding` | Open and parse a GEMPAK sounding file. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `merged` | `bool` | Whether the file uses merged (SNDT) format. |
| `stations` | `list[GempakSoundingStation]` | Station entries in the file. |
| `soundings` | `list[SoundingData]` | All sounding profiles. |
| `num_soundings` | `int` | Number of soundings. |
| `num_stations` | `int` | Number of stations. |

### `GempakSoundingStation` -- Native Rust

A station entry from a GEMPAK sounding file.

| Property | Type | Description |
|---|---|---|
| `id` | `str` | Station identifier. |
| `lat` | `float` | Latitude (degrees). |
| `lon` | `float` | Longitude (degrees). |
| `elevation` | `float` | Elevation (m). |
| `state` | `str` | State code. |
| `country` | `str` | Country code. |
| `station_number` | `int` | WMO station number. |

### `SoundingData` -- Native Rust

A single vertical sounding profile.

#### Properties

| Property | Type | Description |
|---|---|---|
| `station_id` | `str` | Station identifier. |
| `time` | `str` | Observation time string. |
| `pressure` | `numpy.ndarray` | Pressure levels (hPa), float64. |
| `temperature` | `numpy.ndarray` | Temperature values, float64. |
| `dewpoint` | `numpy.ndarray` | Dewpoint values, float64. |
| `wind_direction` | `numpy.ndarray` | Wind direction (degrees), float64. |
| `wind_speed` | `numpy.ndarray` | Wind speed, float64. |
| `height` | `numpy.ndarray` | Geopotential height, float64. |
| `num_levels` | `int` | Number of vertical levels. |
| `parameter_names` | `list[str]` | All parameter names available in this sounding. |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `get_parameter` | `(name: str) -> numpy.ndarray or None` | Get a named parameter as a numpy array. |

### `GempakSurface` -- Native Rust

Reads GEMPAK surface observation files.

```python
from metrust.io import GempakSurface

sfc = GempakSurface.from_file("/path/to/surface.gem")
print(f"Type: {sfc.surface_type}")
print(f"{sfc.num_stations} stations, {sfc.num_observations} observations")

for obs in sfc.observations:
    print(f"{obs.station_id} T={obs.temperature} Td={obs.dewpoint}")
```

#### Static Methods

| Method | Signature | Description |
|---|---|---|
| `from_file` | `(path: str) -> GempakSurface` | Open and parse a GEMPAK surface file. |

#### Properties

| Property | Type | Description |
|---|---|---|
| `surface_type` | `str` | Surface file type: `"standard"`, `"ship"`, or `"climate"`. |
| `stations` | `list[GempakSurfaceStation]` | Stations in the file. |
| `observations` | `list[SurfaceObs]` | All surface observations. |
| `num_observations` | `int` | Number of observations. |
| `num_stations` | `int` | Number of stations. |

### `GempakSurfaceStation` -- Native Rust

A station from a GEMPAK surface file.

| Property | Type | Description |
|---|---|---|
| `id` | `str` | Station identifier. |
| `lat` | `float` | Latitude (degrees). |
| `lon` | `float` | Longitude (degrees). |
| `elevation` | `float` | Elevation (m). |
| `state` | `str` | State code. |
| `country` | `str` | Country code. |
| `station_number` | `int` | WMO station number. |

### `SurfaceObs` -- Native Rust

A single surface observation.

#### Properties

| Property | Type | Description |
|---|---|---|
| `station_id` | `str` | Station identifier. |
| `time` | `str` | Observation time string. |
| `temperature` | `float or None` | Temperature (units depend on file: `TMPC`=Celsius, `TMPF`=Fahrenheit). |
| `dewpoint` | `float or None` | Dewpoint temperature. |
| `wind_direction` | `float or None` | Wind direction (degrees). |
| `wind_speed` | `float or None` | Wind speed. |
| `pressure` | `float or None` | Pressure (PMSL, PRES, or ALTI). |
| `visibility` | `float or None` | Visibility. |
| `sky_cover` | `str or None` | Sky cover string. |
| `parameter_names` | `list[str]` | All parameter names available in this observation. |

#### Methods

| Method | Signature | Description |
|---|---|---|
| `get_parameter` | `(name: str) -> float or None` | Get a named parameter value. |

---

## WPC Surface Bulletins

### `parse_wpc_surface_bulletin(text)` -- Native Rust

Parse a WPC coded surface bulletin into a list of synoptic features
(pressure centers, fronts, troughs).

```python
from metrust.io import parse_wpc_surface_bulletin

bulletin_text = open("coded_sfc_bulletin.txt").read()
features = parse_wpc_surface_bulletin(bulletin_text)

for feat in features:
    print(f"{feat.feature_type}: {len(feat.points)} points, value={feat.value}")
```

**Signature:** `parse_wpc_surface_bulletin(text: str) -> list[SurfaceBulletinFeature]`

### `SurfaceBulletinFeature` -- Native Rust

A decoded feature from a WPC coded surface bulletin.

| Property | Type | Description |
|---|---|---|
| `feature_type` | `str` | Feature type: `"HIGH"`, `"LOW"`, `"WARM"`, `"COLD"`, `"STNRY"`, `"OCFNT"`, `"TROF"`. |
| `points` | `list[tuple[float, float]]` | Lat/lon pairs as `(lat, lon)` tuples. |
| `value` | `float or None` | Central pressure (mb) for HIGH/LOW, or `None` for fronts. |
| `strength` | `str or None` | Strength qualifier (e.g., `"WK"`, `"MDT"`, `"STG"`), if present. |
| `valid_time` | `str or None` | Valid time string from the bulletin. |

---

## Utility Functions

### `is_precip_mode(vcp)` -- Native Rust

Return `True` if the given NEXRAD Volume Coverage Pattern (VCP) number
corresponds to a precipitation scanning mode.

```python
from metrust.io import is_precip_mode

is_precip_mode(215)  # True  -- precipitation mode
is_precip_mode(31)   # False -- clear-air mode
```

**Signature:** `is_precip_mode(vcp: int) -> bool`
