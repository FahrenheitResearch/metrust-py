# I/O Formats and Parsers

This document covers the binary and text format parsers implemented natively in
rustmet.  The focus is on what each parser does, how the Rust implementation
differs from the Python equivalents, and where the current boundaries are.

---

## 1. NEXRAD Level II (Archive II)

**Crate:** `wx-radar` (`crates/wx-radar/src/level2.rs`)

### Format overview

NEXRAD Level II data is the base-data archive produced by the WSR-88D RDA
(Radar Data Acquisition unit).  Files follow the ICD 2620010H specification
and arrive in "Archive II" format:

| Region       | Size  | Contents                                       |
|-------------|-------|-------------------------------------------------|
| Volume header | 24 B  | Filename, ICAO, modified Julian date, time (ms) |
| Compressed blocks | variable | bzip2-compressed messages (signed 4-byte length prefix per block) |

After the 24-byte volume header, the remainder of the file is a sequence of
length-prefixed bzip2 blocks.  A negative length prefix indicates the final
block in a "cut" (elevation scan).

### Decompression

The parser identifies bzip2 blocks by checking for the `BZ` magic after the
4-byte length prefix.  Blocks are decompressed **in parallel** using Rayon's
`par_iter()`, which gives a significant speedup on multi-core systems compared
to Python's serial `bz2.decompress()` calls.  The decompressed blocks are
concatenated after the volume header to form a flat byte stream.

### Message parsing

The decompressed stream contains 2432-byte fixed-size message slots preceded by
12-byte CTM (Channel Terminal Manager) headers.  Each slot has a 16-byte
message header:

- **Message size** (halfwords)
- **RDA channel** and **message type**
- Sequence number, date, time, segment info

Only **Message Type 31** (digital radar data) is processed; all other types
are skipped by advancing the cursor by 2432 bytes.

### Message 31 structure

Within a Type 31 message, the parser reads:

1. **Msg31 header** -- radar ID, collection time/date, azimuth angle,
   elevation angle, azimuth resolution (0.5 or 1.0 degree), radial status,
   elevation number, cut sector, and data block count.

2. **Data block pointers** -- one `u32` offset per data block.  Blocks have a
   type byte: `D` for moment data, `R` for radial data (contains Nyquist
   velocity).

3. **Moment blocks** -- each moment block carries:
   - 3-byte name (`REF`, `VEL`, `SW`, `ZDR`, `RHO`, `PHI`, `KDP`, etc.)
   - Gate count, first gate range, gate size
   - Data word size (8 or 16 bits), scale factor, offset
   - Raw gate values decoded with: `value = (raw - offset) / scale`
   - Raw values of 0 or 1 are mapped to `NaN` (below-threshold / range-folded)

### Sweep splitting

Radials are collected into sweeps using a two-signal approach:

1. **Radial status** -- values 0 (start elevation), 3 (start volume), and
   5 (start elevation mid-volume, used by SAILS/MESO-SAILS) trigger a new
   sweep boundary.

2. **Elevation number change** -- if radial status markers are missing (some
   older or non-standard data), a change in the elevation number field is
   used as a fallback split signal.

This correctly handles SAILS and MESO-SAILS VCPs where the lowest elevation
is re-scanned multiple times within a single volume, producing duplicate
elevation numbers that must map to separate sweeps.  Each sweep records its
sequential `sweep_index`, the start/end radial status, and the VCP cut sector
number.

### Products

The `RadarProduct` enum covers all standard WSR-88D moments:

| Short name | Product | Unit |
|-----------|---------|------|
| REF | Reflectivity | dBZ |
| VEL | Radial velocity | m/s |
| SW  | Spectrum width | m/s |
| ZDR | Differential reflectivity | dB |
| RHO | Correlation coefficient | -- |
| KDP | Specific differential phase | deg/km |
| PHI | Differential phase | deg |

### Rust vs Python

- **Parallel bzip2 decompression** -- Rayon `par_iter` across compressed
  blocks; Python does them serially.
- **Contiguous memory** -- the decompressed result is a single `Vec<u8>`; no
  intermediate Python objects per block.
- **Zero-allocation parsing** -- the cursor advances through the flat buffer
  using `byteorder::ReadBytesExt` with no per-message heap allocation beyond
  the output `Vec<f32>` for gate data.

---

## 2. GRIB2

**Crate:** `wx-core` (`crates/wx-core/src/grib2/`)

### Format overview

GRIB2 (WMO FM 92 GRIB Edition 2) is the standard container for gridded
meteorological fields.  A file may contain multiple concatenated messages, each
starting with the 4-byte magic `GRIB`.

Each message has 8 sections:

| Section | Name | Key fields |
|---------|------|------------|
| 0 | Indicator | Magic, discipline, edition (must be 2), total length (8 bytes) |
| 1 | Identification | Reference time (year through second) |
| 2 | Local Use | Skipped |
| 3 | Grid Definition | Grid template, Nx, Ny, lat/lon corners, dx/dy, scan mode |
| 4 | Product Definition | Parameter category/number, forecast time, level type/value |
| 5 | Data Representation | Packing template, reference value, binary/decimal scale, bits per value |
| 6 | Bitmap | Presence bitmap (indicator 0 = bitmap present, 255 = none) |
| 7 | Data | Packed data bytes |
| 8 | End | `7777` sentinel |

### Grid templates

The parser supports seven grid definition templates:

- **3.0** -- Latitude/Longitude (equidistant cylindrical)
- **3.1** -- Rotated Latitude/Longitude (adds south pole lat/lon, rotation angle)
- **3.10** -- Mercator
- **3.20** -- Polar Stereographic
- **3.30** -- Lambert Conformal Conic
- **3.40** -- Gaussian Latitude/Longitude
- **3.90** -- Space View Perspective (satellite imagery, e.g., GOES)

### Data representation (unpacking)

The `unpack` module implements ten packing templates:

| Template | Name | Notes |
|----------|------|-------|
| 5.0  | Simple packing | Y = (R + X * 2^E) * 10^(-D) |
| 5.2  | Complex packing | Group references, widths, lengths |
| 5.3  | Complex + spatial differencing | Order 1 or 2 prefix reconstruction |
| 5.4  | IEEE float | Direct f32 or f64, no scaling |
| 5.40 | JPEG2000 | Via `openjp2` C FFI (feature-gated) |
| 5.41 | PNG | Via the `png` crate |
| 5.42 | CCSDS/AEC | Parsed but decode not yet implemented (needs libaec) |
| 5.50 | Spectral simple | (0,0) coefficient as IEEE f32 + simple-packed rest |
| 5.51 | Spectral complex | (0,0) coefficient + complex-packed rest |
| 5.61 | Simple with log pre-processing | 10^x - 1 reverse transform |
| 5.200 | Run-length encoding | NCEP local extension for categorical data |

### BitReader

A custom `BitReader` struct handles sub-byte field extraction.  It has a fast
path that reads up to 8 bytes into a `u64` for aligned or near-aligned
accesses (covering reads up to 56 bits), with a bit-by-bit fallback for edge
cases.  This avoids the overhead of Python's `struct.unpack` per field.

### Streaming parser

`StreamingParser` can process GRIB2 messages incrementally as bytes arrive from
a network download.  It accumulates an internal buffer, scans for `GRIB`
magic, reads the 8-byte total length from the indicator section, and yields
complete messages as they become available.  This enables rendering partial
downloads without waiting for the full file.

### Scan mode and row flipping

`unpack_message()` preserves the original scan order.
`unpack_message_normalized()` additionally flips rows when scan mode bit 6
(`0x40`, +j south-to-north) is set, producing north-to-south (top-down) order
suitable for rendering.

### Python bridge

The `rustmet-py` crate exposes `GribMessage` as a PyO3 class with `.values()`
and `.values_2d()` methods that return NumPy arrays, plus `.lats()` and
`.lons()` for coordinate arrays.  This gives Python code direct access to the
Rust-unpacked data without any serialization overhead.

### Rust vs Python

- **Direct binary reads** -- `read_u16`, `read_u32`, `read_f32` use
  `from_be_bytes` on slice windows.  No `struct.unpack` call overhead.
- **Single-pass section dispatch** -- sections are parsed in a `while` loop
  with a `match` on the section number; no need for multiple passes.
- **Feature-gated codecs** -- JPEG2000 decoding via openjp2 is behind a
  `jpeg2000` feature flag, keeping the default build lean.
- **GRIB2 write support** -- the `Grib2Writer` / `MessageBuilder` API can
  produce GRIB2 files, which has no MetPy equivalent.

---

## 3. METAR Parsing

**Crate:** `wx-obs` (`crates/wx-obs/src/metar.rs`)

### Format

METARs are plain-text aviation weather observations following ICAO Annex 3 /
WMO FM 15 format.  A typical report:

```
KOKC 112053Z 18012G20KT 10SM FEW250 32/14 A2985 RMK AO2 SLP089
```

### Parser approach: sequential token consumer

The parser uses a **token-based state machine**, not regex.  The raw string is
first split at ` RMK ` to separate the body from remarks, then
whitespace-tokenized.  A position cursor advances through the token list,
trying each parser function in the expected METAR order:

1. Skip `METAR` / `SPECI` prefix
2. Station ID (4-character ICAO, validated as alphanumeric)
3. Time group (`DDHHMMz` -- 6 digits, optional trailing Z)
4. Skip `AUTO` / `COR`
5. Wind (`DDDSSKT` or `DDDSSGSSGKT` or `VRBSSKT`)
6. Variable wind direction (`DDDVDDD`)
7. Visibility (statute miles, supports fractions like `1 1/2SM` across two
   tokens, and `M1/4SM` for below-quarter-mile)
8. Weather phenomena (intensity prefix `-`/`+`, descriptor codes, phenomenon
   codes -- may repeat for multiple phenomena)
9. Cloud layers (coverage + 3-digit height in hundreds of feet, optional
   CB/TCU suffix -- may repeat)
10. Temperature/dewpoint (`TT/DD`, `M` prefix for negative)
11. Altimeter (`Annnn`, divided by 100 for inHg)

### Extracted fields

| Field | Type | Notes |
|-------|------|-------|
| `station` | String | 4-letter ICAO |
| `time` | MetarTime | Day, hour, minute |
| `wind` | Wind | Direction, speed, gust, variable range |
| `visibility` | Visibility | Statute miles (f64) |
| `weather` | Vec\<WeatherPhenomenon\> | Intensity + descriptor + phenomenon |
| `clouds` | Vec\<CloudLayer\> | Coverage, height AGL, cloud type |
| `temperature` | i32 | Celsius |
| `dewpoint` | i32 | Celsius |
| `altimeter` | f64 | Inches of mercury |
| `remarks` | String | Raw remarks text |
| `flight_category` | FlightCategory | VFR/MVFR/IFR/LIFR (computed) |

### Flight category

Computed from visibility and ceiling (lowest BKN/OVC/VV layer):

| Category | Visibility | Ceiling |
|----------|-----------|---------|
| LIFR | < 1 SM | < 500 ft |
| IFR  | < 3 SM | < 1000 ft |
| MVFR | <= 5 SM | <= 3000 ft |
| VFR  | > 5 SM | > 3000 ft |

### Weather code tables

- **Descriptors:** MI, PR, BC, DR, BL, SH, TS, FZ
- **Phenomena:** DZ, RA, SN, SG, IC, PL, GR, GS, UP, BR, FG, FU, VA, DU,
  SA, HZ, PY, PO, SQ, FC, SS, DS

### TAF reuse

The TAF parser (`taf.rs`) reuses the METAR token parsers by constructing a
synthetic METAR string from TAF forecast group tokens and running it through
`parse_metar()`.  This avoids duplicating the wind/visibility/cloud parsing
logic.

### Rust vs Python

- **No regex** -- the entire parser is string slicing and `starts_with` /
  `parse::<T>()` calls.  Python METAR libraries typically use compiled regex
  patterns, which carry per-match overhead.
- **Lenient by design** -- unknown or malformed tokens are skipped rather than
  causing a parse failure, so partially valid METARs still yield data.
- **Static dispatch** -- each token parser is a plain function; no dynamic
  dispatch or trait objects.

---

## 4. Station Lookup

**Crate:** `wx-obs` (`crates/wx-obs/src/stations.rs`)

### Storage

Station metadata is a compile-time `static` array of `Station` structs with
`&'static str` fields.  The database contains 300+ NWS/ASOS sites covering all
50 US states plus DC, Puerto Rico, Guam, and the US Virgin Islands.

Each entry stores:
- ICAO code, human name, state abbreviation
- Latitude (f64), longitude (f64), elevation in meters (f64)

### Query functions

| Function | Description |
|----------|-------------|
| `find_station(icao)` | Linear scan for exact ICAO match (case-insensitive) |
| `nearest_station(lat, lon)` | Haversine-distance minimum over all stations |
| `stations_within(lat, lon, radius_km)` | All stations within a radius |

The haversine implementation uses the standard formula with Earth radius =
6371 km.

### RAOB station database

A separate `raob_stations` database in `wx-sounding` holds 92 upper-air
(radiosonde) stations with both WMO number and ICAO identifiers.  It provides
`find_raob_station()` (by WMO or ICAO) and `nearest_raob_station()`.

### Rust vs Python

- **Zero runtime cost** -- the station array is embedded in the binary at
  compile time.  No file I/O, no CSV parsing, no JSON deserialization.
- **`&'static str`** -- station name and code strings are string literals
  baked into the binary, so lookups return references with no allocation.

---

## 5. Sounding Data (Wyoming Archive)

**Crate:** `wx-sounding` (`crates/wx-sounding/src/wyoming.rs`)

### Data source

Soundings are fetched from the University of Wyoming upper-air archive via
HTTP.  The response is an HTML page containing a `<pre>` block with a
fixed-width data table.

### Parser

1. **HTML extraction** -- find `<pre>` / `</pre>` tags (case-insensitive)
2. **Header parsing** -- scan for `Station number:`, `Observation time:`,
   `Station latitude:`, etc. lines; fall back to the RAOB station database
3. **Data table** -- look for two `---` separator lines; everything between
   the second separator and the next blank line (or metadata section) is data
4. **Column detection** -- check for `SPED` (m/s) vs `SKNT` (knots) in the
   header row; convert to knots if needed
5. **Row parsing** -- whitespace-split into at least 8 fields; extract
   pressure, height, temperature, dewpoint, wind direction, wind speed
6. **Validation** -- reject rows with out-of-range values (pressure outside
   1--1100 hPa, height outside -500 to 50000 m, etc.); clamp dewpoint to
   not exceed temperature

Output is a `Sounding` struct with a `Vec<SoundingLevel>` plus station
metadata and a `SoundingIndices` struct (populated later by
`wx_math::thermo`).

### Rust vs Python

- **No BeautifulSoup / lxml** -- the HTML is simple enough that string
  searching for `<pre>` tags suffices.
- **No pandas** -- levels are parsed directly into a `Vec<SoundingLevel>`
  rather than loading into a DataFrame.

---

## 6. GINI Format

GINI (GOES Ingest and NOAAPORT Interface) is not currently implemented in
rustmet.  It was a legacy satellite image format used for GOES imagery prior
to the transition to GOES-R (GOES-16/17/18), which uses GRIB2 and NetCDF.
Modern satellite data is handled through the GRIB2 parser (template 3.90 for
space-view perspective grids).

---

## 7. GEMPAK Formats

GEMPAK (GEneral Meteorology PAcKage) binary formats for grid, sounding, and
surface files are not currently implemented in rustmet.  GEMPAK was largely
superseded by GRIB2 and BUFR for operational data distribution.  The
project's focus is on formats that remain in active operational use.

---

## 8. WPC Surface Bulletins

WPC coded surface bulletins are not currently implemented as a native parser.
Surface observation data is accessed through the METAR parser and the
aviationweather.gov API (`wx-obs/src/fetch.rs`), which provides the same
station observations in a more accessible format.

---

## 9. What's NOT Native

### GINI, GEMPAK, WPC bulletins

As noted above, these legacy formats are not implemented.  GINI is obsolete
(replaced by GOES-R GRIB2/NetCDF), GEMPAK binary formats are rarely
encountered in modern pipelines, and WPC bulletins are covered by METAR
ingest.

### NEXRAD Level III

Level III (RPG products) parsing is not yet implemented.  Level III products
come as pre-processed, derived outputs (composite reflectivity, storm total
precipitation, mesocyclone detections, etc.) rather than raw base data.  The
project currently focuses on Level II base data, which provides the full
radial moment data needed for custom analysis.

### NetCDF / HDF5

The `wx-io` crate has placeholder modules for NetCDF/HDF5 support, but they
are not yet populated.  These formats are important for reanalysis data
(ERA5), satellite products (GOES-R ABI), and research model output.

### CCSDS/AEC decoding (GRIB2 template 5.42)

The GRIB2 parser recognizes CCSDS-packed messages and extracts the header
fields (flags, block size, reference sample interval), but the actual AEC
decompression is not implemented.  This requires integration with `libaec`.
CCSDS packing is used by ECMWF products and some NCEP experimental outputs.

### Why Level II is harder than it looks

The native Level II parser handles the core ICD 2620010H message format, but
several real-world complications remain:

- **Super-resolution** -- modern WSR-88D data uses 0.5-degree azimuth spacing
  and 250 m gate spacing for reflectivity at lower tilts, but 1.0-degree / 250
  m for velocity.  The parser handles variable azimuth resolution per radial
  (the `azimuth_resolution` field), and different gate configurations per
  moment, but some downstream processing assumes uniform spacing.

- **Clutter filtering** -- the raw Level II data includes ground clutter.
  Clutter suppression is applied at the RDA using notch-width filters, but
  residual clutter and filter artifacts (especially in clear-air mode) need
  additional post-processing.  Python tools like MetPy don't address this
  either; it's typically handled by the RPG (which produces Level III).

- **Velocity dealiasing** -- radial velocity is limited to the Nyquist
  interval (+/- Vn, where Vn comes from the PRF).  Dealiasing requires
  comparing adjacent radials and gates to unwrap phase ambiguities.  This is
  an algorithmic problem that neither the Rust nor the Python parsers solve
  (it's done by the RPG before Level III generation, or by research tools
  like Py-ART's dealiasing algorithms).

- **SAILS/MESO-SAILS** -- Supplemental Adaptive Intra-Volume Low-level Scans
  interleave extra sweeps of the lowest elevation(s) throughout the volume.
  The parser handles this by tracking both radial status signals and elevation
  number changes, but downstream tools that assume a simple increasing-elevation
  sweep order need to be aware of the non-monotonic structure.

- **Dual-PRF / batch mode** -- some VCPs alternate PRFs within a single
  sweep for velocity dealiasing at the signal processing level.  The parser
  reads whatever moment data the RDA emits, but interpreting the velocity
  correctly may require knowledge of the VCP's PRF pattern.

---

## Design principles across all parsers

### No regex where tokenization suffices

The METAR, TAF, and sounding parsers all use whitespace splitting plus
`starts_with` / `parse::<T>()` rather than regex.  This avoids the cost of
regex compilation and the complexity of maintaining patterns for a
well-structured format.

### Lenient parsing

All text parsers skip unknown tokens rather than failing.  A METAR with a
non-standard remark or a Wyoming sounding with a missing column still produces
as much structured data as possible.

### Compile-time data

Station databases are `static` arrays, meaning they're part of the binary with
no runtime I/O.  This is a significant difference from Python packages that
load CSV or JSON station files at import time.

### Parallel I/O where profitable

The Level II bzip2 decompression uses Rayon for parallel block decompression.
The GRIB2 streaming parser enables progressive processing during downloads.
These patterns would be awkward to express in Python without multiprocessing.

### Borrowed-slice parsing

The GRIB2 parser works on `&[u8]` slices with offset-based reads
(`read_u16(data, offset)`).  Section data is sliced from the message buffer
without copying.  The Level II parser uses `Cursor<&Vec<u8>>` for sequential
reads.  In both cases, the goal is to minimize heap allocations during
parsing -- the main allocations are the output data vectors.
