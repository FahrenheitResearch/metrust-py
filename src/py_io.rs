use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;

// ─── Level3File ─────────────────────────────────────────────────────────

/// Parsed NEXRAD Level 3 (NIDS) product file.
#[pyclass(name = "Level3File")]
#[derive(Clone)]
struct PyLevel3File {
    inner: metrust::io::level3::Level3File,
}

#[pymethods]
impl PyLevel3File {
    /// Parse a Level 3 product from a file path.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner =
            metrust::io::level3::Level3File::from_file(std::path::Path::new(path)).map_err(
                |e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()),
            )?;
        Ok(Self { inner })
    }

    /// Parse a Level 3 product from raw bytes.
    #[staticmethod]
    #[pyo3(text_signature = "(data)")]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = metrust::io::level3::Level3File::from_bytes(data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Numeric product code (e.g. 94 = base reflectivity 0.5 deg).
    #[getter]
    fn product_code(&self) -> u16 {
        self.inner.product_code
    }

    /// Source (radar) ID.
    #[getter]
    fn source_id(&self) -> u16 {
        self.inner.source_id
    }

    /// Latitude of the radar (degrees).
    #[getter]
    fn latitude(&self) -> f64 {
        self.inner.latitude
    }

    /// Longitude of the radar (degrees).
    #[getter]
    fn longitude(&self) -> f64 {
        self.inner.longitude
    }

    /// Height of the radar (feet).
    #[getter]
    fn height(&self) -> f64 {
        self.inner.height
    }

    /// Volume scan time as "YYYY-MM-DD HH:MM:SS".
    #[getter]
    fn volume_time(&self) -> String {
        self.inner.volume_time.clone()
    }

    /// Flattened data values as a numpy array (row-major: radial * bin).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.data.clone().into_pyarray(py)
    }

    /// Number of range bins per radial.
    #[getter]
    fn num_bins(&self) -> usize {
        self.inner.num_bins
    }

    /// Number of radials in the product.
    #[getter]
    fn num_radials(&self) -> usize {
        self.inner.num_radials
    }

    fn __repr__(&self) -> String {
        format!(
            "Level3File(product_code={}, source_id={}, lat={:.3}, lon={:.3}, time='{}', bins={}, radials={})",
            self.inner.product_code,
            self.inner.source_id,
            self.inner.latitude,
            self.inner.longitude,
            self.inner.volume_time,
            self.inner.num_bins,
            self.inner.num_radials,
        )
    }
}

// ─── Metar ──────────────────────────────────────────────────────────────

/// A parsed METAR aviation weather observation.
#[pyclass(name = "Metar")]
#[derive(Clone)]
struct PyMetar {
    inner: metrust::io::metar::Metar,
}

#[pymethods]
impl PyMetar {
    /// Parse a single METAR observation string.
    #[staticmethod]
    #[pyo3(text_signature = "(text)")]
    fn parse(text: &str) -> PyResult<Self> {
        let inner = metrust::io::metar::Metar::parse(text)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// ICAO station identifier (e.g. "KATL").
    #[getter]
    fn station(&self) -> String {
        self.inner.station.clone()
    }

    /// Observation time as "DDHHMMz".
    #[getter]
    fn time(&self) -> String {
        self.inner.time.clone()
    }

    /// Wind direction in degrees (None for calm or variable).
    #[getter]
    fn wind_direction(&self) -> Option<f64> {
        self.inner.wind_direction
    }

    /// Sustained wind speed in knots.
    #[getter]
    fn wind_speed(&self) -> Option<f64> {
        self.inner.wind_speed
    }

    /// Gust speed in knots, if reported.
    #[getter]
    fn wind_gust(&self) -> Option<f64> {
        self.inner.wind_gust
    }

    /// Prevailing visibility in statute miles.
    #[getter]
    fn visibility(&self) -> Option<f64> {
        self.inner.visibility
    }

    /// Temperature in Celsius.
    #[getter]
    fn temperature(&self) -> Option<f64> {
        self.inner.temperature
    }

    /// Dewpoint in Celsius.
    #[getter]
    fn dewpoint(&self) -> Option<f64> {
        self.inner.dewpoint
    }

    /// Altimeter setting in inches of mercury.
    #[getter]
    fn altimeter(&self) -> Option<f64> {
        self.inner.altimeter
    }

    /// Sky condition layers as list of (cover_type, height_in_feet) tuples.
    /// Cover types: CLR, SKC, FEW, SCT, BKN, OVC, VV.
    #[getter]
    fn sky_cover(&self) -> Vec<(String, Option<u32>)> {
        self.inner.sky_cover.clone()
    }

    /// Present weather phenomena tokens (e.g. "RA", "+TSRA", "-SN").
    #[getter]
    fn weather(&self) -> Vec<String> {
        self.inner.weather.clone()
    }

    /// Original raw METAR string.
    #[getter]
    fn raw(&self) -> String {
        self.inner.raw.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Metar(station='{}', time='{}', temp={}, dewpt={}, wind={}/{}kt)",
            self.inner.station,
            self.inner.time,
            self.inner
                .temperature
                .map_or("None".to_string(), |v| format!("{:.1}C", v)),
            self.inner
                .dewpoint
                .map_or("None".to_string(), |v| format!("{:.1}C", v)),
            self.inner
                .wind_direction
                .map_or("VRB".to_string(), |v| format!("{:.0}", v)),
            self.inner
                .wind_speed
                .map_or("?".to_string(), |v| format!("{:.0}", v)),
        )
    }
}

/// Parse a single METAR observation string.
#[pyfunction]
#[pyo3(text_signature = "(text)")]
fn parse_metar(text: &str) -> PyResult<PyMetar> {
    PyMetar::parse(text)
}

/// Parse a multi-line string containing one METAR per line.
///
/// Blank lines and lines starting with '#' are skipped.  Returns all
/// successfully parsed METARs (silently ignoring unparseable lines).
#[pyfunction]
#[pyo3(text_signature = "(content)")]
fn parse_metar_file(content: &str) -> Vec<PyMetar> {
    metrust::io::metar::parse_metar_file(content)
        .into_iter()
        .map(|inner| PyMetar { inner })
        .collect()
}

// ─── StationInfo ────────────────────────────────────────────────────────

/// Metadata for a single weather observation station.
#[pyclass(name = "StationInfo")]
#[derive(Clone)]
struct PyStationInfo {
    inner: metrust::io::station::StationInfo,
}

#[pymethods]
impl PyStationInfo {
    /// Create a new StationInfo.
    #[new]
    #[pyo3(text_signature = "(id, name, state, country, latitude, longitude, elevation)")]
    fn new(
        id: String,
        name: String,
        state: String,
        country: String,
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Self {
        Self {
            inner: metrust::io::station::StationInfo {
                id,
                name,
                state,
                country,
                latitude,
                longitude,
                elevation,
            },
        }
    }

    /// Station identifier (e.g. "KATL", "72219").
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Human-readable station name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// State or province code (may be empty for non-US stations).
    #[getter]
    fn state(&self) -> String {
        self.inner.state.clone()
    }

    /// Country code (e.g. "US").
    #[getter]
    fn country(&self) -> String {
        self.inner.country.clone()
    }

    /// Latitude in decimal degrees.
    #[getter]
    fn latitude(&self) -> f64 {
        self.inner.latitude
    }

    /// Longitude in decimal degrees.
    #[getter]
    fn longitude(&self) -> f64 {
        self.inner.longitude
    }

    /// Elevation in metres above mean sea level.
    #[getter]
    fn elevation(&self) -> f64 {
        self.inner.elevation
    }

    fn __repr__(&self) -> String {
        format!(
            "StationInfo(id='{}', name='{}', lat={:.4}, lon={:.4}, elev={:.1}m)",
            self.inner.id,
            self.inner.name,
            self.inner.latitude,
            self.inner.longitude,
            self.inner.elevation,
        )
    }
}

// ─── StationLookup ──────────────────────────────────────────────────────

/// In-memory station database with spatial lookup.
#[pyclass(name = "StationLookup")]
#[derive(Clone)]
struct PyStationLookup {
    inner: metrust::io::station::StationLookup,
}

#[pymethods]
impl PyStationLookup {
    /// Create an empty station lookup table.
    #[new]
    fn new() -> Self {
        Self {
            inner: metrust::io::station::StationLookup::new(),
        }
    }

    /// Create a lookup table from a list of StationInfo entries.
    #[staticmethod]
    #[pyo3(text_signature = "(entries)")]
    fn from_entries(entries: Vec<PyStationInfo>) -> Self {
        let rust_entries: Vec<metrust::io::station::StationInfo> =
            entries.into_iter().map(|e| e.inner).collect();
        Self {
            inner: metrust::io::station::StationLookup::from_entries(rust_entries),
        }
    }

    /// Return the number of stations in the table.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return the number of stations in the table.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Return true if the table is empty.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Add a station to the table.
    #[pyo3(text_signature = "(self, station)")]
    fn add(&mut self, station: PyStationInfo) {
        self.inner.add(station.inner);
    }

    /// Look up a station by its ID (case-insensitive).
    #[pyo3(text_signature = "(self, id)")]
    fn lookup(&self, id: &str) -> Option<PyStationInfo> {
        self.inner.lookup(id).map(|s| PyStationInfo {
            inner: s.clone(),
        })
    }

    /// Find the station nearest to a given latitude/longitude.
    #[pyo3(text_signature = "(self, lat, lon)")]
    fn nearest(&self, lat: f64, lon: f64) -> Option<PyStationInfo> {
        self.inner.nearest(lat, lon).map(|s| PyStationInfo {
            inner: s.clone(),
        })
    }

    /// Find all stations within `radius_km` of a given point.
    /// Results are sorted by distance (nearest first).
    #[pyo3(text_signature = "(self, lat, lon, radius_km)")]
    fn within_radius(&self, lat: f64, lon: f64, radius_km: f64) -> Vec<PyStationInfo> {
        self.inner
            .within_radius(lat, lon, radius_km)
            .into_iter()
            .map(|s| PyStationInfo {
                inner: s.clone(),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("StationLookup(stations={})", self.inner.len())
    }
}

// ─── Module registration ────────────────────────────────────────────────

pub fn register(_py: Python, parent: &Bound<'_, PyModule>) -> PyResult<()> {
    parent.add_class::<PyLevel3File>()?;
    parent.add_class::<PyMetar>()?;
    parent.add_class::<PyStationInfo>()?;
    parent.add_class::<PyStationLookup>()?;
    parent.add_function(wrap_pyfunction!(parse_metar, parent)?)?;
    parent.add_function(wrap_pyfunction!(parse_metar_file, parent)?)?;
    Ok(())
}
