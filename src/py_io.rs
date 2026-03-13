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

// ─── GiniFile ──────────────────────────────────────────────────────────

/// A parsed GINI satellite image file.
#[pyclass(name = "GiniFile")]
#[derive(Clone)]
struct PyGiniFile {
    inner: metrust::io::gini::GiniFile,
}

#[pymethods]
impl PyGiniFile {
    /// Parse a GINI file from a file path.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = metrust::io::gini::GiniFile::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Parse a GINI file from raw bytes.
    #[staticmethod]
    #[pyo3(text_signature = "(data)")]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = metrust::io::gini::GiniFile::from_bytes(data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Source indicator.
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// Creating entity (satellite platform).
    #[getter]
    fn creating_entity(&self) -> String {
        self.inner.creating_entity.clone()
    }

    /// Image sector name.
    #[getter]
    fn sector(&self) -> String {
        self.inner.sector.clone()
    }

    /// Channel / physical element name.
    #[getter]
    fn channel(&self) -> String {
        self.inner.channel.clone()
    }

    /// Number of pixels per scan line (columns).
    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx
    }

    /// Number of scan lines (rows).
    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny
    }

    /// Map projection name.
    #[getter]
    fn projection(&self) -> String {
        self.inner.projection.clone()
    }

    /// Latitude of the first grid point (degrees).
    #[getter]
    fn lat1(&self) -> f64 {
        self.inner.lat1
    }

    /// Longitude of the first grid point (degrees).
    #[getter]
    fn lon1(&self) -> f64 {
        self.inner.lon1
    }

    /// Grid spacing in x (km).
    #[getter]
    fn dx(&self) -> f64 {
        self.inner.dx
    }

    /// Grid spacing in y (km).
    #[getter]
    fn dy(&self) -> f64 {
        self.inner.dy
    }

    /// Image datetime as "YYYY-MM-DD HH:MM:SS".
    #[getter]
    fn datetime(&self) -> String {
        self.inner.datetime.clone()
    }

    /// Raw pixel data as a numpy uint8 array.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        self.inner.data.clone().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "GiniFile(entity='{}', sector='{}', channel='{}', {}x{}, proj='{}', time='{}')",
            self.inner.creating_entity,
            self.inner.sector,
            self.inner.channel,
            self.inner.nx,
            self.inner.ny,
            self.inner.projection,
            self.inner.datetime,
        )
    }
}

// ─── SurfaceBulletinFeature / WPC parser ────────────────────────────────

/// A decoded feature from a WPC coded surface bulletin.
#[pyclass(name = "SurfaceBulletinFeature")]
#[derive(Clone)]
struct PySurfaceBulletinFeature {
    inner: metrust::io::wpc::SurfaceBulletinFeature,
}

#[pymethods]
impl PySurfaceBulletinFeature {
    /// Feature type: "HIGH", "LOW", "WARM", "COLD", "STNRY", "OCFNT", "TROF".
    #[getter]
    fn feature_type(&self) -> String {
        self.inner.feature_type.clone()
    }

    /// Lat/lon pairs as a list of (lat, lon) tuples.
    #[getter]
    fn points(&self) -> Vec<(f64, f64)> {
        self.inner.points.clone()
    }

    /// Central pressure (mb) for HIGH/LOW, or None for fronts.
    #[getter]
    fn value(&self) -> Option<f64> {
        self.inner.value
    }

    /// Strength qualifier (e.g. "WK", "MDT", "STG"), if present.
    #[getter]
    fn strength(&self) -> Option<String> {
        self.inner.strength.clone()
    }

    /// Valid time string from the bulletin.
    #[getter]
    fn valid_time(&self) -> Option<String> {
        self.inner.valid_time.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SurfaceBulletinFeature(type='{}', points={}, value={:?})",
            self.inner.feature_type,
            self.inner.points.len(),
            self.inner.value,
        )
    }
}

/// Parse a WPC coded surface bulletin text into a list of features.
#[pyfunction]
#[pyo3(text_signature = "(text)")]
fn parse_wpc_surface_bulletin(text: &str) -> Vec<PySurfaceBulletinFeature> {
    metrust::io::parse_wpc_surface_bulletin(text)
        .into_iter()
        .map(|inner| PySurfaceBulletinFeature { inner })
        .collect()
}

/// Return True if the given NEXRAD VCP number is a precipitation mode.
#[pyfunction]
#[pyo3(text_signature = "(vcp)")]
fn is_precip_mode(vcp: u16) -> bool {
    metrust::io::is_precip_mode(vcp)
}

// ─── GempakGrid ─────────────────────────────────────────────────────────

/// A parsed GEMPAK grid file.
#[pyclass(name = "GempakGrid")]
#[derive(Clone)]
struct PyGempakGrid {
    inner: metrust::io::gempak::GempakGrid,
}

/// A single grid record from a GEMPAK file.
#[pyclass(name = "GempakGridRecord")]
#[derive(Clone)]
struct PyGempakGridRecord {
    inner: metrust::io::gempak::GempakGridRecord,
}

#[pymethods]
impl PyGempakGrid {
    /// Open and parse a GEMPAK grid file from a file path.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = metrust::io::gempak::GempakGrid::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        Ok(Self { inner })
    }

    /// Parse a GEMPAK grid file from raw bytes.
    #[staticmethod]
    #[pyo3(text_signature = "(data)")]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = metrust::io::gempak::GempakGrid::from_bytes(data)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        Ok(Self { inner })
    }

    /// Data source description (e.g. "model", "grid").
    #[getter]
    fn source(&self) -> String {
        self.inner.source.clone()
    }

    /// File type (always "grid" for grid files).
    #[getter]
    fn grid_type(&self) -> String {
        self.inner.grid_type.clone()
    }

    /// Number of grid columns (x-dimension).
    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx
    }

    /// Number of grid rows (y-dimension).
    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny
    }

    /// Projection name from the navigation block (e.g. "LCC", "CED").
    #[getter]
    fn projection(&self) -> Option<String> {
        self.inner.navigation.as_ref().map(|n| n.projection.clone())
    }

    /// Lower-left latitude of the grid domain (degrees).
    #[getter]
    fn lower_left_lat(&self) -> Option<f64> {
        self.inner.navigation.as_ref().map(|n| n.lower_left_lat)
    }

    /// Lower-left longitude of the grid domain (degrees).
    #[getter]
    fn lower_left_lon(&self) -> Option<f64> {
        self.inner.navigation.as_ref().map(|n| n.lower_left_lon)
    }

    /// Upper-right latitude of the grid domain (degrees).
    #[getter]
    fn upper_right_lat(&self) -> Option<f64> {
        self.inner.navigation.as_ref().map(|n| n.upper_right_lat)
    }

    /// Upper-right longitude of the grid domain (degrees).
    #[getter]
    fn upper_right_lon(&self) -> Option<f64> {
        self.inner.navigation.as_ref().map(|n| n.upper_right_lon)
    }

    /// Number of grids in the file.
    #[getter]
    fn num_grids(&self) -> usize {
        self.inner.grids.len()
    }

    /// List of all grid records.
    #[getter]
    fn grids(&self) -> Vec<PyGempakGridRecord> {
        self.inner
            .grids
            .iter()
            .map(|g| PyGempakGridRecord { inner: g.clone() })
            .collect()
    }

    /// Return a summary string for each grid in the file.
    fn grid_info(&self) -> Vec<String> {
        self.inner.grid_info()
    }

    /// Find grids matching a parameter name (case-insensitive).
    #[pyo3(text_signature = "(self, parameter)")]
    fn find_grids(&self, parameter: &str) -> Vec<PyGempakGridRecord> {
        self.inner
            .find_grids(parameter)
            .into_iter()
            .map(|g| PyGempakGridRecord { inner: g.clone() })
            .collect()
    }

    /// Get a specific grid by parameter name and level.
    #[pyo3(text_signature = "(self, parameter, level)")]
    fn get_grid(&self, parameter: &str, level: f64) -> Option<PyGempakGridRecord> {
        self.inner
            .get_grid(parameter, level)
            .map(|g| PyGempakGridRecord { inner: g.clone() })
    }

    fn __repr__(&self) -> String {
        format!(
            "GempakGrid(source='{}', nx={}, ny={}, grids={}, proj={})",
            self.inner.source,
            self.inner.nx,
            self.inner.ny,
            self.inner.grids.len(),
            self.inner
                .navigation
                .as_ref()
                .map_or("None".to_string(), |n| format!("'{}'", n.projection)),
        )
    }
}

#[pymethods]
impl PyGempakGridRecord {
    /// Grid number (index within the file).
    #[getter]
    fn grid_number(&self) -> usize {
        self.inner.grid_number
    }

    /// Parameter name (e.g. "TMPK", "HGHT").
    #[getter]
    fn parameter(&self) -> String {
        self.inner.parameter.clone()
    }

    /// Primary vertical level value.
    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }

    /// Secondary vertical level (for layers), or -1.
    #[getter]
    fn level2(&self) -> f64 {
        self.inner.level2
    }

    /// Vertical coordinate type (e.g. "PRES", "HGHT").
    #[getter]
    fn coordinate(&self) -> String {
        self.inner.coordinate.clone()
    }

    /// Valid datetime string.
    #[getter]
    fn time(&self) -> String {
        self.inner.time.clone()
    }

    /// Secondary datetime string.
    #[getter]
    fn time2(&self) -> String {
        self.inner.time2.clone()
    }

    /// Forecast type (e.g. "analysis", "forecast").
    #[getter]
    fn forecast_type(&self) -> String {
        self.inner.forecast_type.clone()
    }

    /// Forecast time offset as "HHH:MM".
    #[getter]
    fn forecast_time(&self) -> String {
        self.inner.forecast_time.clone()
    }

    /// Grid data values as a numpy float64 array (row-major, ny*nx elements).
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.data.clone().into_pyarray(py)
    }

    /// Number of data values.
    #[getter]
    fn data_length(&self) -> usize {
        self.inner.data.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "GempakGridRecord(param='{}', level={}, coord='{}', time='{}', ftime='{}', data_len={})",
            self.inner.parameter,
            self.inner.level,
            self.inner.coordinate,
            self.inner.time,
            self.inner.forecast_time,
            self.inner.data.len(),
        )
    }
}

// ─── GempakSounding ─────────────────────────────────────────────────────

/// A parsed GEMPAK sounding file.
#[pyclass(name = "GempakSounding")]
#[derive(Clone)]
struct PyGempakSounding {
    inner: metrust::io::gempak_sounding::GempakSounding,
}

/// A station entry from a GEMPAK sounding file.
#[pyclass(name = "GempakSoundingStation")]
#[derive(Clone)]
struct PyGempakSoundingStation {
    inner: metrust::io::gempak_sounding::GempakStation,
}

/// A single vertical sounding profile.
#[pyclass(name = "SoundingData")]
#[derive(Clone)]
struct PySoundingData {
    inner: metrust::io::gempak_sounding::SoundingData,
}

#[pymethods]
impl PyGempakSounding {
    /// Open and parse a GEMPAK sounding file from a file path.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = metrust::io::gempak_sounding::GempakSounding::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        Ok(Self { inner })
    }

    /// Whether the file uses merged (SNDT) format.
    #[getter]
    fn merged(&self) -> bool {
        self.inner.merged
    }

    /// Stations in the sounding file.
    #[getter]
    fn stations(&self) -> Vec<PyGempakSoundingStation> {
        self.inner
            .stations
            .iter()
            .map(|s| PyGempakSoundingStation { inner: s.clone() })
            .collect()
    }

    /// All sounding profiles in the file.
    #[getter]
    fn soundings(&self) -> Vec<PySoundingData> {
        self.inner
            .soundings
            .iter()
            .map(|s| PySoundingData { inner: s.clone() })
            .collect()
    }

    /// Number of soundings.
    #[getter]
    fn num_soundings(&self) -> usize {
        self.inner.soundings.len()
    }

    /// Number of stations.
    #[getter]
    fn num_stations(&self) -> usize {
        self.inner.stations.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "GempakSounding(stations={}, soundings={}, merged={})",
            self.inner.stations.len(),
            self.inner.soundings.len(),
            self.inner.merged,
        )
    }
}

#[pymethods]
impl PyGempakSoundingStation {
    /// Station identifier.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Latitude (degrees).
    #[getter]
    fn lat(&self) -> f64 {
        self.inner.lat
    }

    /// Longitude (degrees).
    #[getter]
    fn lon(&self) -> f64 {
        self.inner.lon
    }

    /// Elevation (m).
    #[getter]
    fn elevation(&self) -> f64 {
        self.inner.elevation
    }

    /// State code.
    #[getter]
    fn state(&self) -> String {
        self.inner.state.clone()
    }

    /// Country code.
    #[getter]
    fn country(&self) -> String {
        self.inner.country.clone()
    }

    /// WMO station number.
    #[getter]
    fn station_number(&self) -> i32 {
        self.inner.station_number
    }

    fn __repr__(&self) -> String {
        format!(
            "GempakSoundingStation(id='{}', lat={:.2}, lon={:.2}, elev={:.0})",
            self.inner.id, self.inner.lat, self.inner.lon, self.inner.elevation,
        )
    }
}

#[pymethods]
impl PySoundingData {
    /// Station identifier.
    #[getter]
    fn station_id(&self) -> String {
        self.inner.station_id.clone()
    }

    /// Observation time string.
    #[getter]
    fn time(&self) -> String {
        self.inner.time.clone()
    }

    /// Pressure levels (hPa) as a numpy array.
    #[getter]
    fn pressure<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.pressure.clone().into_pyarray(py)
    }

    /// Temperature values as a numpy array.
    #[getter]
    fn temperature<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.temperature.clone().into_pyarray(py)
    }

    /// Dewpoint values as a numpy array.
    #[getter]
    fn dewpoint<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.dewpoint.clone().into_pyarray(py)
    }

    /// Wind direction (degrees) as a numpy array.
    #[getter]
    fn wind_direction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.wind_direction.clone().into_pyarray(py)
    }

    /// Wind speed as a numpy array.
    #[getter]
    fn wind_speed<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.wind_speed.clone().into_pyarray(py)
    }

    /// Geopotential height as a numpy array.
    #[getter]
    fn height<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.height.clone().into_pyarray(py)
    }

    /// Number of vertical levels.
    #[getter]
    fn num_levels(&self) -> usize {
        self.inner.pressure.len()
    }

    /// List of all parameter names available in this sounding.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameters.keys().cloned().collect()
    }

    /// Get a named parameter as a numpy array. Returns None if not available.
    #[pyo3(text_signature = "(self, name)")]
    fn get_parameter<'py>(&self, py: Python<'py>, name: &str) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.parameters.get(name).map(|v| v.clone().into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "SoundingData(station='{}', time='{}', levels={})",
            self.inner.station_id,
            self.inner.time,
            self.inner.pressure.len(),
        )
    }
}

// ─── GempakSurface ──────────────────────────────────────────────────────

/// A parsed GEMPAK surface observation file.
#[pyclass(name = "GempakSurface")]
#[derive(Clone)]
struct PyGempakSurface {
    inner: metrust::io::gempak_surface::GempakSurface,
}

/// A station from a GEMPAK surface file.
#[pyclass(name = "GempakSurfaceStation")]
#[derive(Clone)]
struct PyGempakSurfaceStation {
    inner: metrust::io::gempak_surface::SurfaceStation,
}

/// A single surface observation.
#[pyclass(name = "SurfaceObs")]
#[derive(Clone)]
struct PySurfaceObs {
    inner: metrust::io::gempak_surface::SurfaceObs,
}

#[pymethods]
impl PyGempakSurface {
    /// Open and parse a GEMPAK surface file from a file path.
    #[staticmethod]
    #[pyo3(text_signature = "(path)")]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = metrust::io::gempak_surface::GempakSurface::from_file(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;
        Ok(Self { inner })
    }

    /// Surface file type: "standard", "ship", or "climate".
    #[getter]
    fn surface_type(&self) -> String {
        self.inner.surface_type.clone()
    }

    /// Stations in the surface file.
    #[getter]
    fn stations(&self) -> Vec<PyGempakSurfaceStation> {
        self.inner
            .stations
            .iter()
            .map(|s| PyGempakSurfaceStation { inner: s.clone() })
            .collect()
    }

    /// All surface observations in the file.
    #[getter]
    fn observations(&self) -> Vec<PySurfaceObs> {
        self.inner
            .observations
            .iter()
            .map(|o| PySurfaceObs { inner: o.clone() })
            .collect()
    }

    /// Number of observations.
    #[getter]
    fn num_observations(&self) -> usize {
        self.inner.observations.len()
    }

    /// Number of stations.
    #[getter]
    fn num_stations(&self) -> usize {
        self.inner.stations.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "GempakSurface(type='{}', stations={}, observations={})",
            self.inner.surface_type,
            self.inner.stations.len(),
            self.inner.observations.len(),
        )
    }
}

#[pymethods]
impl PyGempakSurfaceStation {
    /// Station identifier.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Latitude (degrees).
    #[getter]
    fn lat(&self) -> f64 {
        self.inner.lat
    }

    /// Longitude (degrees).
    #[getter]
    fn lon(&self) -> f64 {
        self.inner.lon
    }

    /// Elevation (m).
    #[getter]
    fn elevation(&self) -> f64 {
        self.inner.elevation
    }

    /// State code.
    #[getter]
    fn state(&self) -> String {
        self.inner.state.clone()
    }

    /// Country code.
    #[getter]
    fn country(&self) -> String {
        self.inner.country.clone()
    }

    /// WMO station number.
    #[getter]
    fn station_number(&self) -> i32 {
        self.inner.station_number
    }

    fn __repr__(&self) -> String {
        format!(
            "GempakSurfaceStation(id='{}', lat={:.2}, lon={:.2}, elev={:.0})",
            self.inner.id, self.inner.lat, self.inner.lon, self.inner.elevation,
        )
    }
}

#[pymethods]
impl PySurfaceObs {
    /// Station identifier.
    #[getter]
    fn station_id(&self) -> String {
        self.inner.station_id.clone()
    }

    /// Observation time string.
    #[getter]
    fn time(&self) -> String {
        self.inner.time.clone()
    }

    /// Temperature (units depend on file: TMPC=Celsius, TMPF=Fahrenheit).
    #[getter]
    fn temperature(&self) -> Option<f64> {
        self.inner.temperature
    }

    /// Dewpoint temperature.
    #[getter]
    fn dewpoint(&self) -> Option<f64> {
        self.inner.dewpoint
    }

    /// Wind direction (degrees).
    #[getter]
    fn wind_direction(&self) -> Option<f64> {
        self.inner.wind_direction
    }

    /// Wind speed.
    #[getter]
    fn wind_speed(&self) -> Option<f64> {
        self.inner.wind_speed
    }

    /// Pressure (PMSL, PRES, or ALTI).
    #[getter]
    fn pressure(&self) -> Option<f64> {
        self.inner.pressure
    }

    /// Visibility.
    #[getter]
    fn visibility(&self) -> Option<f64> {
        self.inner.visibility
    }

    /// Sky cover string.
    #[getter]
    fn sky_cover(&self) -> Option<String> {
        self.inner.sky_cover.clone()
    }

    /// List of all parameter names available in this observation.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.inner.parameters.keys().cloned().collect()
    }

    /// Get a named parameter value. Returns None if not available.
    #[pyo3(text_signature = "(self, name)")]
    fn get_parameter(&self, name: &str) -> Option<f64> {
        self.inner.parameters.get(name).copied()
    }

    fn __repr__(&self) -> String {
        format!(
            "SurfaceObs(station='{}', time='{}', temp={}, dewpt={}, wind={}/{})",
            self.inner.station_id,
            self.inner.time,
            self.inner.temperature.map_or("None".to_string(), |v| format!("{:.1}", v)),
            self.inner.dewpoint.map_or("None".to_string(), |v| format!("{:.1}", v)),
            self.inner.wind_direction.map_or("None".to_string(), |v| format!("{:.0}", v)),
            self.inner.wind_speed.map_or("None".to_string(), |v| format!("{:.0}", v)),
        )
    }
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
    parent.add_class::<PyGiniFile>()?;
    parent.add_class::<PySurfaceBulletinFeature>()?;
    parent.add_class::<PyMetar>()?;
    parent.add_class::<PyGempakGrid>()?;
    parent.add_class::<PyGempakGridRecord>()?;
    parent.add_class::<PyGempakSounding>()?;
    parent.add_class::<PyGempakSoundingStation>()?;
    parent.add_class::<PySoundingData>()?;
    parent.add_class::<PyGempakSurface>()?;
    parent.add_class::<PyGempakSurfaceStation>()?;
    parent.add_class::<PySurfaceObs>()?;
    parent.add_class::<PyStationInfo>()?;
    parent.add_class::<PyStationLookup>()?;
    parent.add_function(wrap_pyfunction!(parse_metar, parent)?)?;
    parent.add_function(wrap_pyfunction!(parse_metar_file, parent)?)?;
    parent.add_function(wrap_pyfunction!(parse_wpc_surface_bulletin, parent)?)?;
    parent.add_function(wrap_pyfunction!(is_precip_mode, parent)?)?;
    Ok(())
}
