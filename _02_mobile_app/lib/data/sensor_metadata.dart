/// Catalog of the 14 engine sensor channels used in this project.
///
/// Single source of truth for keys, display names, units, physical ranges,
/// and display formatting across the Live Telemetry grid, the All Sensors
/// sheet, and any future sensor-aware widgets.
///
/// Keys correspond to the JSON keys emitted in
/// `EngineTelemetryPayload.currentSensorReadings` by
/// `publisher_multi_machine.py`.
///
/// --- Data pipeline note ---
/// The publisher emits PHYSICAL (unscaled) sensor values in the
/// `current_sensor_readings` JSON field.  The MinMaxScaler (feature_range
/// = (-1, 1)) is applied internally to prepare the LSTM input window but
/// the scaled tensor is never sent over MQTT.  Therefore the values
/// received by the Flutter app are already in engineering units and
/// require no further inverse transformation.
///
/// --- Physical ranges ---
/// `minPhysical` and `maxPhysical` reflect the data_min_ / data_max_
/// attributes of the fitted `calibrated_scaler.gz` (sklearn MinMaxScaler),
/// which was trained on the full train_FD001.txt dataset.  They represent
/// the observed physical range of each sensor channel under FD001
/// conditions (one operating point, sea level) and can be used for
/// sparklines, health bars, or range indicators.
///
/// Sensor definitions are taken from Table 2 of:
///   Saxena et al., "Damage Propagation Modeling for Aircraft Engine
///   Run-to-Failure Simulation", PHM 2008.
///
/// The two operational settings (setting_1 = altitude, setting_2 = Mach)
/// are excluded because they represent flight conditions, not engine health
/// channels, and are not shown in the sensor detail sheet.
class SensorMetadata {
  /// JSON key in the telemetry payload (e.g. `s_2`, `s_7`).
  final String key;

  /// Engineering symbol from Table 2 (e.g. `T24`, `P30`, `Nf`).
  final String symbol;

  /// Human-friendly full name from Table 2.
  final String fullName;

  /// Engineering unit string. Empty string means dimensionless.
  final String unit;

  /// Observed physical minimum from the FD001 training set
  /// (scaler data_min_ attribute). Unit matches [unit].
  final double minPhysical;

  /// Observed physical maximum from the FD001 training set
  /// (scaler data_max_ attribute). Unit matches [unit].
  final double maxPhysical;

  /// Decimal places used when formatting the physical value for display.
  /// Chosen to match the precision of the sensor's useful range:
  ///   rpm / °R      -> 1 dp
  ///   psia          -> 2 dp
  ///   lbm/s / pps/psi -> 3 dp
  ///   dimensionless -> 4 dp
  final int decimalPlaces;

  const SensorMetadata({
    required this.key,
    required this.symbol,
    required this.fullName,
    required this.unit,
    required this.minPhysical,
    required this.maxPhysical,
    this.decimalPlaces = 2,
  });
}

/// Ordered list of the 14 engine sensor channels, excluding the two
/// operational-setting channels (altitude, Mach number).
///
/// Physical ranges (minPhysical / maxPhysical) are the exact data_min_ /
/// data_max_ values from `calibrated_scaler.gz`, cross-verified against
/// the raw FD001 test dataset (unit 34, cycles 1-n).
const List<SensorMetadata> kAllSensors = <SensorMetadata>[
  // --- Temperatures ---
  SensorMetadata(
    key: 's_2',
    symbol: 'T24',
    fullName: 'Total Temperature at LPC Outlet',
    unit: '°R',
    minPhysical: 641.27,
    maxPhysical: 644.53,
    decimalPlaces: 1,
  ),
  SensorMetadata(
    key: 's_3',
    symbol: 'T30',
    fullName: 'Total Temperature at HPC Outlet',
    unit: '°R',
    minPhysical: 1574.80,
    maxPhysical: 1612.11,
    decimalPlaces: 1,
  ),
  SensorMetadata(
    key: 's_4',
    symbol: 'T50',
    fullName: 'Total Temperature at LPT Outlet',
    unit: '°R',
    minPhysical: 1387.16,
    maxPhysical: 1438.51,
    decimalPlaces: 1,
  ),

  // --- Pressures ---
  SensorMetadata(
    key: 's_7',
    symbol: 'P30',
    fullName: 'Total Pressure at HPC Outlet',
    unit: 'psia',
    minPhysical: 550.70,
    maxPhysical: 555.57,
    decimalPlaces: 2,
  ),
  SensorMetadata(
    key: 's_11',
    symbol: 'Ps30',
    fullName: 'Static Pressure at HPC Outlet',
    unit: 'psia',
    minPhysical: 46.93,
    maxPhysical: 48.38,
    decimalPlaces: 2,
  ),

  // --- Speeds ---
  SensorMetadata(
    key: 's_8',
    symbol: 'Nf',
    fullName: 'Physical Fan Speed',
    unit: 'rpm',
    minPhysical: 2387.92,
    maxPhysical: 2388.32,
    decimalPlaces: 1,
  ),
  SensorMetadata(
    key: 's_9',
    symbol: 'Nc',
    fullName: 'Physical Core Speed',
    unit: 'rpm',
    minPhysical: 9033.22,
    maxPhysical: 9203.22,
    decimalPlaces: 1,
  ),
  SensorMetadata(
    key: 's_13',
    symbol: 'NRf',
    fullName: 'Corrected Fan Speed',
    unit: 'rpm',
    minPhysical: 2387.92,
    maxPhysical: 2388.35,
    decimalPlaces: 1,
  ),
  SensorMetadata(
    // Paper Table 2 and scaler range 8110-8259 confirm the unit is rpm,
    // not the percentage stated in the integration_notes sensor_map.json.
    key: 's_14',
    symbol: 'NRc',
    fullName: 'Corrected Core Speed',
    unit: 'rpm',
    minPhysical: 8110.93,
    maxPhysical: 8259.42,
    decimalPlaces: 1,
  ),

  // --- Flow ratios ---
  SensorMetadata(
    key: 's_12',
    symbol: 'phi',
    fullName: 'Fuel Flow / Ps30 Ratio',
    unit: 'pps/psi',
    minPhysical: 519.48,
    maxPhysical: 523.26,
    decimalPlaces: 3,
  ),

  // --- Dimensionless ---
  SensorMetadata(
    key: 's_15',
    symbol: 'BPR',
    fullName: 'Bypass Ratio',
    unit: '',
    minPhysical: 8.3428,
    maxPhysical: 8.5462,
    decimalPlaces: 4,
  ),
  SensorMetadata(
    key: 's_17',
    symbol: 'htBleed',
    fullName: 'Bleed Enthalpy',
    unit: '',
    minPhysical: 389.0,
    maxPhysical: 399.0,
    decimalPlaces: 2,
  ),

  // --- Coolant flows ---
  SensorMetadata(
    key: 's_20',
    symbol: 'W31',
    fullName: 'HPT Coolant Bleed Flow',
    unit: 'lbm/s',
    minPhysical: 38.23,
    maxPhysical: 39.29,
    decimalPlaces: 3,
  ),
  SensorMetadata(
    key: 's_21',
    symbol: 'W32',
    fullName: 'LPT Coolant Bleed Flow',
    unit: 'lbm/s',
    minPhysical: 22.9562,
    maxPhysical: 23.6005,
    decimalPlaces: 3,
  ),
];
