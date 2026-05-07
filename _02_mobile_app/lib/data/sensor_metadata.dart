/// Catalog of all telemetry sensor metadata used by the app.
///
/// Single source of truth so the Live Telemetry quick-view, the
/// "All sensors" sheet, and any future sensor-aware widgets agree on
/// keys, display names, units, and ordering.
///
/// Keys correspond to the JSON keys emitted in
/// `EngineTelemetryPayload.currentSensorReadings` and the publisher's
/// payload (see `publisher_multi_machine.py`).
class SensorMetadata {
  /// JSON key in the telemetry payload (e.g. `s_2`, `setting_1`).
  final String key;

  /// Engineering symbol (e.g. `T24`, `Ps30`, `BPR`).
  final String symbol;

  /// Human-friendly full name.
  final String fullName;

  /// Display unit. Empty string means the value is dimensionless.
  final String unit;

  const SensorMetadata({
    required this.key,
    required this.symbol,
    required this.fullName,
    required this.unit,
  });
}

/// Complete sensor list from the C-MAPSS dataset, as supplied in
/// `sensors_details.txt`. Order is preserved for display.
const List<SensorMetadata> kAllSensors = <SensorMetadata>[
  SensorMetadata(key: 'setting_1', symbol: 'alt',     fullName: 'Altitude',                          unit: 'ft'),
  SensorMetadata(key: 'setting_2', symbol: 'Mach',    fullName: 'Mach Number',                       unit: ''),
  SensorMetadata(key: 's_2',       symbol: 'T24',     fullName: 'Total Temperature at LPC Outlet',   unit: '°R'),
  SensorMetadata(key: 's_3',       symbol: 'T30',     fullName: 'Total Temperature at HPC Outlet',   unit: '°R'),
  SensorMetadata(key: 's_4',       symbol: 'T50',     fullName: 'Total Temperature at LPT Outlet',   unit: '°R'),
  SensorMetadata(key: 's_7',       symbol: 'P30',     fullName: 'Total Pressure at HPC Outlet',      unit: 'psia'),
  SensorMetadata(key: 's_8',       symbol: 'Nf',      fullName: 'Physical Fan Speed',                unit: 'rpm'),
  SensorMetadata(key: 's_9',       symbol: 'Nc',      fullName: 'Physical Core Speed',               unit: 'rpm'),
  SensorMetadata(key: 's_11',      symbol: 'Ps30',    fullName: 'Static Pressure at HPC Outlet',     unit: 'psia'),
  SensorMetadata(key: 's_12',      symbol: 'phi',     fullName: 'Fuel Flow / Ps30 Ratio',            unit: 'pps/psi'),
  SensorMetadata(key: 's_13',      symbol: 'NRf',     fullName: 'Corrected Fan Speed',               unit: 'rpm'),
  SensorMetadata(key: 's_14',      symbol: 'NRc',     fullName: 'Corrected Core Speed',              unit: '%'),
  SensorMetadata(key: 's_15',      symbol: 'BPR',     fullName: 'Bypass Ratio',                      unit: ''),
  SensorMetadata(key: 's_17',      symbol: 'htBleed', fullName: 'Bleed Enthalpy',                    unit: ''),
  SensorMetadata(key: 's_20',      symbol: 'W31',     fullName: 'HPT Coolant Bleed Flow',            unit: 'lbm/s'),
  SensorMetadata(key: 's_21',      symbol: 'W32',     fullName: 'LPT Coolant Bleed Flow',            unit: 'lbm/s'),
];
