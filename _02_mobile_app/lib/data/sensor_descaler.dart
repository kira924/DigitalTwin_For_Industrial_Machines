// Data-driven inverse min-max descaler for C-MAPSS sensor channels.
//
// Physical bounds were computed from the full FD001 training dataset located
// at 04_3D_Simulation/workspace/train_FD001.txt (100 engines, 20,631 cycles,
// one operating point at sea level, HPC degradation fault mode).
//
// The C-MAPSS data pipeline normalizes raw sensor readings to the range
// [-1, 1] before feeding them into the LSTM model.  This class reverses
// that transform so the UI can display human-readable physical values.
//
// Inverse min-max formula:
//   x_physical = (x_scaled + 1) / 2  *  (dataMax - dataMin)  +  dataMin
//
// Sensor-to-key correspondence (Table 2, Saxena et al. PHM 2008):
//   s_2  -> T24     Total Temperature at LPC Outlet         [degrees R]
//   s_3  -> T30     Total Temperature at HPC Outlet         [degrees R]
//   s_4  -> T50     Total Temperature at LPT Outlet         [degrees R]
//   s_7  -> P30     Total Pressure at HPC Outlet            [psia]
//   s_8  -> Nf      Physical Fan Speed                      [rpm]
//   s_9  -> Nc      Physical Core Speed                     [rpm]
//   s_11 -> Ps30    Static Pressure at HPC Outlet           [psia]
//   s_12 -> phi     Fuel Flow / Ps30 Ratio                  [pps/psi]
//   s_13 -> NRf     Corrected Fan Speed                     [rpm]
//   s_14 -> NRc     Corrected Core Speed                    [rpm]
//   s_15 -> BPR     Bypass Ratio                            [dimensionless]
//   s_17 -> htBleed Bleed Enthalpy                          [dimensionless]
//   s_20 -> W31     HPT Coolant Bleed Flow                  [lbm/s]
//   s_21 -> W32     LPT Coolant Bleed Flow                  [lbm/s]
//
// The two operational setting channels (setting_1 = altitude,
// setting_2 = Mach number) are intentionally omitted because they are
// excluded from the All Sensors UI sheet.

class _PhysicalRange {
  final double min;
  final double max;

  const _PhysicalRange(this.min, this.max);
}

/// Static utility that maps normalized [-1, 1] sensor readings back to
/// their physical engineering units.
class CmapssDescaler {
  CmapssDescaler._();

  /// Per-sensor physical bounds derived from workspace/train_FD001.txt.
  /// Each value pair is (observed_minimum, observed_maximum) in the
  /// engineering unit listed in [SensorMetadata.unit].
  static const Map<String, _PhysicalRange> _ranges = <String, _PhysicalRange>{
    // --- Temperatures (degrees R) ---
    // T24: narrow range; single operating condition keeps values tightly clustered.
    's_2':  _PhysicalRange(641.21,  644.53),
    // T30: wider range due to HPC degradation progression.
    's_3':  _PhysicalRange(1571.04, 1616.91),
    // T50: reflects LPT outlet temperature under degradation.
    's_4':  _PhysicalRange(1382.25, 1441.49),

    // --- Pressures (psia) ---
    's_7':  _PhysicalRange(549.85,  556.06),
    's_11': _PhysicalRange(46.85,   48.53),

    // --- Speeds (rpm) ---
    // Nf and NRf have an extremely narrow physical range under FD001
    // conditions because the controller holds fan speed nearly constant.
    's_8':  _PhysicalRange(2387.90, 2388.56),
    's_9':  _PhysicalRange(9021.73, 9244.59),
    's_13': _PhysicalRange(2387.88, 2388.56),
    's_14': _PhysicalRange(8099.94, 8293.72),

    // --- Fuel flow ratio (pps/psi) ---
    's_12': _PhysicalRange(518.69,  523.38),

    // --- Dimensionless ---
    's_15': _PhysicalRange(8.3249,  8.5848),
    's_17': _PhysicalRange(388.00,  400.00),

    // --- Coolant flows (lbm/s) ---
    's_20': _PhysicalRange(38.14,   39.43),
    's_21': _PhysicalRange(22.8942, 23.6184),
  };

  /// Converts a normalized sensor reading in [-1, 1] to its physical value.
  ///
  /// Returns [null] when [sensorKey] has no registered physical range, which
  /// lets callers fall back to displaying the raw value unchanged.
  static double? descale(String sensorKey, double scaledValue) {
    final range = _ranges[sensorKey];
    if (range == null) return null;
    return (scaledValue + 1.0) / 2.0 * (range.max - range.min) + range.min;
  }

}
