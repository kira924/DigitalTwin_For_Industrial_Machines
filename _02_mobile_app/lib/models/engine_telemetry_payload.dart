/// Immutable representation of one telemetry MQTT message.
///
/// **Phase 3.3 schema notes.** This payload supports two shapes:
///
/// 1. **Legacy NASA C-MAPSS shape** (the original publisher
///    `publisher_engine1.py` style): keys `engine_id`,
///    `current_cycle`, `predicted_rul`, `current_sensor_readings`,
///    `ai_root_causes`.
///
/// 2. **Phase 3.3 friendly shape** (the new multi-machine
///    publisher): keys `machineId` (or aliases), `temperature`,
///    `vibration`, `power`, `speed`, `status`, `timestamp`. The
///    NASA-style fields are still emitted as `extra` so the SHAP
///    bars and cycle counter keep working; they fall back to
///    sensible defaults if absent.
///
/// `machineId` aliases accepted on the wire:
///   `machineId` | `machine_id` | `machineID` | `engineId` |
///   `engine_id` | `id`
///
/// When only the legacy `engine_id` int is provided, [machineId] is
/// `null` and `AppState.ingestTelemetry` resolves it via
/// `machineIdForEngine(engineId)` (1-based index into the fleet).
class EngineTelemetryPayload {
  const EngineTelemetryPayload({
    required this.engineId,
    required this.currentCycle,
    required this.predictedRul,
    required this.currentSensorReadings,
    required this.aiRootCauses,
    this.machineId,
    this.temperature,
    this.vibration,
    this.power,
    this.speed,
    this.statusHint,
    this.timestamp,
    this.healthScore,
  });

  /// Machine identifier as a string. Phase 3.3 publishers should
  /// always include this. `null` when only [engineId] is on the wire.
  final String? machineId;

  /// Legacy 1-based engine identifier from the NASA dataset. Always
  /// derivable; defaults to `0` when absent.
  final int engineId;

  final int currentCycle;
  final double predictedRul;
  final Map<String, double> currentSensorReadings;
  final Map<String, double> aiRootCauses;

  /// Optional friendly fields from the Phase 3.3 schema. `null` when
  /// the payload didn't include them.
  final double? temperature;
  final double? vibration;
  final double? power;
  final double? speed;
  final double? healthScore;
  final String? statusHint;
  final DateTime? timestamp;

  factory EngineTelemetryPayload.fromJson(Map<String, dynamic> json) {
    // Resolve machineId across aliases. Order matters: prefer the
    // explicit `machineId` if present, then fall through.
    String? machineId;
    for (final key in const [
      'machineId',
      'machine_id',
      'machineID',
      'id',
    ]) {
      final v = json[key];
      if (v != null && v.toString().isNotEmpty) {
        machineId = v.toString();
        break;
      }
    }

    // engineId — accepted as engineId / engine_id / numeric id.
    int engineId = 0;
    for (final key in const ['engine_id', 'engineId']) {
      final v = json[key];
      if (v != null) {
        try {
          engineId = _asInt(v);
          break;
        } catch (_) {
          // ignore, try next alias
        }
      }
    }
    // If we got nothing for engineId but `id` looks numeric, use it.
    if (engineId == 0 && machineId != null) {
      final maybeInt = int.tryParse(machineId);
      if (maybeInt != null) engineId = maybeInt;
    }

    final currentCycle = _asIntOrZero(json['current_cycle']);
    final predictedRul = _asDoubleOrNan(json['predicted_rul'])
        // Tolerate `rul` as Phase 3.3 friendly alias.
        .let((v) => v.isNaN ? _asDoubleOrNan(json['rul']) : v);
    final sensors = json['current_sensor_readings'];
    final causes = json['ai_root_causes'];

    return EngineTelemetryPayload(
      machineId: machineId,
      engineId: engineId,
      currentCycle: currentCycle,
      predictedRul: predictedRul.isNaN ? 0.0 : predictedRul,
      currentSensorReadings: sensors is Map
          ? _asDoubleMap(sensors)
          : const <String, double>{},
      aiRootCauses: causes is Map
          ? _asDoubleMap(causes)
          : const <String, double>{},
      temperature: _maybeDouble(json['temperature']),
      vibration: _maybeDouble(json['vibration']),
      power: _maybeDouble(json['power']),
      speed: _maybeDouble(json['speed']),
      healthScore: _maybeDouble(json['healthScore'])
          ?? _maybeDouble(json['health_score']),
      statusHint: json['status']?.toString(),
      timestamp: _maybeTimestamp(json['timestamp']),
    );
  }

  Map<String, dynamic> toJson() => <String, dynamic>{
        if (machineId != null) 'machineId': machineId,
        'engine_id': engineId,
        'current_cycle': currentCycle,
        'predicted_rul': predictedRul,
        'current_sensor_readings': currentSensorReadings,
        'ai_root_causes': aiRootCauses,
        if (temperature != null) 'temperature': temperature,
        if (vibration != null) 'vibration': vibration,
        if (power != null) 'power': power,
        if (speed != null) 'speed': speed,
        if (healthScore != null) 'healthScore': healthScore,
        if (statusHint != null) 'status': statusHint,
        if (timestamp != null) 'timestamp': timestamp!.toIso8601String(),
      };
}

// ── Parsing helpers ─────────────────────────────────────────────────────

int _asInt(Object? value) {
  if (value is int) return value;
  if (value is num) return value.toInt();
  if (value is String) {
    final n = int.tryParse(value);
    if (n != null) return n;
  }
  throw FormatException('Expected int-compatible value, got $value');
}

int _asIntOrZero(Object? value) {
  try {
    return value == null ? 0 : _asInt(value);
  } catch (_) {
    return 0;
  }
}

double _asDoubleOrNan(Object? value) {
  if (value == null) return double.nan;
  if (value is double) return value;
  if (value is num) return value.toDouble();
  if (value is String) {
    final n = double.tryParse(value);
    return n ?? double.nan;
  }
  return double.nan;
}

double? _maybeDouble(Object? value) {
  final d = _asDoubleOrNan(value);
  return d.isNaN ? null : d;
}

DateTime? _maybeTimestamp(Object? value) {
  if (value == null) return null;
  if (value is DateTime) return value;
  if (value is num) {
    // assume millis since epoch
    return DateTime.fromMillisecondsSinceEpoch(value.toInt());
  }
  if (value is String && value.isNotEmpty) {
    return DateTime.tryParse(value);
  }
  return null;
}

Map<String, double> _asDoubleMap(Object? value) {
  if (value is! Map) return const <String, double>{};
  final out = <String, double>{};
  value.forEach((k, v) {
    final d = _asDoubleOrNan(v);
    if (!d.isNaN) out[k.toString()] = d;
  });
  return out;
}

extension _Let<T> on T {
  R let<R>(R Function(T) f) => f(this);
}
