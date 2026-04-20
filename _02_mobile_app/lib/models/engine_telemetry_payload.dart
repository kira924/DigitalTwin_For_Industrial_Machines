/// Immutable representation of one engine telemetry MQTT message.
class EngineTelemetryPayload {
  /// Creates a telemetry payload.
  const EngineTelemetryPayload({
    required this.engineId,
    required this.currentCycle,
    required this.predictedRul,
    required this.currentSensorReadings,
    required this.aiRootCauses,
  });

  /// Engine identifier from the backend.
  final int engineId;

  /// Current operating cycle index.
  final int currentCycle;

  /// AI-predicted remaining useful life (RUL).
  final double predictedRul;

  /// Normalized sensor readings keyed by sensor name.
  final Map<String, double> currentSensorReadings;

  /// Explainability scores for root-cause signals.
  final Map<String, double> aiRootCauses;

  /// Parses a JSON object map into [EngineTelemetryPayload].
  factory EngineTelemetryPayload.fromJson(Map<String, dynamic> json) {
    return EngineTelemetryPayload(
      engineId: _asInt(json['engine_id']),
      currentCycle: _asInt(json['current_cycle']),
      predictedRul: _asDouble(json['predicted_rul']),
      currentSensorReadings: _asDoubleMap(json['current_sensor_readings']),
      aiRootCauses: _asDoubleMap(json['ai_root_causes']),
    );
  }

  /// Serializes this payload to a JSON-compatible map.
  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'engine_id': engineId,
      'current_cycle': currentCycle,
      'predicted_rul': predictedRul,
      'current_sensor_readings': currentSensorReadings,
      'ai_root_causes': aiRootCauses,
    };
  }
}

int _asInt(Object? value) {
  if (value is int) {
    return value;
  }
  if (value is double) {
    return value.round();
  }
  if (value is num) {
    return value.toInt();
  }
  throw FormatException('Expected int-compatible value, got $value');
}

double _asDouble(Object? value) {
  if (value is double) {
    return value;
  }
  if (value is int) {
    return value.toDouble();
  }
  if (value is num) {
    return value.toDouble();
  }
  throw FormatException('Expected num value, got $value');
}

Map<String, double> _asDoubleMap(Object? value) {
  if (value is! Map) {
    throw FormatException('Expected JSON object map, got $value');
  }
  return value.map<String, double>(
    (Object? key, Object? v) => MapEntry(
      key!.toString(),
      _asDouble(v),
    ),
  );
}
