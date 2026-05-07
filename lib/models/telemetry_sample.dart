class TelemetrySample {
  final int engineId;
  final int cycle;
  final DateTime receivedAt;
  final double setting1;
  final double setting2;
  final double temp;
  final double pressure;
  final double speed;
  final double s7;
  final double s8;
  final double s9;
  final double s11;
  final double s12;
  final double s13;
  final double s14;
  final double s15;
  final double s17;
  final double s20;
  final double s21;

  const TelemetrySample({
    required this.engineId,
    required this.cycle,
    required this.receivedAt,
    required this.setting1,
    required this.setting2,
    required this.temp,
    required this.pressure,
    required this.speed,
    required this.s7,
    required this.s8,
    required this.s9,
    required this.s11,
    required this.s12,
    required this.s13,
    required this.s14,
    required this.s15,
    required this.s17,
    required this.s20,
    required this.s21,
  });

  factory TelemetrySample.fromPayload(Map<String, dynamic> json) {
    return TelemetrySample(
      engineId: _asInt(json['engine_id']),
      cycle: _asInt(json['cycle']),
      receivedAt: DateTime.now(),
      setting1: _asDouble(json['setting_1']),
      setting2: _asDouble(json['setting_2']),
      temp: _asDouble(json['s_2']),
      pressure: _asDouble(json['s_3']),
      speed: _asDouble(json['s_4']),
      s7: _asDouble(json['s_7']),
      s8: _asDouble(json['s_8']),
      s9: _asDouble(json['s_9']),
      s11: _asDouble(json['s_11']),
      s12: _asDouble(json['s_12']),
      s13: _asDouble(json['s_13']),
      s14: _asDouble(json['s_14']),
      s15: _asDouble(json['s_15']),
      s17: _asDouble(json['s_17']),
      s20: _asDouble(json['s_20']),
      s21: _asDouble(json['s_21']),
    );
  }

  List<double> toFeatureVector() {
    return <double>[
      setting1,
      setting2,
      temp,
      pressure,
      speed,
      s7,
      s8,
      s9,
      s11,
      s12,
      s13,
      s14,
      s15,
      s17,
      s20,
      s21,
    ];
  }
}

double _asDouble(dynamic value) {
  if (value is num) {
    return value.toDouble();
  }
  return double.tryParse('$value') ?? 0.0;
}

int _asInt(dynamic value) {
  if (value is num) {
    return value.toInt();
  }
  return int.tryParse('$value') ?? 0;
}
