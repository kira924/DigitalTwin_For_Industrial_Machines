import 'package:flutter/material.dart';

import '../theme/dt_colors.dart';

enum MachineStatus { running, maintenance, fault, stopped, offline }

class MachineMetricPoint {
  final String label;
  final double value;
  const MachineMetricPoint({required this.label, required this.value});

  Map<String, dynamic> toMap() => <String, dynamic>{'label': label, 'value': value};

  factory MachineMetricPoint.fromMap(Map<String, dynamic> map) {
    return MachineMetricPoint(
      label: (map['label'] ?? '').toString(),
      value: ((map['value'] ?? 0) as num).toDouble(),
    );
  }
}

class MachineSensor {
  final String label;
  final String unit;
  final double value;
  final Color color;

  const MachineSensor({
    required this.label,
    required this.unit,
    required this.value,
    required this.color,
  });

  Map<String, dynamic> toMap() => <String, dynamic>{
        'label': label,
        'unit': unit,
        'value': value,
        'color': color.toARGB32(),
      };

  factory MachineSensor.fromMap(Map<String, dynamic> map) {
    return MachineSensor(
      label: (map['label'] ?? '').toString(),
      unit: (map['unit'] ?? '').toString(),
      value: ((map['value'] ?? 0) as num).toDouble(),
      color: Color((map['color'] ?? DT.green.toARGB32()) as int),
    );
  }
}

class MachineTimelineEvent {
  final DateTime time;
  final String title;
  final String detail;
  final Color color;

  const MachineTimelineEvent({
    required this.time,
    required this.title,
    required this.detail,
    required this.color,
  });

  Map<String, dynamic> toMap() => <String, dynamic>{
        'time': time.toIso8601String(),
        'title': title,
        'detail': detail,
        'color': color.toARGB32(),
      };

  factory MachineTimelineEvent.fromMap(Map<String, dynamic> map) {
    return MachineTimelineEvent(
      time: DateTime.tryParse((map['time'] ?? '').toString()) ?? DateTime.now(),
      title: (map['title'] ?? '').toString(),
      detail: (map['detail'] ?? '').toString(),
      color: Color((map['color'] ?? DT.cyan.toARGB32()) as int),
    );
  }
}

class Machine {
  final String id;
  final String name;
  final String type;
  final String code;
  final String location;
  final String imagePath;
  /// User-uploaded photo URL (Firebase Storage). When non-null, the UI
  /// should prefer this over [imagePath] (which is the asset fallback).
  final String? imageUrl;

  /// Firebase Storage object path for the uploaded photo, e.g.
  /// `machine-images/{machineId}/{ts}.jpg`. Only the path can address
  /// the actual blob for deletion — the public URL is one-way.
  final String? imageStoragePath;

  final MachineStatus status;
  final int efficiency;
  final double temperature;
  final double speed;
  final double vibration;
  final double power;
  final double pressure;
  final int productionCount;
  final double productionRate;
  final String? activeAlert;
  final DateTime lastUpdated;
  final List<MachineMetricPoint> performanceSeries;
  final List<MachineSensor> sensors;
  final List<MachineTimelineEvent> timeline;

  const Machine({
    required this.id,
    required this.name,
    required this.type,
    required this.code,
    required this.location,
    required this.imagePath,
    this.imageUrl,
    this.imageStoragePath,
    required this.status,
    required this.efficiency,
    required this.temperature,
    required this.speed,
    required this.vibration,
    required this.power,
    required this.pressure,
    required this.productionCount,
    required this.productionRate,
    required this.activeAlert,
    required this.lastUpdated,
    required this.performanceSeries,
    required this.sensors,
    required this.timeline,
  });

  Machine copyWith({
    String? id,
    String? name,
    String? type,
    String? code,
    String? location,
    String? imagePath,
    String? imageUrl,
    bool clearImageUrl = false,
    String? imageStoragePath,
    bool clearImageStoragePath = false,
    MachineStatus? status,
    int? efficiency,
    double? temperature,
    double? speed,
    double? vibration,
    double? power,
    double? pressure,
    int? productionCount,
    double? productionRate,
    String? activeAlert,
    bool clearAlert = false,
    DateTime? lastUpdated,
    List<MachineMetricPoint>? performanceSeries,
    List<MachineSensor>? sensors,
    List<MachineTimelineEvent>? timeline,
  }) {
    return Machine(
      id: id ?? this.id,
      name: name ?? this.name,
      type: type ?? this.type,
      code: code ?? this.code,
      location: location ?? this.location,
      imagePath: imagePath ?? this.imagePath,
      imageUrl: clearImageUrl ? null : (imageUrl ?? this.imageUrl),
      imageStoragePath: clearImageStoragePath
          ? null
          : (imageStoragePath ?? this.imageStoragePath),
      status: status ?? this.status,
      efficiency: efficiency ?? this.efficiency,
      temperature: temperature ?? this.temperature,
      speed: speed ?? this.speed,
      vibration: vibration ?? this.vibration,
      power: power ?? this.power,
      pressure: pressure ?? this.pressure,
      productionCount: productionCount ?? this.productionCount,
      productionRate: productionRate ?? this.productionRate,
      activeAlert: clearAlert ? null : (activeAlert ?? this.activeAlert),
      lastUpdated: lastUpdated ?? this.lastUpdated,
      performanceSeries: performanceSeries ?? this.performanceSeries,
      sensors: sensors ?? this.sensors,
      timeline: timeline ?? this.timeline,
    );
  }

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'id': id,
      'name': name,
      'type': type,
      'code': code,
      'location': location,
      'imagePath': imagePath,
      'imageUrl': imageUrl,
      'imageStoragePath': imageStoragePath,
      'status': status.name,
      'efficiency': efficiency,
      'temperature': temperature,
      'speed': speed,
      'vibration': vibration,
      'power': power,
      'pressure': pressure,
      'productionCount': productionCount,
      'productionRate': productionRate,
      'activeAlert': activeAlert,
      'lastUpdated': lastUpdated.toIso8601String(),
      'performanceSeries': performanceSeries.map((point) => point.toMap()).toList(growable: false),
      'sensors': sensors.map((sensor) => sensor.toMap()).toList(growable: false),
      'timeline': timeline.map((event) => event.toMap()).toList(growable: false),
    };
  }

  factory Machine.fromMap(Map<String, dynamic> map, {String? fallbackId}) {
    final rawStatus = (map['status'] ?? 'offline').toString();
    return Machine(
      id: (map['id'] ?? fallbackId ?? '').toString(),
      name: (map['name'] ?? '').toString(),
      type: (map['type'] ?? '').toString(),
      code: (map['code'] ?? '').toString(),
      location: (map['location'] ?? '').toString(),
      imagePath: (map['imagePath'] ?? 'assets/images/machine_cnc.png').toString(),
      imageUrl: (map['imageUrl'] as String?)?.isNotEmpty == true
          ? map['imageUrl'] as String
          : null,
      imageStoragePath: (map['imageStoragePath'] as String?)?.isNotEmpty == true
          ? map['imageStoragePath'] as String
          : null,
      status: MachineStatus.values.firstWhere(
        (value) => value.name == rawStatus,
        orElse: () => MachineStatus.offline,
      ),
      efficiency: ((map['efficiency'] ?? 0) as num).toInt(),
      temperature: ((map['temperature'] ?? 0) as num).toDouble(),
      speed: ((map['speed'] ?? 0) as num).toDouble(),
      vibration: ((map['vibration'] ?? 0) as num).toDouble(),
      power: ((map['power'] ?? 0) as num).toDouble(),
      pressure: ((map['pressure'] ?? 0) as num).toDouble(),
      productionCount: ((map['productionCount'] ?? 0) as num).toInt(),
      productionRate: ((map['productionRate'] ?? 0) as num).toDouble(),
      activeAlert: map['activeAlert']?.toString(),
      lastUpdated: DateTime.tryParse((map['lastUpdated'] ?? '').toString()) ?? DateTime.now(),
      performanceSeries: ((map['performanceSeries'] ?? const <Map<String, dynamic>>[]) as List)
          .map((item) => MachineMetricPoint.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false),
      sensors: ((map['sensors'] ?? const <Map<String, dynamic>>[]) as List)
          .map((item) => MachineSensor.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false),
      timeline: ((map['timeline'] ?? const <Map<String, dynamic>>[]) as List)
          .map((item) => MachineTimelineEvent.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false),
    );
  }
}
