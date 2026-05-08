// ignore_for_file: avoid_print

import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/engine_telemetry_payload.dart';
import '../services/mqtt_service.dart';
import 'mqtt_providers.dart';

/// Holds the most recent decoded telemetry frame from MQTT.
class LatestTelemetry extends Notifier<EngineTelemetryPayload?> {
  StreamSubscription<EngineTelemetryPayload>? _telemetrySubscription;

  @override
  EngineTelemetryPayload? build() {
    final MqttService mqtt = ref.watch(mqttServiceProvider);
    _telemetrySubscription = mqtt.telemetryStream.listen(
      (EngineTelemetryPayload payload) {
        state = payload;
      },
      onError: (Object error, StackTrace stackTrace) {
        print(
          'LatestTelemetry stream error: $error\n$stackTrace',
        );
      },
    );

    ref.onDispose(() {
      unawaited(_telemetrySubscription?.cancel());
      _telemetrySubscription = null;
    });

    unawaited(
      mqtt.connect().catchError((Object error, StackTrace stackTrace) {
        print(
          'LatestTelemetry MQTT connect error: $error\n$stackTrace',
        );
      }),
    );
    return null;
  }

  /// Replaces the cached telemetry snapshot (manual override or tests).
  void setTelemetry(EngineTelemetryPayload? value) {
    state = value;
  }
}

/// Latest telemetry exposed to the UI layer.
final latestTelemetryProvider =
    NotifierProvider<LatestTelemetry, EngineTelemetryPayload?>(
  LatestTelemetry.new,
);

/// Maintains a rolling history buffer for each sensor signal.
class SensorHistory extends Notifier<Map<String, List<double>>> {
  static const int _windowSize = 20;

  @override
  Map<String, List<double>> build() {
    ref.listen<EngineTelemetryPayload?>(
      latestTelemetryProvider,
      (EngineTelemetryPayload? previous, EngineTelemetryPayload? next) {
        if (next == null) {
          return;
        }
        _appendTelemetry(next);
      },
    );
    return <String, List<double>>{};
  }

  void _appendTelemetry(EngineTelemetryPayload payload) {
    final Map<String, List<double>> nextState = <String, List<double>>{
      for (final MapEntry<String, List<double>> entry in state.entries)
        entry.key: List<double>.from(entry.value),
    };

    for (final MapEntry<String, double> entry
        in payload.currentSensorReadings.entries) {
      final List<double> history = nextState.putIfAbsent(
        entry.key,
        () => <double>[],
      );
      history.add(entry.value);
      _trimToRollingWindow(history);
    }

    state = nextState;
  }

  /// Enforces a strict rolling window size per sensor series.
  void _trimToRollingWindow(List<double> values) {
    if (values.length <= _windowSize) {
      return;
    }
    values.removeRange(0, values.length - _windowSize);
  }
}

/// Exposes rolling history (last 20 readings) for each sensor.
final sensorHistoryProvider =
    NotifierProvider<SensorHistory, Map<String, List<double>>>(
  SensorHistory.new,
);
