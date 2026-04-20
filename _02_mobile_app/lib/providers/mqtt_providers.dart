import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/mqtt_service.dart';

/// Shared [MqttService] for the application.
final mqttServiceProvider = Provider<MqttService>((Ref ref) {
  final MqttService service = MqttService();
  ref.onDispose(() {
    unawaited(service.dispose());
  });
  return service;
});
