import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:mqtt_client/mqtt_client.dart';

import 'mqtt_engine_client_vm.dart' if (dart.library.html) 'mqtt_engine_client_web.dart';
import '../models/engine_telemetry_payload.dart';

const String _kEngineTelemetryTopic = 'digital_twin/engine_telemetry';

class MqttService {
  MqttClient? _client;
  StreamSubscription<List<MqttReceivedMessage<MqttMessage?>>>? _updatesSub;
  final StreamController<EngineTelemetryPayload> _telemetryController =
      StreamController<EngineTelemetryPayload>.broadcast();
  bool _connectInFlight = false;

  Stream<EngineTelemetryPayload> get telemetryStream => _telemetryController.stream;
  bool get isConnected => _client?.connectionStatus?.state == MqttConnectionState.connected;

  Future<void> connect() async {
    if (isConnected || _connectInFlight) return;
    _connectInFlight = true;
    try {
      await disconnect();
      final clientId = 'digital_twin_${DateTime.now().microsecondsSinceEpoch}_${Random().nextInt(0x7fffffff)}';
      final client = buildEngineTelemetryMqttClient(clientId);
      // Enable internal MQTT logging
      client.logging(on: true);
      client.setProtocolV311();
      client.keepAlivePeriod = 30;
      _client = client;

      try {
        await client.connect();
      } catch (e) {
        // Catch the actual error and print it to the console
        print('MQTT Connection Fatal Error: $e');
        await disconnect();
        return;
      }
      if (client.connectionStatus?.state != MqttConnectionState.connected) {
        await disconnect();
        return;
      }
      client.subscribe(_kEngineTelemetryTopic, MqttQos.atMostOnce);
      final updateStream = client.updates;
      if (updateStream == null) {
        await disconnect();
        return;
      }
      await _updatesSub?.cancel();
      _updatesSub = updateStream.listen(_handleBrokerUpdates);
    } finally {
      _connectInFlight = false;
    }
  }

  Future<void> disconnect() async {
    await _updatesSub?.cancel();
    _updatesSub = null;
    try {
      _client?.disconnect();
    } catch (_) {}
    _client = null;
  }

  Future<void> dispose() async {
    await disconnect();
    if (!_telemetryController.isClosed) {
      await _telemetryController.close();
    }
  }

  void _handleBrokerUpdates(List<MqttReceivedMessage<MqttMessage?>>? batch) {
    if (batch == null || batch.isEmpty) return;
    for (final message in batch) {
      final payload = message.payload;
      if (payload is! MqttPublishMessage) continue;
      final body = MqttPublishPayload.bytesToStringAsString(payload.payload.message);
      try {
        final decoded = jsonDecode(body);
        if (decoded is! Map) continue;
        final model = EngineTelemetryPayload.fromJson(Map<String, dynamic>.from(decoded));
        if (!_telemetryController.isClosed) {
          _telemetryController.add(model);
        }
      } catch (_) {}
    }
  }
}
