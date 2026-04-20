// ignore_for_file: avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:mqtt_client/mqtt_client.dart';

import 'mqtt_engine_client_vm.dart'
    if (dart.library.html) 'mqtt_engine_client_web.dart';

import '../models/engine_telemetry_payload.dart';

/// Topic carrying JSON [EngineTelemetryPayload] messages.
const String _kEngineTelemetryTopic = 'digital_twin/engine_telemetry';

/// MQTT client for engine telemetry (WebSocket on web, TCP on VM builds).
class MqttService {
  /// Creates a service instance. Call [connect] to open the session.
  MqttService();

  MqttClient? _client;
  StreamSubscription<List<MqttReceivedMessage<MqttMessage?>>>? _updatesSub;

  final StreamController<EngineTelemetryPayload> _telemetryController =
      StreamController<EngineTelemetryPayload>.broadcast();

  bool _connectInFlight = false;

  /// Parsed telemetry frames emitted for each valid JSON publish.
  Stream<EngineTelemetryPayload> get telemetryStream =>
      _telemetryController.stream;

  /// Whether the MQTT client reports a live session.
  bool get isConnected =>
      _client?.connectionStatus?.state == MqttConnectionState.connected;

  /// Connects to the broker, subscribes to the telemetry topic, and wires
  /// [client.updates] into [telemetryStream].
  Future<void> connect() async {
    if (isConnected) {
      return;
    }
    if (_connectInFlight) {
      return;
    }
    _connectInFlight = true;
    try {
      await disconnect();

      final String clientId =
          'digital_twin_web_${DateTime.now().microsecondsSinceEpoch}_'
          '${Random().nextInt(0x7fffffff)}';
      final MqttClient client = buildEngineTelemetryMqttClient(clientId);

      client.logging(on: false);
      client.setProtocolV311();
      client.keepAlivePeriod = 30;

      client.onConnected = _onConnected;
      client.onDisconnected = _onDisconnected;
      client.onSubscribed = _onSubscribed;

      _client = client;

      try {
        await client.connect();
      } on Object catch (error, stackTrace) {
        print('MQTT connect exception: $error\n$stackTrace');
        await disconnect();
        return;
      }

      if (client.connectionStatus?.state != MqttConnectionState.connected) {
        print(
          'MQTT connect failed, status=${client.connectionStatus}',
        );
        await disconnect();
        return;
      }

      client.subscribe(_kEngineTelemetryTopic, MqttQos.atMostOnce);

      final Stream<List<MqttReceivedMessage<MqttMessage?>>>? updateStream =
          client.updates;
      if (updateStream == null) {
        print('MQTT updates stream is null after connect');
        await disconnect();
        return;
      }

      await _updatesSub?.cancel();
      _updatesSub = updateStream.listen(
        _handleBrokerUpdates,
        onError: (Object error, StackTrace stackTrace) {
          print('MQTT updates stream error: $error\n$stackTrace');
        },
      );
    } finally {
      _connectInFlight = false;
    }
  }

  /// Tears down the subscription and closes the broker connection.
  Future<void> disconnect() async {
    await _updatesSub?.cancel();
    _updatesSub = null;
    try {
      _client?.disconnect();
    } on Object catch (error, stackTrace) {
      print('MQTT disconnect exception: $error\n$stackTrace');
    }
    _client = null;
  }

  /// Releases stream resources. Invoke when the service is no longer needed.
  Future<void> dispose() async {
    await disconnect();
    if (!_telemetryController.isClosed) {
      await _telemetryController.close();
    }
  }

  void _onConnected() {
    print('onConnected: MQTT session established');
  }

  void _onDisconnected() {
    print('onDisconnected: MQTT session ended');
  }

  void _onSubscribed(String topic) {
    print('onSubscribed: $topic');
  }

  void _handleBrokerUpdates(List<MqttReceivedMessage<MqttMessage?>>? batch) {
    if (batch == null || batch.isEmpty) {
      return;
    }
    for (final MqttReceivedMessage<MqttMessage?> message in batch) {
      final MqttMessage? payload = message.payload;
      if (payload is! MqttPublishMessage) {
        continue;
      }
      final String body = MqttPublishPayload.bytesToStringAsString(
        payload.payload.message,
      );
      try {
        final Object? decoded = jsonDecode(body);
        if (decoded is! Map) {
          print('MQTT payload is not a JSON object, topic=${message.topic}');
          continue;
        }
        final EngineTelemetryPayload model = EngineTelemetryPayload.fromJson(
          Map<String, dynamic>.from(decoded),
        );
        if (!_telemetryController.isClosed) {
          _telemetryController.add(model);
        }
      } on Object catch (error, stackTrace) {
        print(
          'MQTT JSON parse error topic=${message.topic} error=$error\n'
          '$stackTrace',
        );
      }
    }
  }
}
