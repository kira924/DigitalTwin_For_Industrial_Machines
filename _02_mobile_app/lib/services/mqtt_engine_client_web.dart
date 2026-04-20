import 'package:mqtt_client/mqtt_browser_client.dart';
import 'package:mqtt_client/mqtt_client.dart';

/// Builds a WebSocket [MqttBrowserClient] for Flutter Web (Mosquitto WS).
MqttClient buildEngineTelemetryMqttClient(String clientId) {
  final MqttBrowserClient client = MqttBrowserClient(
    'ws://127.0.0.1',
    clientId,
  );

  client.port = 9001;

  //  FIX: required for Mosquitto WS handshake
  client.websocketProtocols = MqttClientConstants.protocolsSingleDefault;

  // optional but recommended
  client.keepAlivePeriod = 20;
  client.logging(on: false);

  return client;
}
