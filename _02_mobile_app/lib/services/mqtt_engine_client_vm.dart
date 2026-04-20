import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

/// Builds a TCP [MqttServerClient] for non-web targets (tests, analyzer, VM).
///
/// Flutter Web builds use [mqtt_engine_client_web.dart] instead.
MqttClient buildEngineTelemetryMqttClient(String clientId) {
  return MqttServerClient.withPort('127.0.0.1', clientId, 1883);
}
