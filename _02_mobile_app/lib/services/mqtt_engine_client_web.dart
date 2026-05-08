import 'package:mqtt_client/mqtt_browser_client.dart';
import 'package:mqtt_client/mqtt_client.dart';

MqttClient buildEngineTelemetryMqttClient(String clientId) {
  final client = MqttBrowserClient('ws://127.0.0.1', clientId);
  client.port = 9001;
  client.websocketProtocols = MqttClientConstants.protocolsSingleDefault;
  client.keepAlivePeriod = 20;
  client.logging(on: false);
  return client;
}
