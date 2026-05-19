import 'package:mqtt_client/mqtt_browser_client.dart';
import 'package:mqtt_client/mqtt_client.dart';

import '../config/live_pipeline_config.dart';

MqttClient buildEngineTelemetryMqttClient(String clientId) {
  // MqttBrowserClient ignores any port embedded inside the URL string.
  // The port MUST be set via client.port — otherwise the library falls back
  // to its default of 1883, which is the raw TCP port and will be rejected
  // by a WebSocket listener with "First packet not CONNECT".
  //
  // URL carries only: scheme + host + optional path (no port).
  // Port is set separately below via client.port = mqttWsPort.
  const scheme = LivePipelineConfig.mqttUseTls ? 'wss' : 'ws';
  const path   = LivePipelineConfig.mqttUseTls ? '/mqtt' : '';
  const url    = '$scheme://${LivePipelineConfig.mqttBroker}$path';

  final client = MqttBrowserClient(url, clientId);
  client.port = LivePipelineConfig.mqttWsPort;
  client.websocketProtocols = MqttClientConstants.protocolsSingleDefault;
  client.keepAlivePeriod = 20;
  client.logging(on: false);
  return client;
}
