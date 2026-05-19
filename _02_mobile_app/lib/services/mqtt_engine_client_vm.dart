import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

import '../config/live_pipeline_config.dart';

MqttClient buildEngineTelemetryMqttClient(String clientId) {
  final client = MqttServerClient.withPort(
    LivePipelineConfig.mqttBroker,
    clientId,
    LivePipelineConfig.mqttPort,
  );

  // Enable TLS for the HiveMQ Cloud cluster. Disabled automatically when
  // the Plan-B fallback (local Mosquitto) is selected via --dart-define.
  client.secure = LivePipelineConfig.mqttUseTls;

  return client;
}
