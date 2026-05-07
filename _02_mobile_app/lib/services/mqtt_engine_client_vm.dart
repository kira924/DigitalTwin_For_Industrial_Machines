import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

MqttClient buildEngineTelemetryMqttClient(String clientId) {
  return MqttServerClient.withPort('127.0.0.1', clientId, 1883);
}
