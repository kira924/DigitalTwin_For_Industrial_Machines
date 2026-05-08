import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';

MqttClient buildEngineTelemetryMqttClient(String clientId) {
  // Return to the clean, native TCP connection on port 1883
  return MqttServerClient.withPort('192.168.1.10', clientId, 1883);
}