class LivePipelineConfig {
  const LivePipelineConfig._();

  static const String liveMachineId = 'engine_1';

  static const String mqttBroker = String.fromEnvironment(
    'MQTT_BROKER',
    defaultValue: 'broker.hivemq.com',
  );

  static const int mqttPort = int.fromEnvironment(
    'MQTT_PORT',
    defaultValue: 1883,
  );

  static const String mqttTopic = String.fromEnvironment(
    'MQTT_TOPIC',
    defaultValue: 'ahmed/elhadyy/engine1',
  );

  static const String backendUrl = String.fromEnvironment(
    'BACKEND_URL',
    defaultValue: 'http://192.168.1.4:8000/predict',
  );

  static const int bufferSize = 30;
  static const int chartHistorySize = 24;
  static const double maxRul = 250;
  static const double warningRulThreshold = 70;
  static const double warningTempThreshold = 645;
}
