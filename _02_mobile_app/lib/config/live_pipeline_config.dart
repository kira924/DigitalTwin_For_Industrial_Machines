class LivePipelineConfig {
  const LivePipelineConfig._();

  static const String liveMachineId = 'engine_1';

  // ── Platform detector ─────────────────────────────────────────────────────
  // Resolves to true at compile time when building for the browser.
  // Uses the same underlying mechanism as Flutter's kIsWeb — no import needed.
  // Mobile/desktop builds see this as false, so their defaults are unaffected.
  static const bool _isWeb = bool.fromEnvironment('dart.library.html');

  // ── MQTT broker ──────────────────────────────────────────────────────────
  //
  // Default by platform:
  //   Web   → local Mosquitto WebSocket (ws://127.0.0.1:9001, no TLS, no auth)
  //   Mobile → HiveMQ Cloud private cluster (TLS port 8883, authenticated)
  //
  // Plan-B override for mobile (switches to local Mosquitto):
  //   flutter run \
  //     --dart-define=MQTT_BROKER=192.168.1.x \
  //     --dart-define=MQTT_PORT=1883 \
  //     --dart-define=MQTT_WS_PORT=9001 \
  //     --dart-define=MQTT_USE_TLS=false \
  //     --dart-define=MQTT_USERNAME= \
  //     --dart-define=MQTT_PASSWORD=

  static const String mqttBroker = String.fromEnvironment(
    'MQTT_BROKER',
    defaultValue: _isWeb
        ? '127.0.0.1'
        : '1c2024b173114f9d9e1577e9d4a5c467.s1.eu.hivemq.cloud',
  );

  /// TCP/TLS port — used by native mobile/desktop builds.
  static const int mqttPort = int.fromEnvironment(
    'MQTT_PORT',
    defaultValue: _isWeb ? 9001 : 8883,
  );

  /// WebSocket port — used by the browser (web) build.
  /// HiveMQ Cloud WSS = 8884 | local Mosquitto WS = 9001.
  static const int mqttWsPort = int.fromEnvironment(
    'MQTT_WS_PORT',
    defaultValue: _isWeb ? 9001 : 8884,
  );

  /// Disabled automatically on web so the local Mosquitto WebSocket is used
  /// without TLS. Enabled by default on mobile for the HiveMQ Cloud cluster.
  static const bool mqttUseTls = bool.fromEnvironment(
    'MQTT_USE_TLS',
    defaultValue: !_isWeb,
  );

  static const String mqttUsername = String.fromEnvironment(
    'MQTT_USERNAME',
    defaultValue: _isWeb ? '' : 'khaled_admin',
  );

  static const String mqttPassword = String.fromEnvironment(
    'MQTT_PASSWORD',
    defaultValue: _isWeb ? '' : 'Test1234',
  );

  // ── Topics ───────────────────────────────────────────────────────────────
  static const String mqttTopic = String.fromEnvironment(
    'MQTT_TOPIC',
    defaultValue: 'digital_twin/engine_telemetry',
  );

  // ── FastAPI backend ───────────────────────────────────────────────────────
  static const String backendUrl = String.fromEnvironment(
    'BACKEND_URL',
    defaultValue: 'http://192.168.1.4:8000/predict',
  );

  // ── Model / chart ────────────────────────────────────────────────────────
  static const int bufferSize = 30;
  static const int chartHistorySize = 24;
  static const double maxRul = 250;
  static const double warningRulThreshold = 70;
  static const double warningTempThreshold = 645;
}
