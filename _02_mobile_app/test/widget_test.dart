import 'dart:async';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:_02_mobile_app/main.dart';
import 'package:_02_mobile_app/providers/mqtt_providers.dart';
import 'package:_02_mobile_app/services/mqtt_service.dart';

/// Avoids real broker I/O during widget tests.
final class _FakeMqttService extends MqttService {
  @override
  Future<void> connect() async {}

  @override
  Future<void> disconnect() async {}
}

void main() {
  testWidgets('App boots with dashboard shell', (WidgetTester tester) async {
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          mqttServiceProvider.overrideWith((Ref ref) {
            final _FakeMqttService service = _FakeMqttService();
            ref.onDispose(() {
              unawaited(service.dispose());
            });
            return service;
          }),
        ],
        child: const MyApp(),
      ),
    );

    expect(find.text('Industrial Dashboard'), findsOneWidget);
    expect(find.text('Dashboard layout will be populated here.'), findsOneWidget);
  });
}
