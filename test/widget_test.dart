import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';

import 'package:final_project/app/app.dart';
import 'package:final_project/app/state.dart';

void main() {
  testWidgets('app boots to login screen', (WidgetTester tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => AppState(),
        child: const DigitalTwinApp(),
      ),
    );

    expect(find.textContaining('Factory'), findsOneWidget);
  });
}
