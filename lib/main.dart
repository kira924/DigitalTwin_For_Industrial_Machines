import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import 'app/app.dart';
import 'app/state.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Firebase initialization is handled inside AppState._initialize() via
  // FirebaseBootstrap.initialize(), which wraps the call in a try-catch and
  // sets firebaseEnabled accordingly. Calling Firebase.initializeApp() here
  // would cause a duplicate-app exception on the second call, making
  // firebaseEnabled report false even when Firebase is properly configured.

  runApp(
    ChangeNotifierProvider(
      create: (_) => AppState(),
      child: const DigitalTwinApp(),
    ),
  );
}