import 'package:firebase_core/firebase_core.dart';
import '../firebase_options.dart';

class FirebaseBootstrapResult {
  final bool configured;
  final String message;

  const FirebaseBootstrapResult({
    required this.configured,
    required this.message,
  });
}

class FirebaseBootstrap {
  static Future<FirebaseBootstrapResult> initialize() async {
    try {
      await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
      );

      return const FirebaseBootstrapResult(
        configured: true,
        message: 'Firebase connected',
      );
    } catch (e) {
      return FirebaseBootstrapResult(
        configured: false,
        message: 'Firebase initialization failed: $e',
      );
    }
  }
}