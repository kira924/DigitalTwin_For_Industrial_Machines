import 'package:flutter/foundation.dart';

/// Centralised API base URL configuration for the FastAPI + Cloudinary
/// upload backend.
///
/// **Where the URL goes per platform:**
///
/// - **Android emulator** — `http://10.0.2.2:8000`
///   (the special host the Android emulator routes to the host machine's
///   localhost; `127.0.0.1` from inside the emulator means the emulator
///   itself, which is wrong)
/// - **iOS simulator** — `http://127.0.0.1:8000`
/// - **Flutter Web (Chrome/Edge running on the same host)** — `http://127.0.0.1:8000`
/// - **Physical device on the same Wi-Fi as the host** — replace with
///   the host machine's LAN IP, e.g. `http://192.168.1.42:8000`
/// - **Production** — replace with the deployed FastAPI URL
///
/// To change the URL globally, edit [ApiConfig.baseUrl] (or pass
/// `--dart-define=API_BASE_URL=...` at build time and edit this file
/// to read it via `String.fromEnvironment`).
class ApiConfig {
  ApiConfig._();

  /// Base URL of the FastAPI upload backend.
  ///
  /// Picked at compile time using `--dart-define=API_BASE_URL=...` if
  /// supplied; otherwise defaults to a sensible per-platform value.
  static const String _envBaseUrl =
      String.fromEnvironment('API_BASE_URL', defaultValue: '');

  /// The effective base URL the app will hit. Reads from
  /// `--dart-define=API_BASE_URL` first; falls back to the platform
  /// default.
  static String get baseUrl {
    if (_envBaseUrl.isNotEmpty) return _envBaseUrl;
    return _platformDefault;
  }

  /// Per-platform default. Hosts are deliberately specific so a stock
  /// `flutter run` works on the typical dev setup without flags.
  static String get _platformDefault {
    if (kIsWeb) return 'http://127.0.0.1:8000';
    // dart:io platforms
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        // Android emulator → host machine. Physical Android device
        // needs your LAN IP instead — override with --dart-define.
        return 'http://10.0.2.2:8000';
      case TargetPlatform.iOS:
      case TargetPlatform.macOS:
      case TargetPlatform.windows:
      case TargetPlatform.linux:
      case TargetPlatform.fuchsia:
        return 'http://127.0.0.1:8000';
    }
  }

  // ── Endpoint paths (relative to baseUrl) ───────────────────────────────
  static const String machineImageEndpoint = '/upload/machine-image';
  static const String userAvatarEndpoint = '/upload/user-avatar';
  static const String deleteImageEndpoint = '/delete-image';

  /// Admin: create a Firebase Auth account + Firestore profile.
  /// Handled by the FastAPI backend using the Firebase Admin SDK.
  static const String createUserEndpoint = '/admin/create-user';

  /// Admin: delete a Firebase Auth account and soft-delete Firestore profile.
  static const String deleteUserEndpoint = '/admin/delete-user';

  // ── Timeouts ───────────────────────────────────────────────────────────
  /// Hard timeout for any single upload request. Mirrors the previous
  /// Firebase Storage timeout so the UI never hangs forever.
  static const Duration uploadTimeout = Duration(seconds: 30);

  /// Shorter timeout for delete requests — they're tiny payloads.
  static const Duration deleteTimeout = Duration(seconds: 15);

  /// Timeout for admin user-creation requests. Includes Auth account creation
  /// and a Firestore write, so allow a bit more headroom than a plain upload.
  static const Duration createUserTimeout = Duration(seconds: 30);

  /// Timeout for admin user-deletion requests.
  static const Duration deleteUserTimeout = Duration(seconds: 20);
}
