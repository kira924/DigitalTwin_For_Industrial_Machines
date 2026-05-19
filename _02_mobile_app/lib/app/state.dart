import 'dart:async';
import 'dart:convert';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../data/machine_catalog.dart';
import '../data/seed_logs.dart';
import '../data/user_catalog.dart';
import '../firebase/firebase_bootstrap.dart';
import '../models/app_user.dart';
import '../models/engine_telemetry_payload.dart';
import '../models/machine.dart';
import '../models/machine_report.dart';
import '../services/firebase_sync_service.dart';
import '../services/machine_status_resolver.dart';
import '../services/storage_service.dart';
import '../theme/dt_colors.dart';

// Re-export models so callers that import state.dart keep working unchanged.
export '../models/app_alert.dart';
export '../models/history_log.dart';

import '../models/app_alert.dart';
import '../models/history_log.dart';

part 'state_firebase.dart';
part 'state_auth.dart';
part 'state_telemetry.dart';
part 'state_machines.dart';
part 'state_users.dart';
part 'state_reports.dart';
part 'state_ui.dart';

enum AppTab { machines, dashboard, alerts, history, admin }

enum DashboardRange { day, week, month, quarter, year, custom }

class AppState extends ChangeNotifier {
  // ── Auth / bootstrap flags ─────────────────────────────────────────────────
  bool _isLoggedIn = false;
  bool get isLoggedIn => _isLoggedIn;

  bool _isBootstrapping = true;
  bool get isBootstrapping => _isBootstrapping;

  bool _authBusy = false;
  bool get authBusy => _authBusy;

  bool _firebaseEnabled = false;
  bool get firebaseEnabled => _firebaseEnabled;

  bool _syncBusy = false;
  bool get syncBusy => _syncBusy;

  String _firebaseStatus = 'Initializing app state...';
  String get firebaseStatus => _firebaseStatus;

  // ── Navigation ─────────────────────────────────────────────────────────────
  AppTab _currentTab = AppTab.dashboard;
  AppTab get currentTab => _currentTab;

  String? _selectedMachineId;
  String? get selectedMachineId => _selectedMachineId;

  // ── Preferences ────────────────────────────────────────────────────────────
  ThemeMode _themeMode = ThemeMode.dark;
  ThemeMode get themeMode => _themeMode;

  bool _notificationsEnabled = true;
  bool get notificationsEnabled => _notificationsEnabled;

  bool _offlineMode = true;
  bool get offlineMode => _offlineMode;

  DashboardRange _range = DashboardRange.day;
  DashboardRange get range => _range;

  // ── Factory settings (thresholds) ─────────────────────────────────────────
  double temperatureThreshold = 80;
  double vibrationThreshold = 0.65;
  double powerThreshold = 28;

  // ── Core data collections ──────────────────────────────────────────────────
  late List<Machine> _machines;
  List<Machine> get machines => List<Machine>.unmodifiable(_machines);

  late List<AppUser> _users;
  List<AppUser> get users => List<AppUser>.unmodifiable(_users);

  late AppUser _currentUser;
  AppUser get currentUser => _currentUser;

  final List<AppAlert> _alerts = <AppAlert>[];
  List<AppAlert> get alerts => List<AppAlert>.unmodifiable(_alerts);

  final List<HistoryLog> _history = <HistoryLog>[];
  List<HistoryLog> get history => List<HistoryLog>.unmodifiable(_history);

  final List<MachineReport> _reports = <MachineReport>[];
  List<MachineReport> get reports => List<MachineReport>.unmodifiable(_reports);

  // ── Per-machine Firestore subscriptions ────────────────────────────────────
  final Map<String, StreamSubscription<List<MachineReport>>>
      _reportsSubscriptions = {};
  final Map<String, StreamSubscription<List<Map<String, dynamic>>>>
      _historySubscriptions = {};

  // ── Telemetry cache ────────────────────────────────────────────────────────
  final Map<String, EngineTelemetryPayload> _latestPayloadByMachine = {};
  final Map<String, DateTime> _lastSeenByMachine = {};

  // ── Status transition tracker ──────────────────────────────────────────────
  final Map<String, MachineStatus> _lastResolvedStatus = {};

  // ── Emergency alert deduplication ─────────────────────────────────────────
  static const double _kCriticalRulCycles = 30.0;
  final Map<String, Set<String>> _activeEmergencyReasons = {};

  // ── Timers + service refs ──────────────────────────────────────────────────
  Timer? _statusSweepTimer;
  Timer? _simulator;
  Timer? _presentationHeartbeatTimer;
  SharedPreferences? _prefs;
  FirebaseSyncService? _firebase;
  StreamSubscription<List<Machine>>? _machineSubscription;
  StreamSubscription<List<AppUser>>? _userSubscription;

  /// True once the first Firestore machine snapshot has arrived.
  /// Guards syncNow() from pushing local seed/cache data before real data loads.
  bool _firestoreLoaded = false;

  // ── Constructor ────────────────────────────────────────────────────────────
  AppState() {
    _machines = buildSeedMachines();
    _users = buildSeedUsers();
    _currentUser = _users.first;

    final seed = buildSeedLogs();
    _history.addAll(seed.history);
    _alerts.addAll(seed.alerts);
    _reports.addAll(seed.reports);

    _initialize();
    _startStatusSweep();
    _startPresentationHeartbeat();
  }

  // Keeps the first 4 machines in a "Running" state for presentation purposes
  // by refreshing their last-seen timestamp every 20 seconds — well within the
  // 30-second live-telemetry window used by MachineStatusResolver.
  // Remove or disable this method once live MQTT data is streaming.
  static const int _presentationRunningCount = 4;

  void _startPresentationHeartbeat() {
    void refresh() {
      final now = DateTime.now();
      final ids = _machines
          .take(_presentationRunningCount)
          .map((m) => m.id);
      for (final id in ids) {
        _lastSeenByMachine[id] = now;
      }
      _notify();
    }

    refresh();
    _presentationHeartbeatTimer =
        Timer.periodic(const Duration(seconds: 20), (_) => refresh());
  }

  void _startSimulator() {}

  /// Thin wrapper so extension methods in part files can trigger UI rebuilds
  /// without violating the @protected annotation on [notifyListeners].
  void _notify() => notifyListeners();

  @override
  void dispose() {
    _simulator?.cancel();
    _statusSweepTimer?.cancel();
    _presentationHeartbeatTimer?.cancel();
    _machineSubscription?.cancel();
    _userSubscription?.cancel();
    super.dispose();
  }
}

extension<T> on Iterable<T> {
  T? firstWhereOrNull(bool Function(T element) test) {
    for (final element in this) {
      if (test(element)) return element;
    }
    return null;
  }
}
