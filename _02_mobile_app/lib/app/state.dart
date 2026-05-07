import 'dart:async';
import 'dart:convert';
import 'dart:math';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../data/machine_catalog.dart';
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

enum AppTab { machines, dashboard, alerts, history, admin }
enum DashboardRange { day, week, month, quarter, year, custom }

class AppAlert {
  final String id;
  final String machineId;
  final String title;
  final String message;
  final DateTime time;
  final Color color;

  const AppAlert({
    required this.id,
    required this.machineId,
    required this.title,
    required this.message,
    required this.time,
    required this.color,
  });
}

/// Categorical kind of a history event. Drives icon/color choice in
/// the UI when callers don't override.
enum HistoryKind {
  fault,
  maintenance,
  alert,
  report,
  status,
  config,
  user,
}

class HistoryLog {
  final String id;
  final DateTime time;
  final String title;
  final String subtitle;
  final IconData icon;
  final Color color;

  /// The machine this event belongs to, if any. Global events
  /// (user added, settings changed) leave this null.
  final String? machineId;

  /// Categorical kind for grouping/filtering in per-machine views.
  final HistoryKind kind;

  /// Severity label shown on the timeline ("INFO", "WARNING", etc.).
  /// Optional so existing callers don't have to specify it.
  final String? severity;

  /// Author display name, e.g. "Belal Saqer". Null for system events.
  final String? author;

  const HistoryLog({
    required this.id,
    required this.time,
    required this.title,
    required this.subtitle,
    required this.icon,
    required this.color,
    this.machineId,
    this.kind = HistoryKind.status,
    this.severity,
    this.author,
  });
}

class AppState extends ChangeNotifier {
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

  AppTab _currentTab = AppTab.dashboard;
  AppTab get currentTab => _currentTab;

  ThemeMode _themeMode = ThemeMode.dark;
  ThemeMode get themeMode => _themeMode;

  /// Always returns false — language toggle was removed in the
  /// English-only refactor. Kept as a getter so legacy call-sites
  /// keep compiling; remove during the next cleanup pass.
  bool get isArabic => false;

  bool _notificationsEnabled = true;
  bool get notificationsEnabled => _notificationsEnabled;

  bool _offlineMode = true;
  bool get offlineMode => _offlineMode;

  DashboardRange _range = DashboardRange.day;
  DashboardRange get range => _range;

  String? _selectedMachineId;
  String? get selectedMachineId => _selectedMachineId;

  late List<Machine> _machines;
  List<Machine> get machines => List<Machine>.unmodifiable(_machines);

  late List<AppUser> _users;
  List<AppUser> get users => List<AppUser>.unmodifiable(_users);

  late AppUser _currentUser;
  AppUser get currentUser => _currentUser;

  double temperatureThreshold = 80;
  double vibrationThreshold = 0.65;
  double powerThreshold = 28;

  final List<AppAlert> _alerts = <AppAlert>[];
  List<AppAlert> get alerts => List<AppAlert>.unmodifiable(_alerts);

  final List<HistoryLog> _history = <HistoryLog>[];
  List<HistoryLog> get history => List<HistoryLog>.unmodifiable(_history);

  // ── Reports ──────────────────────────────────────────────────────────────
  /// Engineer / admin reports filed against machines. Surface in the
  /// machine detail page and the per-machine history timeline.
  final List<MachineReport> _reports = <MachineReport>[];
  List<MachineReport> get reports => List<MachineReport>.unmodifiable(_reports);

  // Firestore-backed per-machine subscriptions. The detail screen calls
  // [subscribeToMachine(id)] in initState and [unsubscribeFromMachine(id)]
  // in dispose — between those two calls, server changes flow into
  // _reports / _history live. Outside that window we still see the
  // local cache so the screen has something to render immediately.
  final Map<String, StreamSubscription<List<MachineReport>>>
      _reportsSubscriptions = {};
  final Map<String, StreamSubscription<List<Map<String, dynamic>>>>
      _historySubscriptions = {};

  /// Subscribe to a machine's reports + history subcollections in
  /// Firestore. Idempotent — calling twice for the same machineId is
  /// safe; the second call is a no-op.
  void subscribeToMachine(String machineId) {
    if (!_firebaseEnabled || _firebase == null) return;
    if (_reportsSubscriptions.containsKey(machineId)) return;

    _reportsSubscriptions[machineId] = _firebase!
        .watchReports(machineId)
        .listen((remoteReports) {
      // Merge: drop any local reports for this machine and re-insert the
      // server set. Other machines' reports are untouched.
      _reports.removeWhere((r) => r.machineId == machineId);
      _reports.addAll(remoteReports);
      _reports.sort((a, b) => b.createdAt.compareTo(a.createdAt));
      notifyListeners();
    }, onError: (_) {
      // Permission denied or offline — keep local cache. Failure is
      // non-fatal because the UI gracefully falls back to local data.
    });

    _historySubscriptions[machineId] =
        _firebase!.watchHistory(machineId).listen((events) {
      // History from Firestore is stored as raw maps (so non-Dart
      // writers can append). Merge by replacing this machine's local
      // server-sourced entries.
      _history.removeWhere(
        (e) => e.machineId == machineId && e.id.startsWith('fs-'),
      );
      for (final raw in events) {
        _history.add(_historyFromFirestore(machineId, raw));
      }
      _history.sort((a, b) => b.time.compareTo(a.time));
      notifyListeners();
    }, onError: (_) {/* non-fatal */});
  }

  /// Cancel the per-machine subscriptions started by [subscribeToMachine].
  void unsubscribeFromMachine(String machineId) {
    _reportsSubscriptions.remove(machineId)?.cancel();
    _historySubscriptions.remove(machineId)?.cancel();
  }

  /// Convert a Firestore history document map into a [HistoryLog].
  HistoryLog _historyFromFirestore(String machineId, Map<String, dynamic> raw) {
    final type = (raw['type'] ?? '').toString();
    final kind = HistoryKind.values.firstWhere(
      (k) => k.name == type,
      orElse: () => HistoryKind.status,
    );
    final severity = raw['severity']?.toString();
    final author = raw['createdByName']?.toString();

    // Map kind to a sensible icon + color so the timeline renders
    // consistently with locally-pushed events.
    IconData icon;
    Color color;
    switch (kind) {
      case HistoryKind.fault:
        icon = Icons.error_rounded;
        color = DT.red;
        break;
      case HistoryKind.maintenance:
        icon = Icons.build_circle_rounded;
        color = DT.yellow;
        break;
      case HistoryKind.alert:
        icon = Icons.crisis_alert_rounded;
        color = DT.red;
        break;
      case HistoryKind.report:
        icon = Icons.fact_check_rounded;
        color = DT.cyan;
        break;
      case HistoryKind.config:
        icon = Icons.tune_rounded;
        color = DT.blue;
        break;
      case HistoryKind.user:
        icon = Icons.person_rounded;
        color = DT.cyan;
        break;
      case HistoryKind.status:
        icon = Icons.info_outline_rounded;
        color = DT.blue;
        break;
    }

    DateTime time = DateTime.now();
    final ts = raw['createdAt'];
    if (ts is DateTime) {
      time = ts;
    } else if (ts is String && ts.isNotEmpty) {
      time = DateTime.tryParse(ts) ?? time;
    } else if (ts != null) {
      // Firestore Timestamp.toDate() handled by the fromMap on
      // FirebaseSyncService side — but raw might be a Timestamp here.
      try {
        // ignore: avoid_dynamic_calls
        time = (ts as dynamic).toDate() as DateTime;
      } catch (_) {/* keep `now` */}
    }

    return HistoryLog(
      id: 'fs-${raw['id'] ?? DateTime.now().millisecondsSinceEpoch}',
      time: time,
      title: (raw['title'] ?? '').toString(),
      subtitle: (raw['description'] ?? '').toString(),
      icon: icon,
      color: color,
      machineId: machineId,
      kind: kind,
      severity: severity,
      author: author,
    );
  }

  // ── Telemetry cache (populated by the dashboard's MQTT listener) ─────────
  //
  // Keyed by machineId, NOT by engineId. The MQTT payload arrives with an
  // integer engineId — at ingest time we translate it to the corresponding
  // machineId by indexing into the current machines list (1-based, matching
  // publisher_engine1.py's convention). This way the cache survives fleet
  // mutations: if a machine is added/removed, other machines' telemetry
  // doesn't get reassigned by accident.
  //
  // Per the brief: "Do NOT use one global latest payload for all machines".
  // Each machine's freshness is tracked independently.
  final Map<String, EngineTelemetryPayload> _latestPayloadByMachine = {};
  final Map<String, DateTime> _lastSeenByMachine = {};

  // ── Automatic machine status (Phase 3.5) ────────────────────────────────
  // The previous Start/Stop buttons mutated `machine.status` directly.
  // Phase 3.5 derives status from telemetry freshness + report
  // existence (see MachineStatusResolver). We track the last resolved
  // status per machine so we only push a history entry when it
  // actually changes — avoids spam on every refresh tick.
  final Map<String, MachineStatus> _lastResolvedStatus = {};

  /// Resolved status for [machineId] using the automatic rule:
  ///   - live telemetry          → running
  ///   - no telemetry, no report → fault
  ///   - no telemetry, has report → maintenance
  MachineStatus resolvedStatusFor(String machineId) {
    return MachineStatusResolver.resolve(
      lastSeen: _lastSeenByMachine[machineId],
      hasReport: _reports.any((r) => r.machineId == machineId),
    );
  }

  /// Detects status changes for all machines and pushes a single
  /// history entry per actual transition. Called from
  /// [ingestTelemetry] (when new data arrives) and from a periodic
  /// timer (so a machine going stale → fault is also recorded even
  /// without new payloads).
  void _recordStatusTransitionsIfAny() {
    bool anyChange = false;
    for (final m in _machines) {
      final next = resolvedStatusFor(m.id);
      final prev = _lastResolvedStatus[m.id];
      if (prev == next) continue;

      _lastResolvedStatus[m.id] = next;
      anyChange = true;

      // Skip the very first observation (prev == null): we don't want
      // a "Status changed to X" entry on app startup for every machine.
      if (prev == null) continue;

      String title;
      IconData icon;
      Color color;
      switch (next) {
        case MachineStatus.running:
          title = 'Status changed to Running';
          icon = Icons.play_arrow_rounded;
          color = DT.green;
          break;
        case MachineStatus.fault:
          title = 'Status changed to Fault';
          icon = Icons.error_rounded;
          color = DT.red;
          break;
        case MachineStatus.maintenance:
          title = 'Status changed to Under Maintenance';
          icon = Icons.build_circle_rounded;
          color = DT.yellow;
          break;
        default:
          // Stopped / offline aren't produced by the resolver.
          continue;
      }

      _pushHistory(
        title,
        m.name,
        icon,
        color,
        machineId: m.id,
        kind: HistoryKind.status,
      );
    }
    if (anyChange) {
      _persistCachedState();
      // notifyListeners() is called by the caller (ingestTelemetry or
      // the periodic sweep) so we don't double-notify.
    }
  }

  /// Periodic sweep — checks every machine for status transitions,
  /// regardless of whether a new payload arrived. Without this, a
  /// machine going from running → fault (because telemetry stopped)
  /// would never produce a history entry.
  Timer? _statusSweepTimer;
  void _startStatusSweep() {
    _statusSweepTimer?.cancel();
    _statusSweepTimer = Timer.periodic(const Duration(seconds: 15), (_) {
      _recordStatusTransitionsIfAny();
      // Also notify so cards re-render their resolved badge as time
      // passes (a machine silently going stale should update without
      // requiring a payload).
      notifyListeners();
    });
  }

  /// Permissions for the currently signed-in user.
  Permissions get permissions => Permissions.of(_currentUser);
  bool get isAdmin => _currentUser.role == AppRole.admin;
  bool get isEngineer => _currentUser.role == AppRole.engineer;
  bool get isViewer => _currentUser.role == AppRole.viewer;

  /// Defense-in-depth gate for write actions. Returns `true` if the
  /// action should be BLOCKED. UI screens already gate every action
  /// button on `permissions.canX`, but write methods call this too so
  /// any bypass (a stale callback, a debug button, a unit test) still
  /// fails closed for viewers and inactive accounts.
  bool _blockWrite(bool capability, String action) {
    if (!capability) {
      debugPrint(
        'AppState: blocked "$action" — '
        '${_currentUser.email} (${roleLabel(_currentUser.role)}, '
        'active=${_currentUser.isActive}) lacks permission.',
      );
      return true;
    }
    return false;
  }

  /// Resolve a machine's engine ID — the index in the machines list,
  /// 1-based. The MQTT publisher uses this convention. Kept for any
  /// caller that still needs the integer (e.g. the publisher topic).
  int engineIdForMachine(String machineId) {
    final idx = _machines.indexWhere((m) => m.id == machineId);
    return idx >= 0 ? idx + 1 : 0;
  }

  /// Resolve which machine an engineId belongs to. Returns null if the
  /// engineId is out of range.
  String? machineIdForEngine(int engineId) {
    final idx = engineId - 1;
    if (idx < 0 || idx >= _machines.length) return null;
    return _machines[idx].id;
  }

  /// Latest MQTT payload for a specific machine, or null if none received.
  /// Reads from the per-machine cache — never returns another machine's
  /// data as a stand-in.
  EngineTelemetryPayload? latestTelemetryFor(String machineId) {
    return _latestPayloadByMachine[machineId];
  }

  /// Wall-clock timestamp of the most recent telemetry sample for a
  /// specific machine.
  DateTime? lastSeenFor(String machineId) {
    return _lastSeenByMachine[machineId];
  }

  // ── Phase 3.3 emergency thresholds ──────────────────────────────────────
  // Critical RUL ceiling — RUL <= this value triggers an emergency alert.
  // Tuned for the NASA C-MAPSS dataset where RUL ≈ remaining cycles.
  static const double _kCriticalRulCycles = 30.0;

  /// Track which machines we've already alerted on for a given
  /// emergency reason so we don't spam the alerts feed on every
  /// payload (1 Hz publisher × 4+ machines × 4 thresholds would
  /// otherwise generate hundreds of duplicate alerts per minute).
  /// Cleared when the condition resolves.
  final Map<String, Set<String>> _activeEmergencyReasons = {};

  /// Called by the dashboard when a new MQTT payload arrives.
  ///
  /// Resolution order for the target machine:
  ///   1. `payload.machineId` (Phase 3.3 friendly schema)
  ///   2. `machineIdForEngine(payload.engineId)` (legacy NASA schema —
  ///      1-based index into the fleet, matching publisher_engine1.py)
  ///
  /// After caching, evaluates emergency conditions against the
  /// payload + machine's current threshold settings. New critical
  /// conditions create a top-level alert; resolved conditions are
  /// cleared from the dedup tracker.
  void ingestTelemetry(EngineTelemetryPayload payload) {
    // Prefer explicit machineId, fall back to engineId translation.
    String? machineId = payload.machineId;
    if (machineId != null) {
      // Sanity-check it matches a known machine. If not, drop.
      final exists = _machines.any((m) => m.id == machineId);
      if (!exists) return;
    } else {
      machineId = machineIdForEngine(payload.engineId);
      if (machineId == null) {
        // Payload references an engineId that doesn't map to any
        // current machine. Drop it rather than silently aliasing it
        // onto the wrong row.
        return;
      }
    }

    _latestPayloadByMachine[machineId] = payload;
    _lastSeenByMachine[machineId] = DateTime.now();

    // Evaluate emergency conditions against this payload.
    _evaluateEmergencyForPayload(machineId, payload);

    // The arrival of a payload can also flip the automatic status:
    // e.g. a machine that was fault (no telemetry, no report) becomes
    // running on the first payload. Record the transition once.
    _recordStatusTransitionsIfAny();

    // Note: intentionally not calling notifyListeners() here for
    // pure cache writes — the dashboard already setStates on every
    // payload. We DO notify when a new alert is produced, since the
    // Alerts tab needs to refresh.
  }

  /// Evaluate emergency rules and emit alerts for newly-triggered
  /// conditions. Idempotent — repeated evaluation of an already-active
  /// emergency is a no-op so the alerts feed doesn't get spammed.
  void _evaluateEmergencyForPayload(
    String machineId,
    EngineTelemetryPayload payload,
  ) {
    final machine = _machines.firstWhereOrNull((m) => m.id == machineId);
    if (machine == null) return;

    // Resolve sensor values: prefer the friendly Phase 3.3 fields,
    // fall back to the NASA `s_*` mapping (s_2 ≈ temp, s_4 ≈ vib,
    // s_11 ≈ power) used elsewhere in the app.
    final temp = payload.temperature
        ?? payload.currentSensorReadings['s_2']
        ?? payload.currentSensorReadings['s2'];
    final vib = payload.vibration
        ?? payload.currentSensorReadings['s_4']
        ?? payload.currentSensorReadings['s4'];
    final power = payload.power
        ?? payload.currentSensorReadings['s_11']
        ?? payload.currentSensorReadings['s11'];
    final rul = payload.predictedRul;
    final statusHint = payload.statusHint?.toLowerCase() ?? '';

    final reasons = _activeEmergencyReasons.putIfAbsent(machineId, () => <String>{});

    void fire(String code, String title, String message) {
      if (reasons.contains(code)) return; // already alerted
      reasons.add(code);
      _alerts.insert(
        0,
        AppAlert(
          id: 'alr-${DateTime.now().microsecondsSinceEpoch}',
          machineId: machineId,
          title: title,
          message: '${machine.name}: $message',
          time: DateTime.now(),
          color: DT.red,
        ),
      );
      _pushHistory(
        title,
        message,
        Icons.crisis_alert_rounded,
        DT.red,
        machineId: machineId,
        kind: HistoryKind.alert,
        severity: 'critical',
      );
      notifyListeners();
    }

    void clear(String code) => reasons.remove(code);

    // Temperature
    if (temp != null && temp >= temperatureThreshold) {
      fire('temp', 'Critical temperature',
           '${temp.toStringAsFixed(1)} ≥ ${temperatureThreshold.toStringAsFixed(0)}');
    } else if (temp != null) {
      clear('temp');
    }

    // Vibration
    if (vib != null && vib >= vibrationThreshold) {
      fire('vibration', 'Dangerous vibration',
           '${vib.toStringAsFixed(2)} ≥ ${vibrationThreshold.toStringAsFixed(2)} mm/s');
    } else if (vib != null) {
      clear('vibration');
    }

    // Power
    if (power != null && power >= powerThreshold) {
      fire('power', 'Power overload',
           '${power.toStringAsFixed(1)} ≥ ${powerThreshold.toStringAsFixed(0)} kW');
    } else if (power != null) {
      clear('power');
    }

    // Status hint from payload
    if (statusHint == 'fault' || statusHint == 'broken' ||
        statusHint == 'down' || statusHint == 'emergency') {
      fire('status', 'Machine fault reported',
           'Telemetry reports status="$statusHint".');
    } else if (statusHint.isNotEmpty) {
      clear('status');
    }

    // RUL — only fire when we have a sensible value (>0)
    if (rul > 0 && rul <= _kCriticalRulCycles) {
      fire('rul', 'Critical RUL',
           'Remaining useful life: ${rul.toStringAsFixed(0)} cycles.');
    } else if (rul > _kCriticalRulCycles) {
      clear('rul');
    }
  }

  // ── History queries ──────────────────────────────────────────────────────
  /// All events relevant to a single machine: status changes, faults,
  /// maintenance, alerts, and reports. Sorted newest-first.
  List<HistoryLog> historyForMachine(String machineId) {
    return _history.where((e) => e.machineId == machineId).toList();
  }

  /// Reports filed against a single machine, newest-first.
  List<MachineReport> reportsForMachine(String machineId) {
    final list =
        _reports.where((r) => r.machineId == machineId).toList(growable: false);
    return list..sort((a, b) => b.createdAt.compareTo(a.createdAt));
  }

  Timer? _simulator;
  final Random _random = Random(7);
  SharedPreferences? _prefs;
  FirebaseSyncService? _firebase;
  StreamSubscription<List<Machine>>? _machineSubscription;
  StreamSubscription<List<AppUser>>? _userSubscription;

  AppState() {
    _machines = buildSeedMachines();
    _users = buildSeedUsers();
    _currentUser = _users.first;
    _seedLogs();
    _initialize();
    _startStatusSweep();
  }

  /// Deprecated: machine detail is now reached via Navigator.push to
  /// [MachineDetailScreen]. Retained so any external/test code that
  /// referenced `_selectedMachineId` still compiles.
  @Deprecated('Use Navigator.push(MachineDetailScreen(machineId: ...)) instead')
  Machine? get selectedMachineOrNull {
    final id = _selectedMachineId;
    if (id == null) return null;
    for (final machine in _machines) {
      if (machine.id == id) return machine;
    }
    return null;
  }

  int get runningCount => _machines.where((m) => m.status == MachineStatus.running).length;
  int get maintenanceCount => _machines.where((m) => m.status == MachineStatus.maintenance).length;
  int get stoppedCount => _machines.where((m) => m.status == MachineStatus.stopped).length;
  int get faultCount => _machines.where((m) => m.status == MachineStatus.fault).length;

  double get overallOee {
    if (_machines.isEmpty) return 0;
    final total = _machines.fold<int>(0, (sum, machine) => sum + machine.efficiency);
    return total / _machines.length;
  }

  double get energyConsumption => _machines.fold<double>(0, (sum, machine) => sum + machine.power) * _rangeMultiplier;
  double get operatingHours => runningCount * 8.4 * _rangeMultiplier;
  double get downtimeHours => (maintenanceCount * 2.2 + stoppedCount * 3.4 + faultCount * 5.1) * _rangeMultiplier;
  int get productionOutput => _machines.fold<int>(0, (sum, machine) => sum + machine.productionCount);

  double get _rangeMultiplier {
    switch (_range) {
      case DashboardRange.day:
        return 1;
      case DashboardRange.week:
        return 7;
      case DashboardRange.month:
        return 30;
      case DashboardRange.quarter:
        return 90;
      case DashboardRange.year:
        return 365;
      case DashboardRange.custom:
        return 14;
    }
  }

  List<MachineMetricPoint> get factoryPerformanceSeries {
    final labels = switch (_range) {
      DashboardRange.day => ['00', '04', '08', '12', '16', '20', '24'],
      DashboardRange.week => ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
      DashboardRange.month => ['W1', 'W2', 'W3', 'W4'],
      DashboardRange.quarter => ['M1', 'M2', 'M3'],
      DashboardRange.year => ['Q1', 'Q2', 'Q3', 'Q4'],
      DashboardRange.custom => ['S1', 'S2', 'S3', 'S4', 'S5'],
    };
    return labels.asMap().entries.map((entry) {
      final base = overallOee - 6 + (entry.key * 2.3);
      return MachineMetricPoint(label: entry.value, value: base.clamp(55, 99).toDouble());
    }).toList(growable: false);
  }

  List<Machine> get topDowntimeMachines {
    final sorted = [..._machines]..sort((a, b) => _downtimeScore(b).compareTo(_downtimeScore(a)));
    return sorted.take(5).toList(growable: false);
  }

  List<Machine> get topEnergyMachines {
    final sorted = [..._machines]..sort((a, b) => b.power.compareTo(a.power));
    return sorted.take(5).toList(growable: false);
  }

  Future<void> _initialize() async {
    _prefs = await SharedPreferences.getInstance();
    await _restoreCachedState();

    final bootstrap = await FirebaseBootstrap.initialize();
    _firebaseEnabled = bootstrap.configured;
    _firebaseStatus = bootstrap.message;

    if (_firebaseEnabled) {
      _firebase = FirebaseSyncService();
      await _firebase!.seedIfEmpty(_machines, _users);
      await _loadRemoteSettings();
      _bindRemoteStreams();
    }

    _isBootstrapping = false;
    notifyListeners();
  }

  void _bindRemoteStreams() {
    final service = _firebase;
    if (service == null) return;

    _machineSubscription?.cancel();
    _userSubscription?.cancel();

    _machineSubscription = service.watchMachines().listen((remoteMachines) {
      if (remoteMachines.isEmpty) return;
      _machines = remoteMachines;
      _persistCachedState();
      notifyListeners();
    });

    _userSubscription = service.watchUsers().listen((remoteUsers) {
      if (remoteUsers.isEmpty) return;
      _users = remoteUsers;
      final updatedUser = _users.firstWhereOrNull((user) => user.email == _currentUser.email);
      if (updatedUser != null) {
        _currentUser = updatedUser;
      }
      _persistCachedState();
      notifyListeners();
    });
  }

  Future<void> _restoreCachedState() async {
    final prefs = _prefs;
    if (prefs == null) return;

    final machineJson = prefs.getString('machines_cache');
    if (machineJson != null && machineJson.isNotEmpty) {
      final decoded = jsonDecode(machineJson) as List<dynamic>;
      _machines = decoded
          .map((item) => Machine.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false);
    }

    final userJson = prefs.getString('users_cache');
    if (userJson != null && userJson.isNotEmpty) {
      final decoded = jsonDecode(userJson) as List<dynamic>;
      _users = decoded
          .map((item) => AppUser.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false);
    }

    // Reports — saved locally; if the cache exists, prefer it over seed.
    final reportsJson = prefs.getString('reports_cache');
    if (reportsJson != null && reportsJson.isNotEmpty) {
      try {
        final decoded = jsonDecode(reportsJson) as List<dynamic>;
        _reports
          ..clear()
          ..addAll(decoded.map(
            (item) => MachineReport.fromMap(Map<String, dynamic>.from(item as Map)),
          ));
      } catch (_) {
        // Corrupt cache — fall back to seed.
      }
    }

    final isDark = prefs.getBool('theme_dark') ?? true;
    _themeMode = isDark ? ThemeMode.dark : ThemeMode.light;
    _notificationsEnabled = prefs.getBool('notifications_enabled') ?? true;
    _offlineMode = prefs.getBool('offline_mode') ?? true;
    temperatureThreshold = prefs.getDouble('temperature_threshold') ?? temperatureThreshold;
    vibrationThreshold = prefs.getDouble('vibration_threshold') ?? vibrationThreshold;
    powerThreshold = prefs.getDouble('power_threshold') ?? powerThreshold;

    final currentUserId = prefs.getString('current_user_id');
    final restoredUser = _users.firstWhereOrNull((user) => user.id == currentUserId);
    if (restoredUser != null) {
      _currentUser = restoredUser;
    }
  }

  Future<void> _persistCachedState() async {
    final prefs = _prefs;
    if (prefs == null) return;

    await prefs.setString('machines_cache', jsonEncode(_machines.map((machine) => machine.toMap()).toList(growable: false)));
    await prefs.setString('users_cache', jsonEncode(_users.map((user) => user.toMap()).toList(growable: false)));
    await prefs.setString('reports_cache', jsonEncode(_reports.map((r) => r.toMap()).toList(growable: false)));
    await prefs.setString('current_user_id', _currentUser.id);
    await prefs.setBool('theme_dark', _themeMode == ThemeMode.dark);
    await prefs.setBool('notifications_enabled', _notificationsEnabled);
    await prefs.setBool('offline_mode', _offlineMode);
    await prefs.setDouble('temperature_threshold', temperatureThreshold);
    await prefs.setDouble('vibration_threshold', vibrationThreshold);
    await prefs.setDouble('power_threshold', powerThreshold);
  }

  Future<void> _loadRemoteSettings() async {
    final service = _firebase;
    if (service == null) return;
    final settings = await service.loadFactorySettings();
    if (settings == null) return;

    temperatureThreshold = ((settings['temperatureThreshold'] ?? temperatureThreshold) as num).toDouble();
    vibrationThreshold = ((settings['vibrationThreshold'] ?? vibrationThreshold) as num).toDouble();
    powerThreshold = ((settings['powerThreshold'] ?? powerThreshold) as num).toDouble();
    _notificationsEnabled = settings['notificationsEnabled'] ?? _notificationsEnabled;
    _offlineMode = settings['offlineMode'] ?? _offlineMode;
    _persistCachedState();
  }

  void _saveFactorySettings() {
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.saveFactorySettings(<String, dynamic>{
        'temperatureThreshold': temperatureThreshold,
        'vibrationThreshold': vibrationThreshold,
        'powerThreshold': powerThreshold,
        'notificationsEnabled': _notificationsEnabled,
        'offlineMode': _offlineMode,
        'updatedAt': DateTime.now().toIso8601String(),
      }).catchError((_) {});
    }
  }

  Future<String?> login({required String email, required String password}) async {
    final normalizedEmail = email.trim().toLowerCase();
    _authBusy = true;
    notifyListeners();

    try {
      if (_firebaseEnabled && _firebase != null) {
        // Step 1: Firebase Auth sign-in (returns uid).
        final uid = await _firebase!.signIn(normalizedEmail, password);

        // Step 2: load-or-auto-create users/{uid}. The service uses
        // resolveSeedRole(email) to assign a role from the allowlist
        // when the document does not yet exist. Unknown emails default
        // to viewer — never admin.
        final profile = await _firebase!.loadOrCreateProfile(
          uid: uid,
          email: normalizedEmail,
          displayName: _firebase!.currentAuthUser?.displayName,
        );

        // Step 3: enforce isActive. Disabled accounts are bounced out
        // immediately so the rest of the app can trust the gate.
        if (!profile.isActive) {
          await _firebase!.signOut();
          return 'This account has been disabled. Contact your administrator.';
        }

        _currentUser = profile;
        // Make sure the locally-cached _users list contains this profile
        // so admin screens see it without waiting for the watchUsers
        // stream to land.
        final existingIndex =
            _users.indexWhere((u) => u.id == profile.id || u.email == profile.email);
        if (existingIndex >= 0) {
          final next = List<AppUser>.from(_users);
          next[existingIndex] = profile;
          _users = next;
        } else {
          _users = [..._users, profile];
        }

        _firebaseStatus = 'Signed in as ${profile.email} (${roleLabel(profile.role)})';
      } else {
        // Demo / offline: fall back to the local catalog by email.
        final localUser = _users.firstWhereOrNull(
          (user) => user.email.toLowerCase() == normalizedEmail,
        );
        if (localUser != null) {
          _currentUser = localUser.copyWith(lastLoginAt: DateTime.now());
        } else {
          // Apply the same allowlist so the role chips in demo mode
          // still behave sanely when an unknown email is entered.
          final role = resolveSeedRole(normalizedEmail);
          _currentUser = AppUser(
            id: 'demo-${DateTime.now().millisecondsSinceEpoch}',
            name: normalizedEmail.split('@').first,
            email: normalizedEmail,
            role: role,
            assignedArea: '',
            isActive: true,
            createdAt: DateTime.now(),
            updatedAt: DateTime.now(),
            lastLoginAt: DateTime.now(),
            createdBy: 'demo-mode',
          );
          _users = [..._users, _currentUser];
        }
        _firebaseStatus =
            'Demo sign-in (${roleLabel(_currentUser.role)}). Firebase will activate once real options are wired in.';
      }

      _isLoggedIn = true;
      // After successful login, every role lands on Dashboard. The
      // post-login tab is reset here in case logout() left it on a
      // different tab in a previous session.
      _currentTab = AppTab.dashboard;
      if (_firebaseEnabled) {
        _simulator?.cancel();
      } else {
        _startSimulator();
      }
      await _persistCachedState();
      return null;
    } on FirebaseAuthException catch (error) {
      // Translate Firebase Auth error codes into UI-safe messages.
      // Critically: user-not-found / invalid-credential both surface
      // as "contact admin" so the user is steered toward the
      // request-an-account flow instead of confused "wrong password"
      // messaging when the account doesn't exist at all.
      switch (error.code) {
        case 'user-not-found':
        case 'invalid-credential':
        case 'invalid-email':
          return 'No account found for that email. Contact admin if you need access.';
        case 'wrong-password':
          return 'Incorrect password. If you forgot it, contact admin.';
        case 'user-disabled':
          return 'This account has been disabled. Contact your administrator.';
        case 'too-many-requests':
          return 'Too many sign-in attempts. Please try again in a few minutes.';
        case 'network-request-failed':
          return 'Network error. Check your connection and try again.';
        default:
          return error.message ?? 'Authentication failed.';
      }
    } catch (error) {
      return 'Login failed: $error';
    } finally {
      _authBusy = false;
      notifyListeners();
    }
  }

  /// Send a Firebase Auth password-reset email to [email].
  ///
  /// Returns `null` on success, or a UI-safe error string on failure.
  /// In demo / offline mode this returns an error immediately since there
  /// is no Firebase Auth to call.
  Future<String?> resetPassword({required String email}) async {
    final normalizedEmail = email.trim().toLowerCase();

    if (!_firebaseEnabled || _firebase == null) {
      return 'Password reset requires a live Firebase connection. '
          'Make sure Firebase is configured and you are online.';
    }

    try {
      await _firebase!.sendPasswordResetEmail(normalizedEmail);
      return null; // success
    } on FirebaseAuthException catch (e) {
      switch (e.code) {
        case 'user-not-found':
        case 'invalid-credential':
          return 'No account found for that email address.';
        case 'invalid-email':
          return 'Enter a valid email address.';
        case 'too-many-requests':
          return 'Too many requests. Please try again later.';
        case 'network-request-failed':
          return 'Network error. Check your connection and try again.';
        default:
          return e.message ?? 'Failed to send reset email.';
      }
    } catch (e) {
      return 'Failed to send reset email: $e';
    }
  }

  void logout() {
    _isLoggedIn = false;
    _selectedMachineId = null;
    // Reset to Dashboard so the next login lands there, regardless
    // of role.
    _currentTab = AppTab.dashboard;
    _simulator?.cancel();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.signOut().catchError((_) {});
    }
    notifyListeners();
  }

  void setTab(AppTab tab) {
    _currentTab = tab;
    if (tab != AppTab.dashboard) {
      _selectedMachineId = null;
    }
    notifyListeners();
  }

  /// Deprecated: machine detail is now reached via Navigator.push to
  /// [MachineDetailScreen]. Kept as a back-compat no-op-ish setter
  /// in case any external code still calls it.
  @Deprecated('Use Navigator.push(MachineDetailScreen(machineId: ...)) instead')
  void openMachineDetail(String machineId) {
    _selectedMachineId = machineId;
    _currentTab = AppTab.dashboard;
    notifyListeners();
  }

  /// Deprecated: pop the route stack instead.
  @Deprecated('Use Navigator.of(context).maybePop() instead')
  void backToMachines() {
    _selectedMachineId = null;
    _currentTab = AppTab.machines;
    notifyListeners();
  }

  void setRange(DashboardRange value) {
    _range = value;
    notifyListeners();
  }

  void toggleTheme(bool dark) {
    _themeMode = dark ? ThemeMode.dark : ThemeMode.light;
    _persistCachedState();
    notifyListeners();
  }

  /// Deprecated — language toggle was removed in the English-only refactor.
  /// Retained as a no-op so legacy callers keep compiling.
  @Deprecated('Removed: app is English-only. This is now a no-op.')
  void toggleLanguage(bool arabic) {
    notifyListeners();
  }

  void toggleNotifications(bool value) {
    _notificationsEnabled = value;
    _saveFactorySettings();
    notifyListeners();
  }

  void toggleOfflineMode(bool value) {
    _offlineMode = value;
    _saveFactorySettings();
    notifyListeners();
  }

  /// **REMOVED in Phase 2.5.** Role switching from the UI is no longer
  /// supported. Roles come exclusively from Firebase Auth + Firestore
  /// `users/{uid}.role`. To change a user's role, an admin must edit
  /// the Firestore document (Admin → User Management → Edit user).
  ///
  /// Kept as a deprecated no-op so any older test/external code keeps
  /// compiling, but it does nothing.
  @Deprecated('Roles come from Firestore. Use the User Management UI to change roles.')
  void switchRole(AppRole role) {
    // Intentionally a no-op. Do not re-enable.
  }

  Future<void> syncNow() async {
    final service = _firebase;
    if (!_firebaseEnabled || service == null) {
      _firebaseStatus = 'Firebase is not configured yet.';
      notifyListeners();
      return;
    }

    _syncBusy = true;
    _firebaseStatus = 'Syncing machines, users, and settings...';
    notifyListeners();

    try {
      for (final machine in _machines) {
        await service.upsertMachine(machine);
      }
      for (final user in _users) {
        await service.upsertUser(user);
      }
      await service.saveFactorySettings(<String, dynamic>{
        'temperatureThreshold': temperatureThreshold,
        'vibrationThreshold': vibrationThreshold,
        'powerThreshold': powerThreshold,
        'notificationsEnabled': _notificationsEnabled,
        'offlineMode': _offlineMode,
        'updatedAt': DateTime.now().toIso8601String(),
      });
      _firebaseStatus = 'Firestore sync complete.';
    } catch (error) {
      _firebaseStatus = 'Sync failed: $error';
    } finally {
      _syncBusy = false;
      notifyListeners();
    }
  }

  void addMachine(Machine machine) {
    if (_blockWrite(permissions.canAddMachines, 'addMachine')) return;
    _machines = [..._machines, machine];
    _pushHistory(
      'Machine added',
      '${machine.name} (${machine.code}) was added by admin.',
      Icons.add_box_rounded,
      DT.cyan,
      machineId: machine.id,
      kind: HistoryKind.config,
    );
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertMachine(machine).catchError((_) {});
    }
    notifyListeners();
  }

  void updateMachine(Machine machine) {
    if (_blockWrite(permissions.canEditMachines, 'updateMachine')) return;
    _machines = _machines.map((item) => item.id == machine.id ? machine : item).toList(growable: false);
    _pushHistory(
      'Machine updated',
      '${machine.name} configuration was updated.',
      Icons.edit_rounded,
      DT.blue,
      machineId: machine.id,
      kind: HistoryKind.config,
    );
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertMachine(machine).catchError((_) {});
    }
    notifyListeners();
  }

  /// Updates the imageUrl + Storage path on a machine. Pass `null`
  /// for both to clear the photo. The old Storage blob (if its path
  /// was previously stored) is best-effort deleted; failure is logged
  /// but not fatal.
  ///
  /// Defense-in-depth gated to admin only.
  Future<void> setMachineImageUrl(
    String machineId, {
    required String? imageUrl,
    required String? imageStoragePath,
  }) async {
    if (_blockWrite(
        permissions.canUploadMachineImages, 'setMachineImageUrl')) {
      return;
    }
    final m = _machines.firstWhereOrNull((it) => it.id == machineId);
    if (m == null) return;
    final oldStoragePath = m.imageStoragePath;

    final updated = m.copyWith(
      imageUrl: imageUrl,
      clearImageUrl: imageUrl == null,
      imageStoragePath: imageStoragePath,
      clearImageStoragePath: imageStoragePath == null,
    );
    _machines = _machines
        .map((item) => item.id == machineId ? updated : item)
        .toList(growable: false);
    _pushHistory(
      imageUrl == null ? 'Machine image removed' : 'Machine image updated',
      m.name,
      Icons.image_outlined,
      DT.cyan,
      machineId: machineId,
      kind: HistoryKind.config,
    );
    _persistCachedState();

    if (_firebaseEnabled && _firebase != null) {
      try {
        await _firebase!.upsertMachine(updated);
      } catch (_) {
        // Non-fatal: the local update still reflects in UI.
      }
    }

    // Best-effort delete of the previous Storage blob, if any. Done
    // AFTER the Firestore write so a delete failure can't strand the
    // doc pointing at a missing object — it's the safe order.
    if (oldStoragePath != null &&
        oldStoragePath.isNotEmpty &&
        oldStoragePath != imageStoragePath) {
      // ignore: unawaited_futures
      StorageService().deleteByPath(oldStoragePath);
    }

    notifyListeners();
  }

  /// Update the signed-in user's own avatar. Anyone can update their
  /// own — there's no role gate, but writes only ever touch the
  /// current user's profile. Pass `null` to clear.
  Future<void> setMyPhotoUrl({
    required String? photoUrl,
    required String? photoStoragePath,
  }) async {
    final oldStoragePath = _currentUser.photoStoragePath;
    final updated = _currentUser.copyWith(
      photoUrl: photoUrl,
      clearPhotoUrl: photoUrl == null,
      photoStoragePath: photoStoragePath,
      clearPhotoStoragePath: photoStoragePath == null,
      updatedAt: DateTime.now(),
    );
    _currentUser = updated;
    _users = _users
        .map((u) => u.id == updated.id ? updated : u)
        .toList(growable: false);
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertUser(updated).catchError((_) {});
    }
    if (oldStoragePath != null &&
        oldStoragePath.isNotEmpty &&
        oldStoragePath != photoStoragePath) {
      // ignore: unawaited_futures
      StorageService().deleteByPath(oldStoragePath);
    }
    notifyListeners();
  }

  void deleteMachine(String machineId) {
    if (_blockWrite(permissions.canDeleteMachines, 'deleteMachine')) return;
    final target = _machines.firstWhereOrNull((machine) => machine.id == machineId);
    _machines = _machines.where((machine) => machine.id != machineId).toList(growable: false);
    if (_selectedMachineId == machineId) {
      _selectedMachineId = null;
    }
    if (target != null) {
      _pushHistory(
        'Machine removed',
        '${target.name} was removed from the factory roster.',
        Icons.delete_forever_rounded,
        DT.red,
        machineId: machineId,
        kind: HistoryKind.config,
      );
    }
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.deleteMachine(machineId).catchError((_) {});
    }
    notifyListeners();
  }

  void addUser(AppUser user) {
    if (_blockWrite(permissions.canManageUsers, 'addUser')) return;
    _users = [..._users, user];
    _pushHistory('User added', '${user.name} joined as ${user.role.name}.', Icons.person_add_alt_1_rounded, DT.cyan);
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertUser(user).catchError((_) {});
    }
    notifyListeners();
  }

  /// Awaitable variant of [addUser] that surfaces Firestore errors so
  /// the UI can show a clear message and retry. Returns `null` on
  /// success, or a UI-safe error string on failure. Defense-in-depth
  /// gate stays in place — viewers still cannot add users.
  ///
  /// Note: this writes the Firestore profile only. Creating the
  /// Firebase Auth account itself still requires a Cloud Function
  /// using the Admin SDK (documented as backend-required in earlier
  /// phases). The local users list is updated either way so the
  /// admin sees the row immediately.
  Future<String?> addUserAsync(AppUser user) async {
    if (_blockWrite(permissions.canManageUsers, 'addUserAsync')) {
      return 'You do not have permission to add users.';
    }
    // Local list is updated optimistically so the row is visible
    // even if Firestore is offline.
    _users = [..._users, user];
    _pushHistory(
      'User added',
      '${user.name} joined as ${user.role.name}.',
      Icons.person_add_alt_1_rounded,
      DT.cyan,
    );
    _persistCachedState();
    notifyListeners();

    if (_firebaseEnabled && _firebase != null) {
      try {
        await _firebase!.upsertUser(user);
        return null;
      } catch (e) {
        return 'User saved locally but Firestore write failed: $e';
      }
    }
    return null;
  }

  void updateUser(AppUser user) {
    // Allowed if (a) admin, OR (b) the user is editing their own profile
    // but only non-privileged fields. Field-level enforcement happens
    // via firestore.rules; the client-side check here is coarse but
    // correct for the in-app surface (user form is admin-only; profile
    // photo edits go through setMyPhotoUrl).
    final isSelf = user.id == _currentUser.id;
    if (_blockWrite(
        permissions.canManageUsers || isSelf, 'updateUser')) {
      return;
    }
    _users = _users.map((item) => item.id == user.id ? user : item).toList(growable: false);
    if (_currentUser.id == user.id) {
      _currentUser = user;
    }
    _pushHistory('User updated', '${user.name} profile was updated.', Icons.manage_accounts_rounded, DT.blue);
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertUser(user).catchError((_) {});
    }
    notifyListeners();
  }

  void deleteUser(String userId) {
    if (_blockWrite(permissions.canDeleteUsers, 'deleteUser')) return;
    if (_currentUser.id == userId) return;

    final target = _users.firstWhereOrNull((user) => user.id == userId);

    // Refuse to remove the last admin — guards against accidental lockout.
    if (target?.role == AppRole.admin) {
      final remainingAdmins = _users
          .where((u) => u.role == AppRole.admin && u.id != userId)
          .length;
      if (remainingAdmins == 0) return;
    }

    _users = _users.where((user) => user.id != userId).toList(growable: false);
    if (target != null) {
      _pushHistory(
        'User removed',
        '${target.name} was removed from access control.',
        Icons.person_remove_alt_1_rounded,
        DT.red,
      );
    }
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.deleteUser(userId).catchError((_) {});
    }
    notifyListeners();
  }

  void startMachine(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'startMachine')) return;
    _updateStatus(machineId, MachineStatus.running, 'Machine started', 'Production resumed manually.', clearAlert: true);
  }

  void stopMachine(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'stopMachine')) return;
    _updateStatus(machineId, MachineStatus.stopped, 'Machine stopped', 'Machine stopped manually from quick actions.', alert: 'Stopped manually');
  }

  void requestMaintenance(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'requestMaintenance')) return;
    _updateStatus(machineId, MachineStatus.maintenance, 'Maintenance requested', 'Maintenance ticket created and routed.', alert: 'Maintenance request sent');
  }

  void reportIssue(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'reportIssue')) return;
    _updateStatus(machineId, MachineStatus.fault, 'Issue reported', 'Critical issue escalated to maintenance team.', alert: 'Critical issue reported');
  }

  void updateThresholds({double? temperature, double? vibration, double? power}) {
    if (temperature != null) temperatureThreshold = temperature;
    if (vibration != null) vibrationThreshold = vibration;
    if (power != null) powerThreshold = power;
    _saveFactorySettings();
    notifyListeners();
  }

  // ── Reports ──────────────────────────────────────────────────────────────
  /// Engineer/admin creates a new report against a machine.
  /// Permission gate: caller must verify [permissions.canWriteReports].
  /// Returns the created report so the UI can scroll to it / show a toast.
  /// Throws [StateError] if the caller lacks permission. The UI gates
  /// the button on `permissions.canWriteReports` so this throw is purely
  /// a defense-in-depth check against bypasses (stale callbacks, tests,
  /// debug builds).
  MachineReport addReport({
    required String machineId,
    required String title,
    required String description,
    required String recommendedAction,
    required ReportSeverity severity,
    ReportStatus status = ReportStatus.open,
  }) {
    if (_blockWrite(permissions.canWriteReports, 'addReport')) {
      throw StateError('addReport blocked: caller lacks permission.');
    }
    final machine = _machines.firstWhereOrNull((m) => m.id == machineId);
    final report = MachineReport(
      id: 'rpt-${DateTime.now().millisecondsSinceEpoch}',
      machineId: machineId,
      title: title,
      description: description,
      recommendedAction: recommendedAction,
      severity: severity,
      status: status,
      authorId: _currentUser.id,
      authorName: _currentUser.name,
      createdAt: DateTime.now(),
    );
    _reports.insert(0, report);
    _pushHistory(
      'Report filed: $title',
      description.isEmpty ? 'No further detail.' : description,
      Icons.fact_check_rounded,
      severity.color,
      machineId: machineId,
      kind: HistoryKind.report,
      severity: severity.label,
      author: _currentUser.name,
    );
    // If the report is critical and the machine is healthy, also surface
    // an alert — emergency-only, matching the new alerts policy.
    if (severity == ReportSeverity.critical) {
      _alerts.insert(
        0,
        AppAlert(
          id: 'alr-${DateTime.now().millisecondsSinceEpoch}',
          machineId: machineId,
          title: 'Critical report filed',
          message: '${machine?.name ?? machineId}: $title',
          time: DateTime.now(),
          color: DT.red,
        ),
      );
    }
    // Push to Firestore: machines/{id}/reports/{reportId}.
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertReport(report).catchError((_) {});
      // Also append a parallel history event in machines/{id}/history so
      // server-only consumers see the same timeline.
      _firebase!.appendHistoryEvent(machineId, {
        'type': 'report',
        'title': 'Report filed: $title',
        'description': description,
        'severity': severity.label,
        'createdByUid': _currentUser.id,
        'createdByName': _currentUser.name,
        'source': _currentUser.role == AppRole.admin ? 'admin' : 'engineer',
      }).catchError((_) {});
    }
    _persistCachedState();
    // A new report can flip a machine from "Fault" (no telemetry, no
    // report) to "Under Maintenance" (no telemetry, has report).
    // Record that transition.
    _recordStatusTransitionsIfAny();
    notifyListeners();
    return report;
  }

  /// Update an existing report. Engineers can update only their own;
  /// admins can update any. Defense-in-depth: caller must be admin or
  /// the original author.
  void updateReport(MachineReport report) {
    final canEdit = permissions.canEditOwnReport(
      report.authorId,
      _currentUser.id,
    );
    if (_blockWrite(canEdit, 'updateReport')) return;
    _reports
      ..removeWhere((r) => r.id == report.id)
      ..insert(0, report);
    _pushHistory(
      'Report updated: ${report.title}',
      'Status: ${report.status.label}',
      Icons.edit_note_rounded,
      report.severity.color,
      machineId: report.machineId,
      kind: HistoryKind.report,
      severity: report.severity.label,
    );
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertReport(report).catchError((_) {});
    }
    _persistCachedState();
    notifyListeners();
  }

  /// Delete a report (admin only).
  void deleteReport(String reportId) {
    if (_blockWrite(permissions.canDeleteReports, 'deleteReport')) return;
    final target = _reports.firstWhereOrNull((r) => r.id == reportId);
    if (target == null) return;
    _reports.removeWhere((r) => r.id == reportId);
    _pushHistory(
      'Report deleted',
      target.title,
      Icons.delete_outline_rounded,
      DT.red,
      machineId: target.machineId,
      kind: HistoryKind.report,
    );
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.deleteReport(target.machineId, reportId).catchError((_) {});
    }
    _persistCachedState();
    notifyListeners();
  }

  void _updateStatus(String machineId, MachineStatus status, String title, String detail, {String? alert, bool clearAlert = false}) {
    final machine = _machines.firstWhereOrNull((item) => item.id == machineId);
    if (machine == null) return;
    final updated = machine.copyWith(
      status: status,
      activeAlert: alert,
      clearAlert: clearAlert,
      lastUpdated: DateTime.now(),
      timeline: [
        MachineTimelineEvent(time: DateTime.now(), title: title, detail: detail, color: _statusColor(status)),
        ...machine.timeline,
      ].take(5).toList(growable: false),
    );
    // Note: updateMachine itself emits a generic 'Machine updated' history
    // entry, so we tag the more specific status-change entry afterwards.
    _machines = _machines.map((item) => item.id == updated.id ? updated : item).toList(growable: false);
    _pushHistory(
      title,
      detail,
      _statusIcon(status),
      _statusColor(status),
      machineId: machine.id,
      kind: status == MachineStatus.fault
          ? HistoryKind.fault
          : status == MachineStatus.maintenance
              ? HistoryKind.maintenance
              : HistoryKind.status,
      severity: status == MachineStatus.fault
          ? 'CRITICAL'
          : status == MachineStatus.maintenance
              ? 'WARNING'
              : 'INFO',
    );
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertMachine(updated).catchError((_) {});
    }
    if (alert != null) {
      _alerts.insert(
        0,
        AppAlert(
          id: '${machine.id}-${DateTime.now().millisecondsSinceEpoch}',
          machineId: machine.id,
          title: title,
          message: '${machine.name}: $alert',
          time: DateTime.now(),
          color: _statusColor(status),
        ),
      );
    }
    notifyListeners();
  }

  IconData _statusIcon(MachineStatus status) {
    switch (status) {
      case MachineStatus.running:
        return Icons.play_circle_rounded;
      case MachineStatus.maintenance:
        return Icons.build_circle_rounded;
      case MachineStatus.fault:
        return Icons.error_rounded;
      case MachineStatus.stopped:
        return Icons.pause_circle_rounded;
      case MachineStatus.offline:
        return Icons.cloud_off_rounded;
    }
  }

  void _startSimulator() {
    if (_firebaseEnabled) return;
    _simulator?.cancel();
    _simulator = Timer.periodic(const Duration(seconds: 5), (_) {
      _machines = _machines.map((machine) {
        if (machine.status == MachineStatus.stopped || machine.status == MachineStatus.maintenance) {
          return machine;
        }
        final delta = _random.nextDouble() * 4 - 2;
        final vibDelta = _random.nextDouble() * 0.08 - 0.04;
        final eff = (machine.efficiency + delta).round().clamp(35, 99);
        final temp = (machine.temperature + delta).clamp(24, 96).toDouble();
        final vibration = (machine.vibration + vibDelta).clamp(0.05, 0.92).toDouble();
        final power = (machine.power + (_random.nextDouble() * 1.4 - 0.7)).clamp(0.4, 32).toDouble();
        final prod = machine.productionCount + (machine.status == MachineStatus.running ? _random.nextInt(12) : 0);
        MachineStatus status = machine.status;
        String? alert = machine.activeAlert;
        if (temp >= temperatureThreshold || vibration >= vibrationThreshold || power >= powerThreshold) {
          status = MachineStatus.fault;
          alert = 'Threshold exceeded';
        } else if (machine.status == MachineStatus.fault) {
          status = MachineStatus.running;
          alert = null;
        }
        return machine.copyWith(
          efficiency: eff,
          temperature: temp,
          vibration: vibration,
          power: power,
          productionCount: prod,
          activeAlert: alert,
          clearAlert: alert == null,
          status: status,
          lastUpdated: DateTime.now(),
          performanceSeries: [
            ...machine.performanceSeries.skip(1),
            MachineMetricPoint(label: '${DateTime.now().hour}:${DateTime.now().minute.toString().padLeft(2, '0')}', value: eff.toDouble()),
          ],
        );
      }).toList(growable: false);
      _persistCachedState();
      notifyListeners();
    });
  }

  void _seedLogs() {
    final now = DateTime.now();
    _history.addAll([
      HistoryLog(
        id: 'h1',
        time: now.subtract(const Duration(minutes: 5)),
        title: 'Welding fault detected',
        subtitle: 'Arc quality alarm triggered automatic stop.',
        icon: Icons.error_outline_rounded,
        color: DT.red,
        machineId: 'wld-06',
        kind: HistoryKind.fault,
        severity: 'CRITICAL',
        author: 'System',
      ),
      HistoryLog(
        id: 'h2',
        time: now.subtract(const Duration(minutes: 14)),
        title: 'Hydraulic maintenance started',
        subtitle: 'Maintenance ticket assigned to Mariam.',
        icon: Icons.build_circle_rounded,
        color: DT.yellow,
        machineId: 'hyd-03',
        kind: HistoryKind.maintenance,
        severity: 'WARNING',
        author: 'Mariam Engineer',
      ),
      HistoryLog(
        id: 'h3',
        time: now.subtract(const Duration(minutes: 24)),
        title: 'Packaging target achieved',
        subtitle: 'Packaging Unit reached 1260 items.',
        icon: Icons.verified_rounded,
        color: DT.green,
        machineId: 'pkg-08',
        kind: HistoryKind.status,
        severity: 'INFO',
        author: 'System',
      ),
      HistoryLog(
        id: 'h4',
        time: now.subtract(const Duration(hours: 2, minutes: 12)),
        title: 'Routine inspection logged',
        subtitle: 'CNC-01 spindle alignment within tolerance.',
        icon: Icons.fact_check_rounded,
        color: DT.green,
        machineId: 'cnc-01',
        kind: HistoryKind.report,
        severity: 'INFO',
        author: 'Alaa Engineer',
      ),
      HistoryLog(
        id: 'h5',
        time: now.subtract(const Duration(hours: 5, minutes: 40)),
        title: 'Coolant level low',
        subtitle: 'Operator topped up to nominal.',
        icon: Icons.water_drop_rounded,
        color: DT.yellow,
        machineId: 'cnc-01',
        kind: HistoryKind.maintenance,
        severity: 'WARNING',
        author: 'Mariam Engineer',
      ),
    ]);
    _alerts.addAll([
      AppAlert(
        id: 'a1',
        machineId: 'wld-06',
        title: 'Critical fault',
        message: 'Welding Robot exceeded vibration threshold.',
        time: now.subtract(const Duration(minutes: 6)),
        color: DT.red,
      ),
    ]);
    // Seed a couple of demo reports tied to the same machines.
    _reports.addAll([
      MachineReport(
        id: 'r-seed-1',
        machineId: 'cnc-01',
        title: 'Spindle vibration trending up',
        description:
            'Vibration on CNC-01 has crept up by ~12% over the last week. '
            'Within tolerance but worth scheduling preventive bearing check.',
        recommendedAction:
            'Schedule a bearing inspection during the next maintenance window.',
        severity: ReportSeverity.medium,
        status: ReportStatus.open,
        authorId: 'u-02',
        authorName: 'Alaa Engineer',
        createdAt: now.subtract(const Duration(hours: 2, minutes: 12)),
      ),
      MachineReport(
        id: 'r-seed-2',
        machineId: 'hyd-03',
        title: 'Hydraulic pressure dip during cycle',
        description:
            'Pressure dropped briefly at start of cycle. Replaced filter; '
            'monitoring next 24h.',
        recommendedAction: 'Re-check pressure curve tomorrow.',
        severity: ReportSeverity.high,
        status: ReportStatus.inProgress,
        authorId: 'u-03',
        authorName: 'Mariam Engineer',
        createdAt: now.subtract(const Duration(hours: 14)),
      ),
    ]);
  }

  void _pushHistory(
    String title,
    String subtitle,
    IconData icon,
    Color color, {
    String? machineId,
    HistoryKind kind = HistoryKind.status,
    String? severity,
    String? author,
  }) {
    _history.insert(
      0,
      HistoryLog(
        id: '${DateTime.now().millisecondsSinceEpoch}',
        time: DateTime.now(),
        title: title,
        subtitle: subtitle,
        icon: icon,
        color: color,
        machineId: machineId,
        kind: kind,
        severity: severity,
        author: author ?? _currentUser.name,
      ),
    );
  }

  double _downtimeScore(Machine machine) {
    return switch (machine.status) {
      MachineStatus.fault => 10,
      MachineStatus.maintenance => 7,
      MachineStatus.stopped => 5,
      MachineStatus.offline => 3,
      MachineStatus.running => 1,
    } +
        (100 - machine.efficiency) / 20;
  }

  Color _statusColor(MachineStatus status) {
    switch (status) {
      case MachineStatus.running:
        return DT.green;
      case MachineStatus.maintenance:
        return DT.yellow;
      case MachineStatus.fault:
        return DT.red;
      case MachineStatus.stopped:
        return DT.purple;
      case MachineStatus.offline:
        return DT.blue;
    }
  }

  @override
  void dispose() {
    _simulator?.cancel();
    _statusSweepTimer?.cancel();
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
