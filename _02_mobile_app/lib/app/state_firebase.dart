part of 'state.dart';

extension AppStateFirebase on AppState {
  Future<void> _initialize() async {
    _prefs = await SharedPreferences.getInstance();
    await _restoreCachedState();

    final bootstrap = await FirebaseBootstrap.initialize();
    _firebaseEnabled = bootstrap.configured;
    _firebaseStatus = bootstrap.message;

    _isBootstrapping = false;
    _notify();

    if (_firebaseEnabled) {
      _firebase = FirebaseSyncService();
      try {
        await Future.wait([
          _firebase!.seedIfEmpty(_machines, _users),
          _loadRemoteSettings(),
        ]);
      } catch (_) {
        // Seed / settings fetch failed (e.g. network blip or cold-start
        // auth race). Streams must still be bound so real data arrives.
      }
      _bindRemoteStreams();
    }
  }

  void _bindRemoteStreams() {
    final service = _firebase;
    if (service == null) return;

    _machineSubscription?.cancel();
    _userSubscription?.cancel();

    _machineSubscription = service.watchMachines().listen(
      (remoteMachines) {
        if (remoteMachines.isEmpty) return;
        _firestoreLoaded = true;
        _machines = remoteMachines;
        _persistCachedState();
        _notify();
      },
      onError: (_) {
        // Non-fatal: keep local cache on pre-auth or network error.
        // The streams are rebound after login so this path is only hit
        // during the brief pre-authentication window in _initialize().
      },
    );

    _userSubscription = service.watchUsers().listen(
      (remoteUsers) {
        if (remoteUsers.isEmpty) return;
        _users = remoteUsers;
        final updatedUser =
            _users.firstWhereOrNull((user) => user.email == _currentUser.email);
        if (updatedUser != null) {
          _currentUser = updatedUser;
        }
        _persistCachedState();
        _notify();
      },
      onError: (_) {
        // Non-fatal: keep local cache on pre-auth or network error.
      },
    );
  }

  Future<void> _restoreCachedState() async {
    final prefs = _prefs;
    if (prefs == null) return;

    final machineJson = prefs.getString('machines_cache');
    if (machineJson != null && machineJson.isNotEmpty) {
      final decoded = jsonDecode(machineJson) as List<dynamic>;
      _machines = decoded
          .map((item) =>
              Machine.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false);
    }

    final userJson = prefs.getString('users_cache');
    if (userJson != null && userJson.isNotEmpty) {
      final decoded = jsonDecode(userJson) as List<dynamic>;
      _users = decoded
          .map((item) =>
              AppUser.fromMap(Map<String, dynamic>.from(item as Map)))
          .toList(growable: false);
    }

    final reportsJson = prefs.getString('reports_cache');
    if (reportsJson != null && reportsJson.isNotEmpty) {
      try {
        final decoded = jsonDecode(reportsJson) as List<dynamic>;
        _reports
          ..clear()
          ..addAll(decoded.map(
            (item) =>
                MachineReport.fromMap(Map<String, dynamic>.from(item as Map)),
          ));
      } catch (_) {
        // Corrupt cache — fall back to seed.
      }
    }

    final isDark = prefs.getBool('theme_dark') ?? true;
    _themeMode = isDark ? ThemeMode.dark : ThemeMode.light;
    _notificationsEnabled = prefs.getBool('notifications_enabled') ?? true;
    _offlineMode = prefs.getBool('offline_mode') ?? true;
    temperatureThreshold =
        prefs.getDouble('temperature_threshold') ?? temperatureThreshold;
    vibrationThreshold =
        prefs.getDouble('vibration_threshold') ?? vibrationThreshold;
    powerThreshold = prefs.getDouble('power_threshold') ?? powerThreshold;

    final currentUserId = prefs.getString('current_user_id');
    final restoredUser =
        _users.firstWhereOrNull((user) => user.id == currentUserId);
    if (restoredUser != null) {
      _currentUser = restoredUser;
    }
  }

  Future<void> _persistCachedState() async {
    final prefs = _prefs;
    if (prefs == null) return;

    await prefs.setString(
        'machines_cache',
        jsonEncode(_machines.map((m) => m.toMap()).toList(growable: false)));
    await prefs.setString(
        'users_cache',
        jsonEncode(_users.map((u) => u.toMap()).toList(growable: false)));
    await prefs.setString(
        'reports_cache',
        jsonEncode(_reports.map((r) => r.toMap()).toList(growable: false)));
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

    temperatureThreshold =
        ((settings['temperatureThreshold'] ?? temperatureThreshold) as num)
            .toDouble();
    vibrationThreshold =
        ((settings['vibrationThreshold'] ?? vibrationThreshold) as num)
            .toDouble();
    powerThreshold =
        ((settings['powerThreshold'] ?? powerThreshold) as num).toDouble();
    _notificationsEnabled =
        settings['notificationsEnabled'] ?? _notificationsEnabled;
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

  Future<void> syncNow() async {
    final service = _firebase;
    if (!_firebaseEnabled || service == null) {
      _firebaseStatus = 'Firebase is not configured yet.';
      _notify();
      return;
    }
    if (!_firestoreLoaded) {
      _firebaseStatus =
          'Waiting for Firestore data to finish loading. Try again in a moment.';
      _notify();
      return;
    }

    _syncBusy = true;
    _firebaseStatus = 'Syncing machines, users, and settings...';
    _notify();

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
      _notify();
    }
  }
}
