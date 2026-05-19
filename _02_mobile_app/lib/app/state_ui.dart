part of 'state.dart';

extension AppStateUi on AppState {
  void setTab(AppTab tab) {
    _currentTab = tab;
    if (tab != AppTab.dashboard) {
      _selectedMachineId = null;
    }
    _notify();
  }

  void setRange(DashboardRange value) {
    _range = value;
    _notify();
  }

  void toggleTheme(bool dark) {
    _themeMode = dark ? ThemeMode.dark : ThemeMode.light;
    _persistCachedState();
    _notify();
  }

  void toggleNotifications(bool value) {
    _notificationsEnabled = value;
    _saveFactorySettings();
    _notify();
  }

  void toggleOfflineMode(bool value) {
    _offlineMode = value;
    _saveFactorySettings();
    _notify();
  }

  // ── Deprecated stubs ───────────────────────────────────────────────────────

  @Deprecated('Use Navigator.push(MachineDetailScreen(machineId: ...)) instead')
  Machine? get selectedMachineOrNull {
    final id = _selectedMachineId;
    if (id == null) return null;
    for (final machine in _machines) {
      if (machine.id == id) return machine;
    }
    return null;
  }

  @Deprecated('Use Navigator.push(MachineDetailScreen(machineId: ...)) instead')
  void openMachineDetail(String machineId) {
    _selectedMachineId = machineId;
    _currentTab = AppTab.dashboard;
    _notify();
  }

  @Deprecated('Use Navigator.of(context).maybePop() instead')
  void backToMachines() {
    _selectedMachineId = null;
    _currentTab = AppTab.machines;
    _notify();
  }

  /// Always returns false — language toggle was removed in the
  /// English-only refactor.
  @Deprecated('Removed: app is English-only. This is now a no-op.')
  bool get isArabic => false;

  @Deprecated('Removed: app is English-only. This is now a no-op.')
  void toggleLanguage(bool arabic) {
    _notify();
  }

  @Deprecated(
      'Roles come from Firestore. Use the User Management UI to change roles.')
  void switchRole(AppRole role) {
    // Intentionally a no-op.
  }
}
