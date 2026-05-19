part of 'state.dart';

extension AppStateMachines on AppState {
  // ── Computed fleet metrics ─────────────────────────────────────────────────

  // Count getters use resolvedStatusFor() so they stay in sync with the
  // filter chips, which also resolve live status from telemetry + reports.
  int get runningCount =>
      _machines.where((m) => resolvedStatusFor(m.id) == MachineStatus.running).length;
  int get maintenanceCount =>
      _machines.where((m) => resolvedStatusFor(m.id) == MachineStatus.maintenance).length;
  int get stoppedCount =>
      _machines.where((m) => resolvedStatusFor(m.id) == MachineStatus.stopped).length;
  int get faultCount =>
      _machines.where((m) => resolvedStatusFor(m.id) == MachineStatus.fault).length;

  double get overallOee {
    if (_machines.isEmpty) return 0;
    final total = _machines.fold<int>(0, (sum, m) => sum + m.efficiency);
    return total / _machines.length;
  }

  double get energyConsumption =>
      _machines.fold<double>(0, (sum, m) => sum + m.power) * _rangeMultiplier;
  double get operatingHours => runningCount * 8.4 * _rangeMultiplier;
  double get downtimeHours =>
      (maintenanceCount * 2.2 + stoppedCount * 3.4 + faultCount * 5.1) *
      _rangeMultiplier;
  int get productionOutput =>
      _machines.fold<int>(0, (sum, m) => sum + m.productionCount);

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
      return MachineMetricPoint(
          label: entry.value, value: base.clamp(55, 99).toDouble());
    }).toList(growable: false);
  }

  List<Machine> get topDowntimeMachines {
    final sorted = [..._machines]
      ..sort((a, b) => _downtimeScore(b).compareTo(_downtimeScore(a)));
    return sorted.take(5).toList(growable: false);
  }

  List<Machine> get topEnergyMachines {
    final sorted = [..._machines]..sort((a, b) => b.power.compareTo(a.power));
    return sorted.take(5).toList(growable: false);
  }

  // ── Machine CRUD ───────────────────────────────────────────────────────────

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
    _notify();
  }

  void updateMachine(Machine machine) {
    if (_blockWrite(permissions.canEditMachines, 'updateMachine')) return;
    _machines = _machines
        .map((item) => item.id == machine.id ? machine : item)
        .toList(growable: false);
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
    _notify();
  }

  void deleteMachine(String machineId) {
    if (_blockWrite(permissions.canDeleteMachines, 'deleteMachine')) return;
    final target =
        _machines.firstWhereOrNull((machine) => machine.id == machineId);
    _machines = _machines
        .where((machine) => machine.id != machineId)
        .toList(growable: false);
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
    _notify();
  }

  Future<void> setMachineImageUrl(
    String machineId, {
    required String? imageUrl,
    required String? imageStoragePath,
  }) async {
    if (_blockWrite(permissions.canUploadMachineImages, 'setMachineImageUrl')) {
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
        if (imageUrl == null) {
          await _firebase!.clearMachineImageFields(machineId);
        } else {
          await _firebase!.upsertMachine(updated);
        }
      } catch (_) {
        // Non-fatal: the local update still reflects in UI.
      }
    }

    // Re-apply the image fields after the async Firestore write. The
    // watchMachines() stream callback may have replaced _machines with
    // a stale Firestore snapshot while we were suspended above.
    _machines = _machines
        .map((item) => item.id == machineId ? updated : item)
        .toList(growable: false);

    if (oldStoragePath != null &&
        oldStoragePath.isNotEmpty &&
        oldStoragePath != imageStoragePath) {
      // ignore: unawaited_futures
      StorageService().deleteByPath(oldStoragePath);
    }

    _notify();
  }

  // ── Machine status actions ─────────────────────────────────────────────────

  void startMachine(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'startMachine')) return;
    _updateStatus(machineId, MachineStatus.running, 'Machine started',
        'Production resumed manually.',
        clearAlert: true);
  }

  void stopMachine(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'stopMachine')) return;
    _updateStatus(machineId, MachineStatus.stopped, 'Machine stopped',
        'Machine stopped manually from quick actions.',
        alert: 'Stopped manually');
  }

  void requestMaintenance(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'requestMaintenance')) return;
    _updateStatus(
        machineId,
        MachineStatus.maintenance,
        'Maintenance requested',
        'Maintenance ticket created and routed.',
        alert: 'Maintenance request sent');
  }

  void reportIssue(String machineId) {
    if (_blockWrite(permissions.canActOnMachine, 'reportIssue')) return;
    _updateStatus(machineId, MachineStatus.fault, 'Issue reported',
        'Critical issue escalated to maintenance team.',
        alert: 'Critical issue reported');
  }

  void updateThresholds(
      {double? temperature, double? vibration, double? power}) {
    if (temperature != null) temperatureThreshold = temperature;
    if (vibration != null) vibrationThreshold = vibration;
    if (power != null) powerThreshold = power;
    _saveFactorySettings();
    _notify();
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  void _updateStatus(
    String machineId,
    MachineStatus status,
    String title,
    String detail, {
    String? alert,
    bool clearAlert = false,
  }) {
    final machine = _machines.firstWhereOrNull((item) => item.id == machineId);
    if (machine == null) return;
    final updated = machine.copyWith(
      status: status,
      activeAlert: alert,
      clearAlert: clearAlert,
      lastUpdated: DateTime.now(),
      timeline: [
        MachineTimelineEvent(
            time: DateTime.now(),
            title: title,
            detail: detail,
            color: _statusColor(status)),
        ...machine.timeline,
      ].take(5).toList(growable: false),
    );
    _machines = _machines
        .map((item) => item.id == updated.id ? updated : item)
        .toList(growable: false);
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
    _notify();
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
}
