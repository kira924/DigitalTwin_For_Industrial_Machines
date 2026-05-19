part of 'state.dart';

extension AppStateTelemetry on AppState {
  int engineIdForMachine(String machineId) {
    final idx = _machines.indexWhere((m) => m.id == machineId);
    return idx >= 0 ? idx + 1 : 0;
  }

  String? machineIdForEngine(int engineId) {
    final idx = engineId - 1;
    if (idx < 0 || idx >= _machines.length) return null;
    return _machines[idx].id;
  }

  EngineTelemetryPayload? latestTelemetryFor(String machineId) {
    return _latestPayloadByMachine[machineId];
  }

  DateTime? lastSeenFor(String machineId) {
    return _lastSeenByMachine[machineId];
  }

  MachineStatus resolvedStatusFor(String machineId) {
    return MachineStatusResolver.resolve(
      lastSeen: _lastSeenByMachine[machineId],
      hasReport: _reports.any((r) => r.machineId == machineId),
    );
  }

  void ingestTelemetry(EngineTelemetryPayload payload) {
    String? machineId = payload.machineId;
    if (machineId != null) {
      final exists = _machines.any((m) => m.id == machineId);
      if (!exists) return;
    } else {
      machineId = machineIdForEngine(payload.engineId);
      if (machineId == null) return;
    }

    _latestPayloadByMachine[machineId] = payload;
    _lastSeenByMachine[machineId] = DateTime.now();

    _evaluateEmergencyForPayload(machineId, payload);
    _recordStatusTransitionsIfAny();

    _notify();
  }

  void _evaluateEmergencyForPayload(
    String machineId,
    EngineTelemetryPayload payload,
  ) {
    final machine = _machines.firstWhereOrNull((m) => m.id == machineId);
    if (machine == null) return;

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

    final reasons =
        _activeEmergencyReasons.putIfAbsent(machineId, () => <String>{});

    void fire(String code, String title, String message) {
      if (reasons.contains(code)) return;
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
      _notify();
    }

    void clear(String code) => reasons.remove(code);

    if (temp != null && temp >= temperatureThreshold) {
      fire('temp', 'Critical temperature',
          '${temp.toStringAsFixed(1)} >= ${temperatureThreshold.toStringAsFixed(0)}');
    } else if (temp != null) {
      clear('temp');
    }

    if (vib != null && vib >= vibrationThreshold) {
      fire('vibration', 'Dangerous vibration',
          '${vib.toStringAsFixed(2)} >= ${vibrationThreshold.toStringAsFixed(2)} mm/s');
    } else if (vib != null) {
      clear('vibration');
    }

    if (power != null && power >= powerThreshold) {
      fire('power', 'Power overload',
          '${power.toStringAsFixed(1)} >= ${powerThreshold.toStringAsFixed(0)} kW');
    } else if (power != null) {
      clear('power');
    }

    if (statusHint == 'fault' || statusHint == 'broken' ||
        statusHint == 'down' || statusHint == 'emergency') {
      fire('status', 'Machine fault reported',
          'Telemetry reports status="$statusHint".');
    } else if (statusHint.isNotEmpty) {
      clear('status');
    }

    if (rul > 0 && rul <= AppState._kCriticalRulCycles) {
      fire('rul', 'Critical RUL',
          'Remaining useful life: ${rul.toStringAsFixed(0)} cycles.');
    } else if (rul > AppState._kCriticalRulCycles) {
      clear('rul');
    }
  }

  void _recordStatusTransitionsIfAny() {
    bool anyChange = false;
    for (final m in _machines) {
      final next = resolvedStatusFor(m.id);
      final prev = _lastResolvedStatus[m.id];
      if (prev == next) continue;

      _lastResolvedStatus[m.id] = next;
      anyChange = true;

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
    }
  }

  void _startStatusSweep() {
    _statusSweepTimer?.cancel();
    _statusSweepTimer = Timer.periodic(const Duration(seconds: 15), (_) {
      _recordStatusTransitionsIfAny();
      _notify();
    });
  }
}
