part of 'state.dart';

extension AppStateReports on AppState {
  // ── Per-machine Firestore subscriptions ───────────────────────────────────

  void subscribeToMachine(String machineId) {
    if (!_firebaseEnabled || _firebase == null) return;
    if (_reportsSubscriptions.containsKey(machineId)) return;

    _reportsSubscriptions[machineId] =
        _firebase!.watchReports(machineId).listen((remoteReports) {
      _reports.removeWhere((r) => r.machineId == machineId);
      _reports.addAll(remoteReports);
      _reports.sort((a, b) => b.createdAt.compareTo(a.createdAt));
      _notify();
    }, onError: (_) {
      // Permission denied or offline — keep local cache.
    });

    _historySubscriptions[machineId] =
        _firebase!.watchHistory(machineId).listen((events) {
      _history.removeWhere(
        (e) => e.machineId == machineId && e.id.startsWith('fs-'),
      );
      for (final raw in events) {
        _history.add(_historyFromFirestore(machineId, raw));
      }
      _history.sort((a, b) => b.time.compareTo(a.time));
      _notify();
    }, onError: (_) {/* non-fatal */});
  }

  void unsubscribeFromMachine(String machineId) {
    _reportsSubscriptions.remove(machineId)?.cancel();
    _historySubscriptions.remove(machineId)?.cancel();
  }

  // ── Queries ────────────────────────────────────────────────────────────────

  List<HistoryLog> historyForMachine(String machineId) {
    return _history.where((e) => e.machineId == machineId).toList();
  }

  List<MachineReport> reportsForMachine(String machineId) {
    final list = _reports
        .where((r) => r.machineId == machineId)
        .toList(growable: false);
    return list..sort((a, b) => b.createdAt.compareTo(a.createdAt));
  }

  // ── Report mutations ───────────────────────────────────────────────────────

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
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertReport(report).catchError((_) {});
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
    _recordStatusTransitionsIfAny();
    _notify();
    return report;
  }

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
    _notify();
  }

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
    _notify();
  }

  // ── Firestore history conversion ───────────────────────────────────────────

  HistoryLog _historyFromFirestore(
      String machineId, Map<String, dynamic> raw) {
    final type = (raw['type'] ?? '').toString();
    final kind = HistoryKind.values.firstWhere(
      (k) => k.name == type,
      orElse: () => HistoryKind.status,
    );
    final severity = raw['severity']?.toString();
    final author = raw['createdByName']?.toString();

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
      try {
        // ignore: avoid_dynamic_calls
        time = (ts as dynamic).toDate() as DateTime;
      } catch (_) {/* keep now */}
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

  // ── Shared history helper ──────────────────────────────────────────────────

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
}
