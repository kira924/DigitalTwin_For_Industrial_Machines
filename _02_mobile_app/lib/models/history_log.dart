import 'package:flutter/material.dart';

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
