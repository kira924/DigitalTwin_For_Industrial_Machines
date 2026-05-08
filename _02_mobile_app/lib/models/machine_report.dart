import 'package:flutter/material.dart';

import '../theme/dt_tokens.dart';

/// Severity tier for an engineer's machine report.
/// Lower index = lower urgency.
enum ReportSeverity { info, low, medium, high, critical }

extension ReportSeverityX on ReportSeverity {
  String get label {
    switch (this) {
      case ReportSeverity.info:
        return 'INFO';
      case ReportSeverity.low:
        return 'LOW';
      case ReportSeverity.medium:
        return 'MEDIUM';
      case ReportSeverity.high:
        return 'HIGH';
      case ReportSeverity.critical:
        return 'CRITICAL';
    }
  }

  Color get color {
    switch (this) {
      case ReportSeverity.info:
        return DTTokens.accentPrimary;
      case ReportSeverity.low:
        return DTTokens.statusHealthy;
      case ReportSeverity.medium:
        return DTTokens.statusWarning;
      case ReportSeverity.high:
        return DTTokens.statusWarning;
      case ReportSeverity.critical:
        return DTTokens.statusCritical;
    }
  }
}

/// Lifecycle status of a report.
enum ReportStatus { open, inProgress, resolved }

extension ReportStatusX on ReportStatus {
  String get label {
    switch (this) {
      case ReportStatus.open:
        return 'Open';
      case ReportStatus.inProgress:
        return 'In progress';
      case ReportStatus.resolved:
        return 'Resolved';
    }
  }

  Color get color {
    switch (this) {
      case ReportStatus.open:
        return DTTokens.statusCritical;
      case ReportStatus.inProgress:
        return DTTokens.statusWarning;
      case ReportStatus.resolved:
        return DTTokens.statusHealthy;
    }
  }
}

ReportSeverity reportSeverityFromName(String? raw) {
  return ReportSeverity.values.firstWhere(
    (v) => v.name == (raw ?? '').toLowerCase(),
    orElse: () => ReportSeverity.medium,
  );
}

ReportStatus reportStatusFromName(String? raw) {
  return ReportStatus.values.firstWhere(
    (v) => v.name == (raw ?? '').toLowerCase(),
    orElse: () => ReportStatus.open,
  );
}

/// An engineer's report against a specific machine. Visible in the
/// machine's history timeline and surfaced separately under "Reports"
/// on the machine detail page.
class MachineReport {
  final String id;
  final String machineId;
  final String title;
  final String description;
  final String recommendedAction;
  final ReportSeverity severity;
  final ReportStatus status;
  final String authorId;
  final String authorName;
  final DateTime createdAt;

  const MachineReport({
    required this.id,
    required this.machineId,
    required this.title,
    required this.description,
    required this.recommendedAction,
    required this.severity,
    required this.status,
    required this.authorId,
    required this.authorName,
    required this.createdAt,
  });

  MachineReport copyWith({
    String? title,
    String? description,
    String? recommendedAction,
    ReportSeverity? severity,
    ReportStatus? status,
  }) {
    return MachineReport(
      id: id,
      machineId: machineId,
      title: title ?? this.title,
      description: description ?? this.description,
      recommendedAction: recommendedAction ?? this.recommendedAction,
      severity: severity ?? this.severity,
      status: status ?? this.status,
      authorId: authorId,
      authorName: authorName,
      createdAt: createdAt,
    );
  }

  Map<String, dynamic> toMap() => <String, dynamic>{
        'id': id,
        'machineId': machineId,
        'title': title,
        'description': description,
        'recommendedAction': recommendedAction,
        'severity': severity.name,
        'status': status.name,
        'authorId': authorId,
        'authorName': authorName,
        'createdAt': createdAt.toIso8601String(),
      };

  factory MachineReport.fromMap(Map<String, dynamic> map) {
    return MachineReport(
      id: (map['id'] ?? '').toString(),
      machineId: (map['machineId'] ?? '').toString(),
      title: (map['title'] ?? '').toString(),
      description: (map['description'] ?? '').toString(),
      recommendedAction: (map['recommendedAction'] ?? '').toString(),
      severity: reportSeverityFromName(map['severity']?.toString()),
      status: reportStatusFromName(map['status']?.toString()),
      authorId: (map['authorId'] ?? '').toString(),
      authorName: (map['authorName'] ?? '').toString(),
      createdAt: DateTime.tryParse((map['createdAt'] ?? '').toString()) ??
          DateTime.now(),
    );
  }
}
