import 'package:flutter/material.dart';

import '../models/app_alert.dart';
import '../models/history_log.dart';
import '../models/machine_report.dart';
import '../theme/dt_colors.dart';

typedef SeedLogData = ({
  List<HistoryLog> history,
  List<AppAlert> alerts,
  List<MachineReport> reports,
});

SeedLogData buildSeedLogs() {
  final now = DateTime.now();

  final history = <HistoryLog>[
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
  ];

  final alerts = <AppAlert>[
    AppAlert(
      id: 'a1',
      machineId: 'wld-06',
      title: 'Critical fault',
      message: 'Welding Robot exceeded vibration threshold.',
      time: now.subtract(const Duration(minutes: 6)),
      color: DT.red,
    ),
  ];

  final reports = <MachineReport>[
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
  ];

  return (history: history, alerts: alerts, reports: reports);
}
