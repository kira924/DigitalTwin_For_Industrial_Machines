import 'dart:math';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../data/sensor_metadata.dart';
import '../models/machine.dart';
import '../models/machine_report.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';
import '../widgets/form_sheets.dart';

/// Dedicated, focused page for a single machine.
///
/// Reads telemetry from [AppState] which the dashboard's MQTT listener
/// keeps fresh — this screen does not own its own subscription. That
/// decision keeps the data path single-sourced and avoids a second
/// connection attempt to the broker.
///
/// Sections (top-to-bottom):
///   1. Header (status, location, last-seen freshness)
///   2. KPI row (cycle, RUL, health)
///   3. RUL prediction card (if telemetry present)
///   4. Live telemetry grid
///   5. AI root cause chart (if telemetry present)
///   6. History timeline (per-machine)
///   7. Reports list with "Add report" CTA (engineer/admin only)
///   8. Action buttons (start/stop/maintenance/issue) — engineer/admin only
class MachineDetailScreen extends StatefulWidget {
  const MachineDetailScreen({super.key, required this.machineId});

  final String machineId;

  @override
  State<MachineDetailScreen> createState() => _MachineDetailScreenState();
}

class _MachineDetailScreenState extends State<MachineDetailScreen> {
  @override
  void initState() {
    super.initState();
    // Subscribe to live Firestore reports + history for this specific
    // machine. The subscriptions stream into AppState's local lists so
    // [reportsForMachine] / [historyForMachine] return server-fresh data
    // while we're on this screen. We unsubscribe in dispose.
    //
    // Done in a post-frame callback so context is safe to read here.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) return;
      context.read<AppState>().subscribeToMachine(widget.machineId);
    });
  }

  @override
  void dispose() {
    // Cancel subscriptions when leaving the screen so we don't leak
    // streams. Safe to call even if subscribe was a no-op
    // (firebaseEnabled was false or subscriptions never started).
    context.read<AppState>().unsubscribeFromMachine(widget.machineId);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final perms = app.permissions;

    // Resolve the machine. If it was deleted while we were on this screen,
    // pop back to the previous route.
    final machine = app.machines
        .where((m) => m.id == widget.machineId)
        .cast<Machine?>()
        .firstWhere((_) => true, orElse: () => null);

    if (machine == null) {
      return Scaffold(
        backgroundColor: palette.canvas,
        appBar: appHeader(title: 'Machine not found'),
        body: const Center(
          child: EmptyView(
            icon: Icons.error_outline_rounded,
            title: 'Machine no longer available',
            description:
                'It may have been removed. Go back to the machines list.',
          ),
        ),
      );
    }

    final payload = app.latestTelemetryFor(machine.id);
    final lastSeen = app.lastSeenFor(machine.id);

    final stamp = lastSeen ?? machine.lastUpdated;
    final dataState =
        payload == null ? DataState.awaiting : dataStateFromTimestamp(stamp);

    // Always show the same 4 summary tiles in the Live Telemetry grid so
    // the layout stays consistent whether the machine is awaiting data or
    // actively receiving MQTT telemetry. Live payload values take priority
    // over the machine's stored seed values.
    final sensors = <MapEntry<String, double>>[
      MapEntry(
        'Temp',
        payload?.temperature ??
            payload?.currentSensorReadings['s_2'] ??
            machine.temperature,
      ),
      MapEntry(
        'Speed',
        payload?.speed ??
            payload?.currentSensorReadings['s_8'] ??
            machine.speed,
      ),
      MapEntry(
        'Pressure',
        payload?.currentSensorReadings['s_7'] ??
            payload?.currentSensorReadings['s_11'] ??
            machine.pressure,
      ),
      MapEntry(
        'Efficiency',
        payload?.healthScore ?? machine.efficiency.toDouble(),
      ),
    ];

    final causes = payload != null
        ? (payload.aiRootCauses.entries.toList()
              ..sort((a, b) => b.value.compareTo(a.value)))
            .take(4)
            .toList()
        : const <MapEntry<String, double>>[];

    final rul = payload?.predictedRul;
    final health = payload?.healthScore;
    final rulColor = rul == null
        ? DTTokens.statusOffline
        : rul < 50
            ? DTTokens.statusCritical
            : rul < 100
                ? DTTokens.statusWarning
                : DTTokens.statusHealthy;

    final history = app.historyForMachine(machine.id);
    final reports = app.reportsForMachine(machine.id);

    return Scaffold(
      backgroundColor: palette.canvas,
      appBar: appHeader(
        title: machine.name,
        subtitle: '${machine.code} · ${machine.location}',
        leading: _BackButton(onTap: () => Navigator.of(context).maybePop()),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
        children: [
          // 1. Header card
          _HeaderCard(
            machine: machine,
            dataState: dataState,
            lastSeenStamp: stamp,
            // Phase 3.5: badge shows the auto-derived status, not the
            // (now-unused) machine.status field.
            resolvedStatus: app.resolvedStatusFor(machine.id),
          ),
          const SizedBox(height: DTTokens.cardGap),

          // 2. KPI row
          Row(
            children: [
              Expanded(
                child: KpiCard(
                  label: 'Cycle',
                  value: payload != null ? '${payload.currentCycle}' : '—',
                ),
              ),
              const SizedBox(width: DTTokens.cardGap),
              Expanded(
                child: KpiCard(
                  // Phase 3.5: renamed from "RUL". The cycle-count
                  // value is unchanged; only the label moves to a
                  // user-facing term.
                  label: 'RUL.',
                  value: rul != null ? rul.toStringAsFixed(0) : '—',
                  unit: rul != null ? ' cyc' : null,
                  valueColor: rulColor,
                ),
              ),
              const SizedBox(width: DTTokens.cardGap),
              Expanded(
                child: KpiCard(
                  label: 'Health',
                  value: health != null
                      ? health.toStringAsFixed(1)
                      : '${machine.efficiency}',
                  unit: '%',
                  valueColor: (health ?? machine.efficiency.toDouble()) >= 70
                      ? DTTokens.statusHealthy
                      : (health ?? machine.efficiency.toDouble()) >= 30
                          ? DTTokens.statusWarning
                          : DTTokens.statusCritical,
                ),
              ),
            ],
          ),
          const SizedBox(height: DTTokens.space20),

          // Phase 3.5: this slot used to host the duplicate Predictive
          // Maintenance card (large "remaining cycles" block). It's
          // been removed; the KPI tile above carries the Predictive
          // Maintenance value. The slot is now the AI Root Causes
          // (SHAP Top 3) chart, sourced from `payload.aiRootCauses`.
          const SectionHeader(title: 'Key Risk Factors'),
          DtCard(
            child: SizedBox(
              height: 220,
              child: causes.isEmpty
                  ? const _RootCausesEmpty()
                  : _ShapBars(causes: causes),
            ),
          ),
          const SizedBox(height: DTTokens.space20),

          // 4. Telemetry grid
          Row(
            children: [
              const Expanded(child: SectionHeader(title: 'Live Telemetry')),
              TextButton.icon(
                onPressed: () => _showAllSensorsSheet(context, machine.id),
                icon: const Icon(Icons.view_list_rounded, size: 16),
                label: const Text('More sensors'),
                style: TextButton.styleFrom(
                  foregroundColor: DTTokens.accentLive,
                ),
              ),
            ],
          ),
          if (sensors.isEmpty)
            const EmptyView(
              icon: Icons.sensors_off_rounded,
              title: 'Waiting for data',
              description:
                  'Telemetry from this machine has not arrived yet.',
            )
          else
            GridView.builder(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemCount: sensors.length,
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                mainAxisSpacing: DTTokens.cardGap,
                crossAxisSpacing: DTTokens.cardGap,
                childAspectRatio: 1.5,
              ),
              itemBuilder: (ctx, i) {
                final entry = sensors[i];
                return _SensorCard(label: entry.key, value: entry.value);
              },
            ),
          const SizedBox(height: DTTokens.space20),

          // (AI Root Causes was moved above the Live Telemetry section
          // as part of Phase 3.5.)

          // 6. History timeline (per-machine)
          SectionHeader(
            title: 'Machine History',
            trailingText: history.isEmpty ? null : '${history.length} event${history.length == 1 ? '' : 's'}',
            onTrailingTap: history.isEmpty ? null : () {},
          ),
          if (history.isEmpty)
            const EmptyView(
              icon: Icons.history_rounded,
              title: 'No history yet',
              description:
                  'Status changes, faults, maintenance and reports will appear here.',
            )
          else
            _HistoryTimeline(events: history),
          const SizedBox(height: DTTokens.space20),

          // 7. Reports
          SectionHeader(
            title: 'Reports',
            trailingText: reports.isEmpty ? null : '${reports.length}',
            onTrailingTap: reports.isEmpty ? null : () {},
          ),
          if (reports.isEmpty)
            const EmptyView(
              icon: Icons.fact_check_outlined,
              title: 'No reports yet',
              description:
                  'Engineers can file reports against this machine. They will appear here and in the history timeline.',
            )
          else
            ...reports.map(
              (r) => Padding(
                padding: const EdgeInsets.only(bottom: DTTokens.cardGap),
                child: _ReportCard(
                  report: r,
                  canEdit: perms.canEditOwnReport(r.authorId, app.currentUser.id),
                  canDelete: perms.canDeleteReports,
                  onEdit: () => _openReportSheet(context, machine, r),
                  onDelete: () => _confirmDeleteReport(context, r),
                ),
              ),
            ),
          if (perms.canWriteReports) ...[
            const SizedBox(height: DTTokens.space8),
            DtActionButton(
              label: 'Write a report',
              color: DTTokens.accentPrimary,
              icon: Icons.note_add_outlined,
              onTap: () => _openReportSheet(context, machine, null),
            ),
          ],
          const SizedBox(height: DTTokens.space20),

          // 8. Actions — engineer/admin only
          if (perms.canActOnMachine) ...[
            const SectionHeader(title: 'Actions'),
            DtActionButton(
              label: 'Report issue',
              color: DTTokens.statusCritical,
              icon: Icons.report_outlined,
              onTap: () => context.read<AppState>().reportIssue(machine.id),
            ),
            const SizedBox(height: DTTokens.space8),
            DtActionButton(
              label: 'Schedule maintenance',
              color: DTTokens.statusWarning,
              icon: Icons.build_outlined,
              onTap: () =>
                  context.read<AppState>().requestMaintenance(machine.id),
            ),
            // Phase 3.5: manual Start/Stop buttons were removed.
            // Machine status is now derived automatically from
            // telemetry freshness + report existence
            // (see MachineStatusResolver).
            if (perms.canEditMachines) ...[
              const SizedBox(height: DTTokens.space8),
              DtActionButton(
                label: 'Edit machine',
                color: DTTokens.accentPrimary,
                icon: Icons.edit_outlined,
                onTap: () => showMachineFormSheet(context, machine: machine),
              ),
            ],
          ] else
            // Viewer message — explicit so it doesn't feel like missing UI.
            DtCard(
              child: Row(
                children: [
                  Icon(Icons.visibility_outlined,
                      size: 18, color: palette.textTertiary),
                  const SizedBox(width: DTTokens.space12),
                  Expanded(
                    child: Text(
                      'You have read-only access to this machine.',
                      style: DTTokens.caption(palette.textSecondary),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Future<void> _openReportSheet(
    BuildContext context,
    Machine machine,
    MachineReport? existing,
  ) async {
    await showReportSheet(context, machine: machine, existing: existing);
  }

  Future<void> _confirmDeleteReport(
    BuildContext context,
    MachineReport report,
  ) async {
    final palette = DTPalette.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: palette.surface,
        title: const Text('Delete report?'),
        content: Text(
          '“${report.title}” will be removed from this machine.',
          style: DTTokens.body(palette.textSecondary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: DTTokens.statusCritical,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok == true && context.mounted) {
      context.read<AppState>().deleteReport(report.id);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Header card — status badge, freshness indicator, location summary
// ─────────────────────────────────────────────────────────────────────────────

class _HeaderCard extends StatelessWidget {
  const _HeaderCard({
    required this.machine,
    required this.dataState,
    required this.lastSeenStamp,
    required this.resolvedStatus,
  });

  final Machine machine;
  final DataState dataState;
  final DateTime lastSeenStamp;

  /// Phase 3.5: status badge shows the auto-derived status from
  /// `MachineStatusResolver`, not the manual `machine.status` field
  /// (which is no longer mutated by the UI).
  final MachineStatus resolvedStatus;

  Color _statusColor(MachineStatus s) {
    switch (s) {
      case MachineStatus.running:
        return DTTokens.statusHealthy;
      case MachineStatus.maintenance:
        return DTTokens.statusWarning;
      case MachineStatus.fault:
        return DTTokens.statusCritical;
      case MachineStatus.stopped:
        return DTTokens.statusOffline;
      case MachineStatus.offline:
        return DTTokens.statusOffline;
    }
  }

  String _statusLabel(MachineStatus s) {
    switch (s) {
      case MachineStatus.running:
        return 'RUNNING';
      case MachineStatus.maintenance:
        return 'UNDER MAINTENANCE';
      case MachineStatus.fault:
        return 'FAULT';
      case MachineStatus.stopped:
        return 'OFFLINE';
      case MachineStatus.offline:
        return 'OFFLINE';
    }
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final color = _statusColor(resolvedStatus);

    return DtCard(
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                StatusBadge(label: _statusLabel(resolvedStatus), color: color),
                const SizedBox(height: DTTokens.space12),
                Text(
                  machine.type,
                  style: DTTokens.body(palette.textPrimary).copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  machine.location,
                  style: DTTokens.caption(palette.textTertiary),
                ),
                if (machine.activeAlert != null) ...[
                  const SizedBox(height: DTTokens.space8),
                  Row(
                    children: [
                      const Icon(
                        Icons.warning_amber_rounded,
                        color: DTTokens.statusWarning,
                        size: 14,
                      ),
                      const SizedBox(width: 4),
                      Expanded(
                        child: Text(
                          machine.activeAlert!,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: DTTokens.caption(DTTokens.statusWarning)
                              .copyWith(fontWeight: FontWeight.w600),
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Row(
                children: [
                  LiveDot(state: dataState, size: 6),
                  const SizedBox(width: 6),
                  Text(
                    dataState.label,
                    style: DTTokens.caption(dataState.color)
                        .copyWith(fontWeight: FontWeight.w700),
                  ),
                ],
              ),
              const SizedBox(height: 2),
              Text(
                formatAge(lastSeenStamp),
                style: DTTokens.caption(palette.textTertiary),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Back button (matches the polished header style)
// ─────────────────────────────────────────────────────────────────────────────

class _BackButton extends StatelessWidget {
  const _BackButton({required this.onTap});
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(DTTokens.radiusSm),
      child: Container(
        width: 36,
        height: 36,
        decoration: BoxDecoration(
          color: palette.surface,
          border: Border.all(color: palette.borderSubtle),
          borderRadius: BorderRadius.circular(DTTokens.radiusSm),
        ),
        child: Icon(
          Icons.arrow_back_rounded,
          size: 18,
          color: palette.textSecondary,
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Telemetry sensor card (no sparkline — kept compact for the detail page)
// ─────────────────────────────────────────────────────────────────────────────

class _SensorCard extends StatelessWidget {
  const _SensorCard({required this.label, required this.value});
  final String label;
  final double value;

  // Units for the four fixed summary tiles whose labels are not sensor keys.
  static const Map<String, String> _summaryUnits = <String, String>{
    'Temp':       '°R',
    'Speed':      'rpm',
    'Pressure':   'psia',
    'Efficiency': '%',
  };

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    // Resolve sensor metadata — fall back gracefully when key is unknown.
    final meta = kAllSensors.cast<SensorMetadata?>()
        .firstWhere((s) => s?.key == label, orElse: () => null);
    final symbol = meta?.symbol ?? label.toUpperCase();
    final fullName = meta?.fullName ?? label;
    final unit = (meta != null && meta.unit.isNotEmpty && meta.unit != '—')
        ? meta.unit
        : (_summaryUnits[label] ?? '');

    return DtCard(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Symbol label
          Text(
            symbol,
            style: DTTokens.label(palette.textTertiary)
                .copyWith(fontSize: 10, letterSpacing: 0.8),
          ),
          // Value + unit
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Flexible(
                child: Text(
                  value.toStringAsFixed(1),
                  style: DTTokens.kpi(DTTokens.accentLive, size: 22),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (unit.isNotEmpty) ...[
                const SizedBox(width: 3),
                Text(unit, style: DTTokens.caption(palette.textTertiary)),
              ],
            ],
          ),
          // Full sensor name
          Text(
            fullName,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: DTTokens.caption(palette.textTertiary)
                .copyWith(fontSize: 10),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SHAP root-cause bars (compact)
// ─────────────────────────────────────────────────────────────────────────────

class _ShapBars extends StatelessWidget {
  const _ShapBars({required this.causes});
  final List<MapEntry<String, double>> causes;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final maxY = causes.map((e) => e.value).reduce(max) * 1.25;
    return BarChart(
      BarChartData(
        alignment: BarChartAlignment.spaceAround,
        maxY: maxY,
        minY: 0,
        barGroups: List.generate(causes.length, (i) {
          return BarChartGroupData(
            x: i,
            barRods: [
              BarChartRodData(
                toY: causes[i].value,
                width: 28,
                color: DTTokens.accentPrimary,
                borderRadius: const BorderRadius.vertical(
                  top: Radius.circular(4),
                ),
              ),
            ],
          );
        }),
        titlesData: FlTitlesData(
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              getTitlesWidget: (v, m) => Text(
                v.toInt().toString(),
                style: DTTokens.caption(palette.textTertiary),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 28,
              getTitlesWidget: (v, m) {
                final idx = v.toInt();
                if (idx < 0 || idx >= causes.length) return const SizedBox();
                final rawKey = causes[idx].key;
                final symbol = kAllSensors
                    .cast<SensorMetadata?>()
                    .firstWhere((s) => s?.key == rawKey, orElse: () => null)
                    ?.symbol ?? rawKey;
                return Padding(
                  padding: const EdgeInsets.only(top: 6),
                  child: Text(
                    symbol,
                    style: DTTokens.caption(palette.textSecondary),
                  ),
                );
              },
            ),
          ),
        ),
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          getDrawingHorizontalLine: (_) =>
              FlLine(color: palette.borderSubtle, strokeWidth: 1),
        ),
        borderData: FlBorderData(show: false),
        barTouchData: BarTouchData(
          touchTooltipData: BarTouchTooltipData(
            getTooltipItem: (group, _, rod, __) => BarTooltipItem(
              rod.toY.toStringAsFixed(2),
              DTTokens.mono(palette.textPrimary, size: 12),
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// History timeline — vertical rail with day grouping
// ─────────────────────────────────────────────────────────────────────────────

class _HistoryTimeline extends StatelessWidget {
  const _HistoryTimeline({required this.events});
  final List<HistoryLog> events;

  Map<String, List<HistoryLog>> _grouped() {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final yesterday = today.subtract(const Duration(days: 1));
    final groups = <String, List<HistoryLog>>{};
    for (final e in events) {
      final d = DateTime(e.time.year, e.time.month, e.time.day);
      String key;
      if (d == today) {
        key = 'Today';
      } else if (d == yesterday) {
        key = 'Yesterday';
      } else {
        key = '${e.time.day}/${e.time.month}/${e.time.year}';
      }
      groups.putIfAbsent(key, () => []).add(e);
    }
    return groups;
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final groups = _grouped();
    final children = <Widget>[];
    var first = true;
    for (final entry in groups.entries) {
      if (!first) children.add(const SizedBox(height: DTTokens.space12));
      first = false;
      children.add(
        Padding(
          padding: const EdgeInsets.only(bottom: 6, left: 4),
          child: Text(
            entry.key.toUpperCase(),
            style: DTTokens.label(palette.textTertiary),
          ),
        ),
      );
      for (var i = 0; i < entry.value.length; i++) {
        final e = entry.value[i];
        final isLast = i == entry.value.length - 1;
        children.add(_TimelineRow(event: e, isLast: isLast));
      }
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: children,
    );
  }
}

class _TimelineRow extends StatelessWidget {
  const _TimelineRow({required this.event, required this.isLast});
  final HistoryLog event;
  final bool isLast;

  String _formatTime(DateTime t) {
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    return '$hh:$mm';
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Rail (dot + connector line)
          SizedBox(
            width: 28,
            child: Column(
              children: [
                const SizedBox(height: 14),
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: event.color,
                    shape: BoxShape.circle,
                    border: Border.all(color: palette.canvas, width: 2),
                  ),
                ),
                if (!isLast)
                  Expanded(
                    child: Container(
                      width: 2,
                      margin: const EdgeInsets.symmetric(vertical: 4),
                      color: palette.borderSubtle,
                    ),
                  ),
              ],
            ),
          ),
          // Card
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: DTTokens.space8),
              child: DtCard(
                padding: const EdgeInsets.all(DTTokens.space12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(event.icon, size: 14, color: event.color),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            event.title,
                            style: DTTokens.body(palette.textPrimary).copyWith(
                              fontWeight: FontWeight.w700,
                            ),
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ),
                        if (event.severity != null)
                          StatusBadge(
                            label: event.severity!,
                            color: event.color,
                          ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      event.subtitle,
                      style: DTTokens.body(palette.textSecondary),
                    ),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Text(
                          _formatTime(event.time),
                          style: DTTokens.mono(palette.textTertiary, size: 11),
                        ),
                        if (event.author != null) ...[
                          Text(
                            '  ·  ',
                            style: DTTokens.caption(palette.textTertiary),
                          ),
                          Flexible(
                            child: Text(
                              event.author!,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: DTTokens.caption(palette.textTertiary),
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Report card
// ─────────────────────────────────────────────────────────────────────────────

class _ReportCard extends StatelessWidget {
  const _ReportCard({
    required this.report,
    required this.canEdit,
    required this.canDelete,
    required this.onEdit,
    required this.onDelete,
  });

  final MachineReport report;
  final bool canEdit;
  final bool canDelete;
  final VoidCallback onEdit;
  final VoidCallback onDelete;

  String _formatTime(DateTime t) {
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    return '${t.day}/${t.month} · $hh:$mm';
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              StatusBadge(
                label: report.severity.label,
                color: report.severity.color,
              ),
              const SizedBox(width: 6),
              StatusBadge(
                label: report.status.label.toUpperCase(),
                color: report.status.color,
              ),
              const Spacer(),
              if (canEdit)
                IconButton(
                  visualDensity: VisualDensity.compact,
                  onPressed: onEdit,
                  icon: Icon(Icons.edit_outlined,
                      size: 16, color: palette.textSecondary),
                  tooltip: 'Edit',
                ),
              if (canDelete)
                IconButton(
                  visualDensity: VisualDensity.compact,
                  onPressed: onDelete,
                  icon: const Icon(Icons.delete_outline_rounded,
                      size: 16, color: DTTokens.statusCritical),
                  tooltip: 'Delete',
                ),
            ],
          ),
          const SizedBox(height: DTTokens.space8),
          Text(
            report.title,
            style: DTTokens.h3(palette.textPrimary),
          ),
          const SizedBox(height: 4),
          Text(
            report.description,
            style: DTTokens.body(palette.textSecondary),
          ),
          if (report.recommendedAction.trim().isNotEmpty) ...[
            const SizedBox(height: DTTokens.space8),
            Container(
              padding: const EdgeInsets.all(DTTokens.space12),
              decoration: BoxDecoration(
                color: palette.surfaceElevated,
                borderRadius: BorderRadius.circular(DTTokens.radiusSm),
                border: Border.all(color: palette.borderSubtle),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Icon(Icons.lightbulb_outline,
                      size: 14, color: palette.textTertiary),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      report.recommendedAction,
                      style: DTTokens.caption(palette.textSecondary)
                          .copyWith(height: 1.4),
                    ),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(height: DTTokens.space8),
          Row(
            children: [
              Icon(Icons.person_outline,
                  size: 12, color: palette.textTertiary),
              const SizedBox(width: 4),
              Flexible(
                child: Text(
                  report.authorName.isEmpty ? 'Unknown' : report.authorName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: DTTokens.caption(palette.textTertiary),
                ),
              ),
              Text(
                '  ·  ',
                style: DTTokens.caption(palette.textTertiary),
              ),
              Text(
                _formatTime(report.createdAt),
                style: DTTokens.mono(palette.textTertiary, size: 11),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AI Root Causes empty-state placeholder (Phase 3.5)
// Shown when the current payload has no aiRootCauses entries (e.g. before
// the first MQTT message arrives, or when running against a publisher
// that doesn't include them).
// ─────────────────────────────────────────────────────────────────────────────

class _RootCausesEmpty extends StatelessWidget {
  const _RootCausesEmpty();

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.auto_graph_rounded,
            size: 28,
            color: palette.textTertiary,
          ),
          const SizedBox(height: 8),
          Text(
            'Awaiting AI root-cause telemetry',
            style: DTTokens.body(palette.textSecondary)
                .copyWith(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: 4),
          Text(
            'Top contributing sensors will appear here once the model emits a SHAP attribution.',
            textAlign: TextAlign.center,
            style: DTTokens.caption(palette.textTertiary)
                .copyWith(height: 1.4),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// All-sensors sheet
// Shows all 14 engine sensor channels from kAllSensors with descaled,
// human-readable physical values derived from live MQTT telemetry.
// The sheet subscribes to AppState so values update in real time as new
// MQTT messages arrive without needing to close and reopen the sheet.
// ─────────────────────────────────────────────────────────────────────────────

void _showAllSensorsSheet(BuildContext context, String machineId) {
  // Capture the AppState instance before opening the sheet. The modal
  // bottom sheet runs in its own route with an independent widget tree, so
  // we re-inject it via ChangeNotifierProvider.value to keep the Consumer
  // inside the sheet reactive to incoming telemetry.
  final appState = context.read<AppState>();

  showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF07111F),
    shape: const RoundedRectangleBorder(
      borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
    ),
    builder: (ctx) {
      return ChangeNotifierProvider<AppState>.value(
        value: appState,
        child: _AllSensorsSheetContent(machineId: machineId),
      );
    },
  );
}

// Stateless widget for the bottom-sheet body. The DraggableScrollableSheet is
// built once; only the Consumer subtree rebuilds when telemetry updates arrive.
class _AllSensorsSheetContent extends StatelessWidget {
  const _AllSensorsSheetContent({required this.machineId});
  final String machineId;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);

    return DraggableScrollableSheet(
      initialChildSize: 0.78,
      minChildSize: 0.4,
      maxChildSize: 0.95,
      expand: false,
      builder: (_, scrollController) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          child: Consumer<AppState>(
            builder: (_, app, __) {
              final payload = app.latestTelemetryFor(machineId);
              final readings =
                  payload?.currentSensorReadings ?? const <String, double>{};
              final reportingCount =
                  kAllSensors.where((s) => readings.containsKey(s.key)).length;

              return Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Drag handle
                  Center(
                    child: Container(
                      width: 36,
                      height: 4,
                      decoration: BoxDecoration(
                        color: palette.borderSubtle,
                        borderRadius: BorderRadius.circular(2),
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text('All sensors', style: DTTokens.h3(palette.textPrimary)),
                  const SizedBox(height: 4),
                  Text(
                    payload == null
                        ? 'No live telemetry yet — values appear once data arrives.'
                        : '$reportingCount of ${kAllSensors.length} sensors reporting.',
                    style: DTTokens.caption(palette.textTertiary),
                  ),
                  const SizedBox(height: 12),
                  Expanded(
                    child: ListView.separated(
                      controller: scrollController,
                      itemCount: kAllSensors.length,
                      separatorBuilder: (_, __) =>
                          const SizedBox(height: DTTokens.cardGap),
                      itemBuilder: (_, i) {
                        final s = kAllSensors[i];
                        return _SensorRow(
                          meta: s,
                          physicalValue: readings[s.key],
                        );
                      },
                    ),
                  ),
                ],
              );
            },
          ),
        );
      },
    );
  }
}

class _SensorRow extends StatelessWidget {
  const _SensorRow({required this.meta, required this.physicalValue});
  final SensorMetadata meta;

  // Raw sensor value received directly from the MQTT telemetry payload.
  // Already in physical engineering units — no scaling or math is applied.
  // Null when no telemetry has been received yet for this sensor channel.
  final double? physicalValue;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final hasValue = physicalValue != null;
    final showUnit = meta.unit.isNotEmpty && meta.unit != '—';

    final String displayValue = hasValue
        ? physicalValue!.toStringAsFixed(2)
        : '—';

    return DtCard(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 12),
      child: Row(
        children: [
          // Symbol (primary identifier) + raw key (secondary)
          SizedBox(
            width: 68,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  meta.symbol,
                  style: DTTokens.mono(palette.textPrimary, size: 13)
                      .copyWith(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 2),
                Text(
                  meta.key,
                  style: DTTokens.caption(palette.textTertiary)
                      .copyWith(fontSize: 10),
                ),
              ],
            ),
          ),
          const SizedBox(width: DTTokens.space12),
          // Full sensor name
          Expanded(
            child: Text(
              meta.fullName,
              style: DTTokens.body(palette.textSecondary).copyWith(
                fontSize: 12,
                height: 1.3,
              ),
            ),
          ),
          const SizedBox(width: DTTokens.space12),
          // Descaled physical value + unit
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Text(
                displayValue,
                style: DTTokens.mono(
                  hasValue ? DTTokens.accentLive : palette.textTertiary,
                  size: 15,
                  weight: FontWeight.w700,
                ),
              ),
              if (showUnit && hasValue)
                Text(
                  meta.unit,
                  style: DTTokens.caption(palette.textTertiary)
                      .copyWith(fontSize: 10),
                ),
            ],
          ),
        ],
      ),
    );
  }

}
