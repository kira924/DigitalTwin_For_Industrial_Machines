import 'dart:async';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../models/engine_telemetry_payload.dart';
import '../models/machine.dart';
import '../services/mqtt_service.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';
import '../widgets/machine_image.dart';
import 'machine_detail_screen.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Top-level controller — owns the MQTT subscription. Logic untouched
// from the previous version; only the rendering layer was rebuilt.
// ─────────────────────────────────────────────────────────────────────────────

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  final MqttService _mqtt = MqttService();
  StreamSubscription<EngineTelemetryPayload>? _sub;
  final Map<int, EngineTelemetryPayload> _latestByEngine = {};
  final Map<int, Map<String, List<double>>> _historyByEngine = {};
  final Map<int, DateTime> _lastSeenByEngine = {};
  String _connectionState = 'Connecting…';

  @override
  void initState() {
    super.initState();
    _sub = _mqtt.telemetryStream.listen((payload) {
      if (!mounted) return;
      // Mirror the payload into AppState so other screens (notably the
      // dedicated machine detail page) can render the same realtime
      // values without owning a parallel MQTT subscription.
      // Riverpod providers handle fine-grained updates automatically.
      // ignore: use_build_context_synchronously
      context.read<AppState>().ingestTelemetry(payload);
    });
    _mqtt.connect().then((_) {
      if (!mounted) return;
      setState(() {
        _connectionState =
            _mqtt.isConnected ? 'Awaiting telemetry' : 'Disconnected';
      });
    });
  }

  @override
  void dispose() {
    _sub?.cancel();
    _mqtt.dispose();
    super.dispose();
  }

  int _engineId(Machine m, List<Machine> list) {
    final idx = list.indexWhere((x) => x.id == m.id);
    return idx >= 0 ? idx + 1 : 0;
  }

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    // Note: machine click now navigates via Navigator.push to the dedicated
    // MachineDetailScreen. The old inline detail branch was removed.
    return _FactoryHomeView(
      connectionState: _connectionState,
      latestByEngine: _latestByEngine,
      lastSeenByEngine: _lastSeenByEngine,
      engineIdForMachine: (m) => _engineId(m, app.machines),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory home view — the main dashboard
// ─────────────────────────────────────────────────────────────────────────────

class _FactoryHomeView extends StatefulWidget {
  const _FactoryHomeView({
    required this.connectionState,
    required this.latestByEngine,
    required this.lastSeenByEngine,
    required this.engineIdForMachine,
  });

  final String connectionState;
  final Map<int, EngineTelemetryPayload> latestByEngine;
  final Map<int, DateTime> lastSeenByEngine;
  final int Function(Machine) engineIdForMachine;

  @override
  State<_FactoryHomeView> createState() => _FactoryHomeViewState();
}

class _FactoryHomeViewState extends State<_FactoryHomeView> {
  String _query = '';

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final firstName = app.currentUser.name.split(' ').first;

    final filtered = _query.isEmpty
        ? app.machines
        : app.machines
            .where((m) =>
                m.name.toLowerCase().contains(_query.toLowerCase()) ||
                m.code.toLowerCase().contains(_query.toLowerCase()))
            .toList(growable: false);

    final atRisk = [...app.machines]
      ..sort((a, b) {
        final riskA = calculateRiskFromHealth(a.efficiency.toDouble());
        final riskB = calculateRiskFromHealth(b.efficiency.toDouble());
        return riskB.compareTo(riskA);
      });

    return Scaffold(
      backgroundColor: palette.canvas,
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _DashboardHeader(
              userName: firstName,
              alertCount: app.alerts.length,
              connectionState: widget.connectionState,
            ),
            Expanded(
              child: ListView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 90),
                children: [
                  // Search
                  _DtSearchBar(onChanged: (q) => setState(() => _query = q)),
                  const SizedBox(height: DTTokens.space20),

                  // ── KPI mini-cards + Top Risks (Phase 3.6 layout) ──────
                  // Sketch: Online / OEE / Alerts / Faults laid out 2x2
                  // on the left, Top Risks card on the right. On narrow
                  // screens the panels stack vertically.
                  LayoutBuilder(
                    builder: (ctx, c) {
                      // 600 dp covers tablets in both orientations; phones
                      // (≤428 dp) always get the stacked layout.
                      final wide = c.maxWidth >= 600;
                      final kpis = _DashboardMetricGrid(app: app);
                      final risks = _TopRisksCard(machines: atRisk);
                      if (wide) {
                        return Row(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Expanded(flex: 5, child: kpis),
                            const SizedBox(width: DTTokens.cardGap),
                            Expanded(flex: 4, child: risks),
                          ],
                        );
                      }
                      return Column(
                        children: [
                          kpis,
                          const SizedBox(height: DTTokens.cardGap),
                          risks,
                        ],
                      );
                    },
                  ),
                  const SizedBox(height: DTTokens.space20),

                  // OEE / performance trend card
                  _PerformanceTrendCard(app: app),
                  const SizedBox(height: DTTokens.space20),

                  // Machines section
                  SectionHeader(title: 'Machines (${filtered.length})'),
                  GridView.builder(
                    shrinkWrap: true,
                    physics: const NeverScrollableScrollPhysics(),
                    itemCount: filtered.length,
                    gridDelegate:
                        const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 2,
                      mainAxisSpacing: DTTokens.cardGap,
                      crossAxisSpacing: DTTokens.cardGap,
                      childAspectRatio: 0.78,
                    ),
                    itemBuilder: (ctx, i) {
                      final m = filtered[i];
                      final eid = widget.engineIdForMachine(m);
                      return _MachineGridCard(
                        machine: m,
                        payload: widget.latestByEngine[eid],
                        lastTelemetry: widget.lastSeenByEngine[eid],
                      );
                    },
                  ),
                  const SizedBox(height: DTTokens.space20),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Header
// ─────────────────────────────────────────────────────────────────────────────

class _DashboardHeader extends StatelessWidget {
  const _DashboardHeader({
    required this.userName,
    required this.alertCount,
    required this.connectionState,
  });

  final String userName;
  final int alertCount;
  final String connectionState;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final isLive = connectionState == 'Live';
    final dotColor = isLive
        ? DTTokens.accentLive
        : (connectionState == 'Disconnected'
            ? DTTokens.statusCritical
            : DTTokens.statusWarning);

    return Container(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 14),
      decoration: BoxDecoration(
        color: palette.canvas,
        border: Border(
          bottom: BorderSide(color: palette.borderSubtle, width: 1),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: DTTokens.accentPrimary.withOpacity(0.15),
              borderRadius: BorderRadius.circular(DTTokens.radiusMd),
              border: Border.all(
                color: DTTokens.accentPrimary.withOpacity(0.30),
                width: 1,
              ),
            ),
            child: Center(
              child: Text(
                userName.isNotEmpty ? userName[0].toUpperCase() : 'U',
                style: const TextStyle(
                  color: DTTokens.accentPrimary,
                  fontWeight: FontWeight.w700,
                  fontSize: 16,
                ),
              ),
            ),
          ),
          const SizedBox(width: DTTokens.space12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Factory Overview',
                  style: DTTokens.h2(palette.textPrimary),
                ),
                const SizedBox(height: 2),
                Row(
                  children: [
                    Container(
                      width: 6,
                      height: 6,
                      decoration: BoxDecoration(
                        color: dotColor,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 6),
                    Text(
                      connectionState,
                      style: DTTokens.caption(palette.textSecondary),
                    ),
                    Text(
                      '  ·  Welcome, $userName',
                      style: DTTokens.caption(palette.textTertiary),
                    ),
                  ],
                ),
              ],
            ),
          ),
          _AlertBellButton(alertCount: alertCount),
        ],
      ),
    );
  }
}

class _AlertBellButton extends StatelessWidget {
  const _AlertBellButton({required this.alertCount});

  final int alertCount;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return InkWell(
      onTap: () => context.read<AppState>().setTab(AppTab.alerts),
      borderRadius: BorderRadius.circular(DTTokens.radiusMd),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: palette.surface,
              borderRadius: BorderRadius.circular(DTTokens.radiusMd),
              border: Border.all(color: palette.borderSubtle),
            ),
            child: Icon(
              Icons.notifications_outlined,
              color: palette.textPrimary,
              size: 18,
            ),
          ),
          if (alertCount > 0)
            Positioned(
              top: -4,
              right: -4,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
                constraints: const BoxConstraints(minWidth: 18, minHeight: 18),
                decoration: BoxDecoration(
                  color: DTTokens.statusCritical,
                  borderRadius: BorderRadius.circular(DTTokens.radiusPill),
                  border: Border.all(color: palette.canvas, width: 1.5),
                ),
                child: Text(
                  alertCount > 9 ? '9+' : '$alertCount',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 10,
                    fontWeight: FontWeight.w700,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Search bar — dropped the orange/red gradient from the previous version
// ─────────────────────────────────────────────────────────────────────────────

class _DtSearchBar extends StatelessWidget {
  const _DtSearchBar({required this.onChanged});
  final ValueChanged<String> onChanged;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Container(
      decoration: BoxDecoration(
        color: palette.surface,
        borderRadius: BorderRadius.circular(DTTokens.radiusMd),
        border: Border.all(color: palette.borderSubtle),
      ),
      child: TextField(
        onChanged: onChanged,
        style: TextStyle(color: palette.textPrimary, fontSize: 14),
        decoration: InputDecoration(
          hintText: 'Search machines',
          hintStyle: TextStyle(color: palette.textTertiary, fontSize: 14),
          prefixIcon: Icon(Icons.search_rounded,
              color: palette.textTertiary, size: 20),
          border: InputBorder.none,
          enabledBorder: InputBorder.none,
          focusedBorder: InputBorder.none,
          filled: false,
          contentPadding: const EdgeInsets.symmetric(vertical: 14),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard metric grid — 2×2 cards matching the reference photo style
// ─────────────────────────────────────────────────────────────────────────────

class _DashboardMetricGrid extends StatelessWidget {
  const _DashboardMetricGrid({required this.app});
  final AppState app;

  @override
  Widget build(BuildContext context) {
    final running = app.runningCount;
    final total = app.machines.length;
    final faults = app.faultCount;
    final oee = app.overallOee;
    final alerts = app.alerts.length;

    Color oeeColor() {
      if (oee >= 75) return DTTokens.statusHealthy;
      if (oee >= 60) return DTTokens.statusWarning;
      return DTTokens.statusCritical;
    }

    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: _MetricProgressCard(
                title: 'ONLINE',
                value: '$running',
                valueSuffix: '/$total',
                subtitle: 'machines active',
                valueColor: DTTokens.statusHealthy,
                progress: total == 0 ? 0.0 : running / total,
                progressColor: DTTokens.statusHealthy,
              ),
            ),
            const SizedBox(width: DTTokens.cardGap),
            Expanded(
              child: _MetricProgressCard(
                title: 'OEE SCORE',
                value: oee.toStringAsFixed(0),
                valueSuffix: '%',
                subtitle: 'target: 85%',
                valueColor: oeeColor(),
                progress: (oee / 100).clamp(0.0, 1.0),
                progressColor: oeeColor(),
              ),
            ),
          ],
        ),
        const SizedBox(height: DTTokens.cardGap),
        Row(
          children: [
            Expanded(
              child: _MetricIconCard(
                title: 'FAULTS',
                value: faults,
                accentColor: DTTokens.statusCritical,
                icon: Icons.warning_amber_rounded,
              ),
            ),
            const SizedBox(width: DTTokens.cardGap),
            Expanded(
              child: _MetricIconCard(
                title: 'ALERTS',
                value: alerts,
                accentColor: DTTokens.statusWarning,
                icon: Icons.notifications_outlined,
              ),
            ),
          ],
        ),
      ],
    );
  }
}

// Online / OEE cards: large value + progress bar at the bottom
class _MetricProgressCard extends StatelessWidget {
  const _MetricProgressCard({
    required this.title,
    required this.value,
    required this.valueSuffix,
    required this.subtitle,
    required this.valueColor,
    required this.progress,
    required this.progressColor,
  });

  final String title;
  final String value;
  final String valueSuffix;
  final String subtitle;
  final Color valueColor;
  final double progress;
  final Color progressColor;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DtCard(
      padding: EdgeInsets.zero,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 14, 14, 10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: DTTokens.label(palette.textSecondary)
                      .copyWith(letterSpacing: 1.0, fontSize: 11),
                ),
                const SizedBox(height: 10),
                Row(
                  crossAxisAlignment: CrossAxisAlignment.baseline,
                  textBaseline: TextBaseline.alphabetic,
                  children: [
                    Text(
                      value,
                      style: DTTokens.kpi(valueColor, size: 30),
                    ),
                    Text(
                      valueSuffix,
                      style: DTTokens.caption(palette.textTertiary)
                          .copyWith(fontSize: 15),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(subtitle, style: DTTokens.caption(palette.textTertiary)),
              ],
            ),
          ),
          // Progress bar
          Padding(
            padding: const EdgeInsets.fromLTRB(14, 0, 14, 14),
            child: Stack(
              children: [
                Container(
                  height: 4,
                  decoration: BoxDecoration(
                    color: palette.borderSubtle,
                    borderRadius: BorderRadius.circular(DTTokens.radiusPill),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: progress.clamp(0.0, 1.0),
                  child: Container(
                    height: 4,
                    decoration: BoxDecoration(
                      color: progressColor,
                      borderRadius: BorderRadius.circular(DTTokens.radiusPill),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Faults / Alerts cards: tinted background + icon block + count
class _MetricIconCard extends StatelessWidget {
  const _MetricIconCard({
    required this.title,
    required this.value,
    required this.accentColor,
    required this.icon,
  });

  final String title;
  final int value;
  final Color accentColor;
  final IconData icon;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final isZero = value == 0;
    final effectiveColor = isZero ? DTTokens.statusHealthy : accentColor;

    return DtCard(
      color: isZero ? null : accentColor.alphaF(0.07),
      borderColor: isZero ? null : accentColor.alphaF(0.30),
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
      child: Row(
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(
              color: effectiveColor.alphaF(0.15),
              borderRadius: BorderRadius.circular(DTTokens.radiusMd),
            ),
            child: Icon(icon, color: effectiveColor, size: 22),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: DTTokens.label(
                    isZero ? palette.textSecondary : accentColor,
                  ).copyWith(letterSpacing: 1.0, fontSize: 11),
                ),
                const SizedBox(height: 6),
                Text(
                  '$value',
                  style: DTTokens.kpi(effectiveColor, size: 28),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top Risks card — compact preview beside KPI grid; tappable to open full list
// ─────────────────────────────────────────────────────────────────────────────

class _TopRisksCard extends StatelessWidget {
  const _TopRisksCard({required this.machines});

  /// Pre-sorted descending by risk. Top 3 are previewed; all are shown in the
  /// bottom sheet when the card or "See all" is tapped.
  final List<Machine> machines;

  void _open(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      useSafeArea: true,
      builder: (_) => _AllMachinesSheet(machines: machines),
    );
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final preview = machines.take(3).toList(growable: false);

    return DtCard(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
      onTap: () => _open(context),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
            child: Row(
              children: [
                const Icon(Icons.priority_high_rounded,
                    size: 16, color: DTTokens.statusCritical),
                const SizedBox(width: 6),
                Text(
                  'Top Risks',
                  style: DTTokens.label(palette.textSecondary)
                      .copyWith(fontWeight: FontWeight.w700),
                ),
                const Spacer(),
                Text(
                  '${machines.length}',
                  style: DTTokens.caption(palette.textTertiary),
                ),
              ],
            ),
          ),
          // Preview rows (no InkWell — taps propagate to card)
          if (preview.isEmpty)
            Padding(
              padding: const EdgeInsets.fromLTRB(12, 4, 12, 8),
              child: Text(
                'All machines healthy.',
                style: DTTokens.caption(palette.textTertiary),
              ),
            )
          else
            ...List.generate(
              preview.length,
              (i) => _AtRiskRow(
                machine: preview[i],
                divider: i < preview.length - 1,
              ),
            ),
          // See all footer
          Container(
            decoration: BoxDecoration(
              border: Border(top: BorderSide(color: palette.borderSubtle)),
            ),
            child: InkWell(
              onTap: () => _open(context),
              borderRadius: const BorderRadius.only(
                bottomLeft: Radius.circular(DTTokens.radiusMd),
                bottomRight: Radius.circular(DTTokens.radiusMd),
              ),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 10),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'See all ${machines.length} machines',
                      style: DTTokens.caption(DTTokens.accentPrimary)
                          .copyWith(fontWeight: FontWeight.w600),
                    ),
                    const SizedBox(width: 4),
                    const Icon(
                      Icons.arrow_forward_rounded,
                      size: 12,
                      color: DTTokens.accentPrimary,
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
// Performance trend card
// ─────────────────────────────────────────────────────────────────────────────

class _PerformanceTrendCard extends StatelessWidget {
  const _PerformanceTrendCard({required this.app});
  final AppState app;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final series = app.factoryPerformanceSeries;
    if (series.isEmpty) return const SizedBox.shrink();

    final spots = List.generate(
      series.length,
      (i) => FlSpot(i.toDouble(), series[i].value),
    );
    final last = series.last.value;

    return DtCard(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Factory performance',
                  style: DTTokens.h3(palette.textPrimary),
                ),
              ),
              Text(
                '${last.toStringAsFixed(0)}%',
                style: DTTokens.kpi(DTTokens.accentLive, size: 22),
              ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            'Today · OEE rolling average',
            style: DTTokens.caption(palette.textTertiary),
          ),
          const SizedBox(height: DTTokens.space12),
          SizedBox(
            height: 130,
            child: LineChart(
              LineChartData(
                minY: 50,
                maxY: 100,
                lineBarsData: [
                  LineChartBarData(
                    spots: spots,
                    isCurved: true,
                    curveSmoothness: 0.30,
                    barWidth: 2,
                    color: DTTokens.accentLive,
                    isStrokeCapRound: true,
                    dotData: FlDotData(
                      show: true,
                      checkToShowDot: (spot, _) =>
                          spot.x == spots.last.x,
                      getDotPainter: (spot, _, __, ___) =>
                          FlDotCirclePainter(
                        radius: 4,
                        color: DTTokens.accentLive,
                        strokeWidth: 2,
                        strokeColor: palette.surface,
                      ),
                    ),
                    belowBarData: BarAreaData(
                      show: true,
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          DTTokens.accentLive.withOpacity(0.20),
                          DTTokens.accentLive.withOpacity(0.0),
                        ],
                      ),
                    ),
                  ),
                ],
                titlesData: FlTitlesData(
                  leftTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  rightTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  topTitles: const AxisTitles(
                      sideTitles: SideTitles(showTitles: false)),
                  bottomTitles: AxisTitles(
                    sideTitles: SideTitles(
                      showTitles: true,
                      reservedSize: 22,
                      interval: 1,
                      getTitlesWidget: (value, _) {
                        final i = value.toInt();
                        if (i < 0 || i >= series.length) {
                          return const SizedBox();
                        }
                        return Padding(
                          padding: const EdgeInsets.only(top: 6),
                          child: Text(
                            series[i].label,
                            style: TextStyle(
                              color: palette.textTertiary,
                              fontSize: 10,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ),
                gridData: FlGridData(
                  show: true,
                  drawVerticalLine: false,
                  horizontalInterval: 10,
                  getDrawingHorizontalLine: (_) => FlLine(
                    color: palette.borderSubtle,
                    strokeWidth: 1,
                  ),
                ),
                borderData: FlBorderData(show: false),
                lineTouchData: const LineTouchData(enabled: false),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Machine grid card — with realtime data state
// ─────────────────────────────────────────────────────────────────────────────

class _MachineGridCard extends StatelessWidget {
  const _MachineGridCard({
    required this.machine,
    required this.payload,
    required this.lastTelemetry,
  });

  final Machine machine;
  final EngineTelemetryPayload? payload;
  final DateTime? lastTelemetry;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final color = _statusColor(machine.status);
    final label = _statusLabel(machine.status);

    // Determine data freshness — derived from MQTT timestamp if we have
    // one, otherwise from the machine's lastUpdated field.
    final stamp = lastTelemetry ?? machine.lastUpdated;
    final dataState = payload == null
        ? DataState.awaiting
        : dataStateFromTimestamp(stamp);

    final temp = (payload?.currentSensorReadings['s2'] ?? machine.temperature)
        .toStringAsFixed(0);

    return DtCard(
      padding: EdgeInsets.zero,
      onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => MachineDetailScreen(machineId: machine.id))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Top: status + live dot
          Padding(
            padding:
                const EdgeInsets.fromLTRB(12, 10, 10, 4),
            child: Row(
              children: [
                StatusBadge(label: label, color: color),
                const Spacer(),
                LiveDot(state: dataState, size: 6),
              ],
            ),
          ),
          // Image
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              child: MachineImage(
                machine: machine,
                fit: BoxFit.contain,
              ),
            ),
          ),
          const SizedBox(height: 8),
          // Body
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  machine.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: DTTokens.h3(palette.textPrimary),
                ),
                const SizedBox(height: 2),
                Text(
                  machine.code,
                  style: DTTokens.caption(palette.textTertiary),
                ),
                const SizedBox(height: DTTokens.space8),
                Row(
                  children: [
                    Text(
                      '$temp°',
                      style: DTTokens.mono(palette.textPrimary,
                          size: 13, weight: FontWeight.w700),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      '${machine.efficiency}%',
                      style: DTTokens.mono(palette.textSecondary,
                          size: 12),
                    ),
                    const Spacer(),
                    Text(
                      formatAge(stamp),
                      style: DTTokens.caption(palette.textTertiary)
                          .copyWith(fontSize: 10),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// At-risk row — compact inline preview, no tap (propagates to card wrapper)
// ─────────────────────────────────────────────────────────────────────────────

class _AtRiskRow extends StatelessWidget {
  const _AtRiskRow({required this.machine, required this.divider});

  final Machine machine;
  final bool divider;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final statusDot = _statusColor(machine.status);
    final risk = calculateRiskFromHealth(machine.efficiency.toDouble());
    final riskCol = _riskColor(risk);
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          child: Row(
            children: [
              Container(
                width: 8,
                height: 8,
                decoration: BoxDecoration(color: statusDot, shape: BoxShape.circle),
              ),
              const SizedBox(width: DTTokens.space12),
              Expanded(
                child: Text(
                  machine.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: DTTokens.body(palette.textPrimary)
                      .copyWith(fontWeight: FontWeight.w600, fontSize: 13),
                ),
              ),
              Text(
                '${risk.round()}% Risk',
                style: DTTokens.mono(riskCol, size: 12, weight: FontWeight.w700),
              ),
            ],
          ),
        ),
        if (divider)
          Container(
            height: 1,
            margin: const EdgeInsets.symmetric(horizontal: 12),
            color: palette.borderSubtle,
          ),
      ],
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Full machines list — draggable bottom sheet
// ─────────────────────────────────────────────────────────────────────────────

class _AllMachinesSheet extends StatelessWidget {
  const _AllMachinesSheet({required this.machines});

  /// Already sorted by risk (fault + lowest efficiency first).
  final List<Machine> machines;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DraggableScrollableSheet(
      initialChildSize: 0.65,
      minChildSize: 0.4,
      maxChildSize: 0.92,
      snap: true,
      snapSizes: const [0.65, 0.92],
      builder: (_, controller) {
        return Container(
          decoration: BoxDecoration(
            color: palette.surface,
            borderRadius: const BorderRadius.vertical(
              top: Radius.circular(DTTokens.radiusXl),
            ),
            border: Border.all(color: palette.borderSubtle),
          ),
          child: Column(
            children: [
              // Drag handle
              Padding(
                padding: const EdgeInsets.only(top: 10, bottom: 6),
                child: Container(
                  width: 36,
                  height: 4,
                  decoration: BoxDecoration(
                    color: palette.borderStrong,
                    borderRadius: BorderRadius.circular(DTTokens.radiusPill),
                  ),
                ),
              ),
              // Header
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 10),
                child: Row(
                  children: [
                    Text('All Machines', style: DTTokens.h3(palette.textPrimary)),
                    const Spacer(),
                    Text(
                      '${machines.length} · sorted by risk',
                      style: DTTokens.caption(palette.textTertiary),
                    ),
                  ],
                ),
              ),
              Container(height: 1, color: palette.borderSubtle),
              // Machine list
              Expanded(
                child: ListView.builder(
                  controller: controller,
                  itemCount: machines.length,
                  itemBuilder: (ctx, i) => _MachineSheetRow(
                    machine: machines[i],
                    divider: i < machines.length - 1,
                  ),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Machine row inside the full list sheet
// ─────────────────────────────────────────────────────────────────────────────

class _MachineSheetRow extends StatelessWidget {
  const _MachineSheetRow({required this.machine, required this.divider});

  final Machine machine;
  final bool divider;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final statusDot = _statusColor(machine.status);
    final risk = calculateRiskFromHealth(machine.efficiency.toDouble());
    final riskCol = _riskColor(risk);

    return Column(
      children: [
        InkWell(
          onTap: () {
            final nav = Navigator.of(context);
            nav.pop();
            nav.push(MaterialPageRoute(
              builder: (_) => MachineDetailScreen(machineId: machine.id),
            ));
          },
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: Row(
              children: [
                Container(
                  width: 10,
                  height: 10,
                  decoration: BoxDecoration(
                    color: statusDot,
                    shape: BoxShape.circle,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        machine.name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: DTTokens.body(palette.textPrimary)
                            .copyWith(fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        machine.location.isNotEmpty
                            ? '${machine.id}  ·  ${machine.location}'
                            : machine.id,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: DTTokens.caption(palette.textTertiary),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 12),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    Text(
                      '${risk.round()}% Risk',
                      style: DTTokens.mono(riskCol,
                          size: 13, weight: FontWeight.w700),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${machine.efficiency}% Health',
                      style: DTTokens.caption(palette.textTertiary)
                          .copyWith(fontSize: 10),
                    ),
                  ],
                ),
                const SizedBox(width: 8),
                Icon(
                  Icons.chevron_right_rounded,
                  size: 18,
                  color: palette.textTertiary,
                ),
              ],
            ),
          ),
        ),
        if (divider)
          Container(
            height: 1,
            margin: const EdgeInsets.symmetric(horizontal: 16),
            color: palette.borderSubtle,
          ),
      ],
    );
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// Status helpers (kept exported because machines_screen.dart imports them)
// ─────────────────────────────────────────────────────────────────────────────

Color _statusColor(MachineStatus status) => switch (status) {
      MachineStatus.running => DTTokens.statusHealthy,
      MachineStatus.maintenance => DTTokens.statusWarning,
      MachineStatus.fault => DTTokens.statusCritical,
      // Phase 3.6: legacy `stopped` enum value renders as Offline so
      // the word "Stopped" never appears in the UI.
      MachineStatus.stopped => DTTokens.statusOffline,
      MachineStatus.offline => DTTokens.statusOffline,
    };

String _statusLabel(MachineStatus status) => switch (status) {
      MachineStatus.running => 'RUNNING',
      MachineStatus.maintenance => 'UNDER MAINTENANCE',
      MachineStatus.fault => 'FAULT',
      MachineStatus.stopped => 'OFFLINE',
      MachineStatus.offline => 'OFFLINE',
    };

Color statusColor(MachineStatus status) => _statusColor(status);

String statusText(MachineStatus status, [bool arabic = false]) =>
    switch (status) {
      MachineStatus.running => 'Running',
      MachineStatus.maintenance => 'Under Maintenance',
      MachineStatus.fault => 'Fault',
      MachineStatus.stopped => 'Offline',
      MachineStatus.offline => 'Offline',
    };

// ─────────────────────────────────────────────────────────────────────────────
// Risk helpers
// ─────────────────────────────────────────────────────────────────────────────

double calculateRiskFromHealth(double health) {
  final clamped = health.clamp(0.0, 100.0);
  return (100.0 - clamped).clamp(0.0, 100.0);
}

Color _riskColor(double risk) {
  if (risk >= 70) return DTTokens.statusCritical;
  if (risk >= 35) return DTTokens.statusWarning;
  return DTTokens.statusHealthy;
}
