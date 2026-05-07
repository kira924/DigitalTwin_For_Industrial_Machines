import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../models/machine.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';
import '../widgets/machine_image.dart';
import 'machine_detail_screen.dart';

class MachineListScreen extends StatefulWidget {
  const MachineListScreen({super.key});

  @override
  State<MachineListScreen> createState() => _MachineListScreenState();
}

class _MachineListScreenState extends State<MachineListScreen> {
  String query = '';
  MachineStatus? filter;

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final list = app.machines.where((machine) {
      final q = query.toLowerCase();
      final matchQuery = q.isEmpty ||
          machine.name.toLowerCase().contains(q) ||
          machine.code.toLowerCase().contains(q) ||
          machine.type.toLowerCase().contains(q);
      // Phase 3.5: filter on the auto-resolved status, not the
      // (now-unused) machine.status field. Keeps the chip behavior
      // consistent with the badges and detail header.
      final resolved = app.resolvedStatusFor(machine.id);
      final matchFilter = filter == null || resolved == filter;
      return matchQuery && matchFilter;
    }).toList(growable: false);

    return Scaffold(
      backgroundColor: palette.canvas,
      // Add Machine FAB removed — machine creation is now restricted to
      // the Admin section per the product spec. Public Machines page
      // is browse-only for everyone (admin still gets the action via Admin).
      appBar: appHeader(
        title: 'Machines',
        subtitle: '${app.machines.length} total · ${app.runningCount} running',
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 90),
        children: [
          // Status pills
          Row(
            children: [
              Expanded(
                child: _StatusMiniStat(
                  label: 'Running',
                  value: app.runningCount,
                  color: DTTokens.statusHealthy,
                ),
              ),
              const SizedBox(width: DTTokens.cardGap),
              Expanded(
                child: _StatusMiniStat(
                  label: 'Maintenance',
                  value: app.maintenanceCount,
                  color: DTTokens.statusWarning,
                ),
              ),
              const SizedBox(width: DTTokens.cardGap),
              Expanded(
                child: _StatusMiniStat(
                  label: 'Faults',
                  value: app.faultCount,
                  color: DTTokens.statusCritical,
                ),
              ),
            ],
          ),
          const SizedBox(height: DTTokens.space16),

          // Search
          Container(
            decoration: BoxDecoration(
              color: palette.surface,
              borderRadius: BorderRadius.circular(DTTokens.radiusMd),
              border: Border.all(color: palette.borderSubtle),
            ),
            child: TextField(
              decoration: InputDecoration(
                hintText: 'Search machines, codes, types',
                prefixIcon: Icon(
                  Icons.search_rounded,
                  size: 20,
                  color: palette.textTertiary,
                ),
                border: InputBorder.none,
                enabledBorder: InputBorder.none,
                focusedBorder: InputBorder.none,
                filled: false,
                contentPadding: const EdgeInsets.symmetric(vertical: 14),
              ),
              onChanged: (value) => setState(() => query = value),
            ),
          ),
          const SizedBox(height: DTTokens.space12),

          // Filter chips
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                _DtFilterChip(
                  label: 'All',
                  active: filter == null,
                  onTap: () => setState(() => filter = null),
                ),
                _DtFilterChip(
                  label: 'Running',
                  active: filter == MachineStatus.running,
                  color: DTTokens.statusHealthy,
                  onTap: () => setState(
                      () => filter = MachineStatus.running),
                ),
                _DtFilterChip(
                  label: 'Maintenance',
                  active: filter == MachineStatus.maintenance,
                  color: DTTokens.statusWarning,
                  onTap: () => setState(
                      () => filter = MachineStatus.maintenance),
                ),
                _DtFilterChip(
                  label: 'Fault',
                  active: filter == MachineStatus.fault,
                  color: DTTokens.statusCritical,
                  onTap: () =>
                      setState(() => filter = MachineStatus.fault),
                ),
              ],
            ),
          ),
          const SizedBox(height: DTTokens.space16),

          // Grid
          if (list.isEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 32),
              child: EmptyView(
                icon: Icons.precision_manufacturing_outlined,
                title: 'No machines match',
                description: query.isNotEmpty
                    ? 'Try a different search term or clear filters.'
                    : 'Add your first machine from the Admin tab to start monitoring.',
              ),
            )
          else
            LayoutBuilder(
              builder: (context, constraints) {
                final width = constraints.maxWidth;
                final crossAxisCount =
                    width > 1000 ? 3 : (width > 640 ? 2 : 1);
                final ratio =
                    width > 640 ? 1.05 : (crossAxisCount == 1 ? 1.6 : 1.05);
                return GridView.builder(
                  itemCount: list.length,
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: crossAxisCount,
                    mainAxisSpacing: DTTokens.cardGap,
                    crossAxisSpacing: DTTokens.cardGap,
                    childAspectRatio: ratio,
                  ),
                  itemBuilder: (context, index) =>
                      _MachineListCard(machine: list[index]),
                );
              },
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini stat
// ─────────────────────────────────────────────────────────────────────────────

class _StatusMiniStat extends StatelessWidget {
  final String label;
  final int value;
  final Color color;

  const _StatusMiniStat({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DtCard(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 6,
                height: 6,
                decoration: BoxDecoration(
                  color: color,
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 6),
              Text(
                label,
                style: DTTokens.label(palette.textSecondary)
                    .copyWith(letterSpacing: 0.4),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '$value',
            style: DTTokens.kpi(color, size: 22),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Filter chip
// ─────────────────────────────────────────────────────────────────────────────

class _DtFilterChip extends StatelessWidget {
  final String label;
  final bool active;
  final VoidCallback onTap;
  final Color? color;

  const _DtFilterChip({
    required this.label,
    required this.active,
    required this.onTap,
    this.color,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final accent = color ?? DTTokens.accentPrimary;
    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(DTTokens.radiusPill),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 160),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
          decoration: BoxDecoration(
            color: active ? accent.alphaF(0.16) : palette.surface,
            borderRadius: BorderRadius.circular(DTTokens.radiusPill),
            border: Border.all(
              color: active ? accent.alphaF(0.40) : palette.borderSubtle,
            ),
          ),
          child: Text(
            label,
            style: TextStyle(
              color: active ? accent : palette.textSecondary,
              fontSize: 12,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.2,
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Machine card (list view)
// ─────────────────────────────────────────────────────────────────────────────

class _MachineListCard extends StatelessWidget {
  final Machine machine;
  const _MachineListCard({required this.machine});

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final color = statusColor(machine.status);
    final dataState = dataStateFromTimestamp(machine.lastUpdated);

    return DtCard(
      padding: EdgeInsets.zero,
      onTap: () => Navigator.of(context).push(MaterialPageRoute(builder: (_) => MachineDetailScreen(machineId: machine.id))),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Image with status overlay (no heavy gradient)
          ClipRRect(
            borderRadius: const BorderRadius.vertical(
              top: Radius.circular(DTTokens.radiusMd),
            ),
            child: Stack(
              children: [
                Container(
                  height: 110,
                  width: double.infinity,
                  color: palette.surfaceElevated,
                  child: MachineImage(
                    machine: machine,
                    fit: BoxFit.cover,
                  ),
                ),
                // Subtle bottom-fade so text stays legible if alert overlays
                Positioned.fill(
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.transparent,
                          Colors.black.alphaF(0.45),
                        ],
                      ),
                    ),
                  ),
                ),
                Positioned(
                  top: 10,
                  left: 10,
                  child: StatusBadge(
                    label: statusText(machine.status).toUpperCase(),
                    color: color,
                  ),
                ),
                Positioned(
                  top: 10,
                  right: 10,
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                    decoration: BoxDecoration(
                      color: Colors.black.alphaF(0.40),
                      borderRadius: BorderRadius.circular(DTTokens.radiusPill),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        LiveDot(state: dataState, size: 5),
                        const SizedBox(width: 4),
                        Text(
                          dataState.label,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 10,
                            fontWeight: FontWeight.w600,
                            letterSpacing: 0.2,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                if (machine.activeAlert != null)
                  Positioned(
                    left: 10,
                    right: 10,
                    bottom: 8,
                    child: Text(
                      machine.activeAlert!,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
              ],
            ),
          ),
          // Body
          Padding(
            padding: const EdgeInsets.all(14),
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
                  '${machine.code} · ${machine.location}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: DTTokens.caption(palette.textTertiary),
                ),
                const SizedBox(height: DTTokens.space12),
                Row(
                  children: [
                    Expanded(
                      child: _InlineMetric(
                        label: 'Output',
                        value: '${machine.productionRate.toStringAsFixed(0)}%',
                        color: color,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _InlineMetric(
                        label: 'Power',
                        value: '${machine.power.toStringAsFixed(1)} kW',
                        color: DTTokens.accentLive,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Icon(
                      Icons.schedule_rounded,
                      size: 12,
                      color: palette.textTertiary,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Updated ${formatAge(machine.lastUpdated)}',
                      style: DTTokens.caption(palette.textTertiary),
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

class _InlineMetric extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _InlineMetric({
    required this.label,
    required this.value,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: palette.surfaceElevated,
        borderRadius: BorderRadius.circular(DTTokens.radiusSm),
        border: Border.all(color: palette.borderSubtle),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: DTTokens.caption(palette.textTertiary)
                .copyWith(letterSpacing: 0.4),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: DTTokens.mono(color, size: 14, weight: FontWeight.w700),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Local helpers (status color/text). Kept in sync with dashboard helpers.
// ─────────────────────────────────────────────────────────────────────────────

Color statusColor(MachineStatus status) => switch (status) {
      MachineStatus.running => DTTokens.statusHealthy,
      MachineStatus.maintenance => DTTokens.statusWarning,
      MachineStatus.fault => DTTokens.statusCritical,
      MachineStatus.stopped => DTTokens.statusOffline,
      MachineStatus.offline => DTTokens.statusOffline,
    };

String statusText(MachineStatus status, [bool arabic = false]) =>
    switch (status) {
      MachineStatus.running => 'Running',
      MachineStatus.maintenance => 'Under Maintenance',
      MachineStatus.fault => 'Fault',
      MachineStatus.stopped => 'Offline',
      MachineStatus.offline => 'Offline',
    };

/// Kept for compatibility with old call sites.
String timeAgo(DateTime value, [bool arabic = false]) =>
    formatAge(value);
