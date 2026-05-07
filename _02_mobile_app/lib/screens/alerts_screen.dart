import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';
import 'machine_detail_screen.dart';

/// Emergency-only alerts feed.
///
/// Per the product spec, Alerts is no longer a configuration screen.
/// All threshold sliders and notification toggles have moved to the
/// Admin → Settings panel. This screen now shows only critical /
/// warning incidents and lets the user tap through to the machine.
class AlertsScreen extends StatelessWidget {
  const AlertsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);

    final critical = app.alerts
        .where((a) => a.color == DTTokens.statusCritical || a.color == DT.red)
        .length;
    final warnings = app.alerts.length - critical;

    return Scaffold(
      backgroundColor: palette.canvas,
      appBar: appHeader(
        title: 'Alerts',
        subtitle: app.alerts.isEmpty
            ? 'No emergency events right now'
            : '${app.alerts.length} active · $critical critical · $warnings warning',
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 90),
        children: [
          // Banner explaining what alerts is for now.
          if (app.alerts.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(bottom: DTTokens.space16),
              child: DtCard(
                padding: const EdgeInsets.all(DTTokens.space12),
                child: Row(
                  children: [
                    const Icon(Icons.crisis_alert_rounded,
                        size: 16, color: DTTokens.statusCritical),
                    const SizedBox(width: DTTokens.space8),
                    Expanded(
                      child: Text(
                        'Emergency alerts only. Notification settings and '
                        'thresholds moved to Admin → Settings.',
                        style: DTTokens.caption(palette.textSecondary)
                            .copyWith(height: 1.4),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          if (app.alerts.isEmpty)
            const Padding(
              padding: EdgeInsets.only(top: 32, bottom: 24),
              child: EmptyView(
                icon: Icons.shield_outlined,
                title: 'All systems nominal',
                description:
                    'No emergency events right now. Critical incidents will appear here when machines exceed safe operating limits.',
              ),
            )
          else
            ...app.alerts.map(
              (alert) => Padding(
                padding: const EdgeInsets.only(bottom: DTTokens.cardGap),
                child: _AlertCard(alert: alert),
              ),
            ),
        ],
      ),
    );
  }
}

class _AlertCard extends StatelessWidget {
  const _AlertCard({required this.alert});
  final AppAlert alert;

  String _severity() {
    if (alert.color == DTTokens.statusCritical || alert.color == DT.red) {
      return 'CRITICAL';
    }
    if (alert.color == DTTokens.statusWarning || alert.color == DT.yellow) {
      return 'WARNING';
    }
    return 'INFO';
  }

  String _formatTime(DateTime t) {
    final now = DateTime.now();
    final age = now.difference(t);
    if (age.inMinutes < 1) return 'just now';
    if (age.inMinutes < 60) return '${age.inMinutes}m ago';
    if (age.inHours < 24) return '${age.inHours}h ago';
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    return '$hh:$mm';
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DtCard(
      padding: const EdgeInsets.fromLTRB(0, 0, 14, 14),
      onTap: () {
        // Tap an alert to jump to the affected machine's detail page.
        // Only navigate if the machine still exists in the roster.
        final app = context.read<AppState>();
        final exists = app.machines.any((m) => m.id == alert.machineId);
        if (!exists) return;
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => MachineDetailScreen(machineId: alert.machineId),
          ),
        );
      },
      child: IntrinsicHeight(
        child: Row(
          children: [
            // Left severity bar
            Container(
              width: 4,
              decoration: BoxDecoration(
                color: alert.color,
                borderRadius: const BorderRadius.horizontal(
                  left: Radius.circular(DTTokens.radiusMd),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(top: 14),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        StatusBadge(label: _severity(), color: alert.color),
                        const Spacer(),
                        Text(
                          _formatTime(alert.time),
                          style: DTTokens.caption(palette.textTertiary),
                        ),
                      ],
                    ),
                    const SizedBox(height: DTTokens.space8),
                    Text(
                      alert.title,
                      style: DTTokens.h3(palette.textPrimary),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      alert.message,
                      style: DTTokens.body(palette.textSecondary),
                    ),
                    const SizedBox(height: 6),
                    Row(
                      children: [
                        Icon(
                          Icons.chevron_right_rounded,
                          size: 14,
                          color: palette.textTertiary,
                        ),
                        Text(
                          'View machine',
                          style: DTTokens.caption(palette.textTertiary)
                              .copyWith(fontWeight: FontWeight.w600),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
