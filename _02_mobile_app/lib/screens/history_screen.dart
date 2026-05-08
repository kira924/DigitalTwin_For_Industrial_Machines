import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);

    return Scaffold(
      backgroundColor: palette.canvas,
      appBar: appHeader(
        title: 'Activity history',
        subtitle: 'Machine, alert, and admin events',
      ),
      body: app.history.isEmpty
          ? const EmptyView(
              icon: Icons.history_rounded,
              title: 'No activity yet',
              description: 'Machine and alert events will appear here.',
            )
          : _HistoryList(history: app.history),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// History list
// ─────────────────────────────────────────────────────────────────────────────

class _HistoryList extends StatelessWidget {
  const _HistoryList({required this.history});
  final List<HistoryLog> history;

  Map<String, List<HistoryLog>> _grouped() {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final yesterday = today.subtract(const Duration(days: 1));
    final groups = <String, List<HistoryLog>>{};
    for (final event in history) {
      final date = DateTime(event.time.year, event.time.month, event.time.day);
      final String key;
      if (date == today) {
        key = 'Today';
      } else if (date == yesterday) {
        key = 'Yesterday';
      } else {
        key = '${event.time.day}/${event.time.month}/${event.time.year}';
      }
      groups.putIfAbsent(key, () => []).add(event);
    }
    return groups;
  }

  @override
  Widget build(BuildContext context) {
    final groups = _grouped();
    final items = <Widget>[];
    groups.forEach((day, events) {
      items.add(_DaySectionHeader(label: day));
      for (var i = 0; i < events.length; i++) {
        items.add(_TimelineItem(
          event: events[i],
          isFirst: i == 0,
          isLast: i == events.length - 1,
        ));
      }
    });
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 90),
      children: items,
    );
  }
}

class _DaySectionHeader extends StatelessWidget {
  const _DaySectionHeader({required this.label});
  final String label;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(0, 14, 0, 8),
      child: Row(
        children: [
          Text(
            label.toUpperCase(),
            style: DTTokens.label(palette.textSecondary)
                .copyWith(letterSpacing: 1.4),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Container(
              height: 1,
              color: palette.borderSubtle,
            ),
          ),
        ],
      ),
    );
  }
}

class _TimelineItem extends StatelessWidget {
  const _TimelineItem({
    required this.event,
    required this.isFirst,
    required this.isLast,
  });

  final HistoryLog event;
  final bool isFirst;
  final bool isLast;

  String _time(DateTime t) {
    final hh = t.hour.toString().padLeft(2, '0');
    final mm = t.minute.toString().padLeft(2, '0');
    return '$hh:$mm';
  }

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // Timeline rail
          SizedBox(
            width: 28,
            child: Column(
              children: [
                Container(
                  width: 2,
                  height: 8,
                  color: isFirst ? Colors.transparent : palette.borderSubtle,
                ),
                Container(
                  width: 12,
                  height: 12,
                  decoration: BoxDecoration(
                    color: event.color,
                    shape: BoxShape.circle,
                    border: Border.all(color: palette.canvas, width: 2),
                  ),
                ),
                Expanded(
                  child: Container(
                    width: 2,
                    color: isLast ? Colors.transparent : palette.borderSubtle,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 8),
          // Card
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: DTTokens.cardGap),
              child: DtCard(
                padding: const EdgeInsets.all(14),
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
                            style: DTTokens.body(palette.textPrimary)
                                .copyWith(fontWeight: FontWeight.w600),
                          ),
                        ),
                        Text(
                          _time(event.time),
                          style: DTTokens.mono(palette.textTertiary, size: 11),
                        ),
                      ],
                    ),
                    const SizedBox(height: 4),
                    Text(
                      event.subtitle,
                      style: DTTokens.caption(palette.textSecondary),
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
