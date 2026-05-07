import 'package:flutter/material.dart';

import 'dt_colors.dart';
import 'dt_tokens.dart';

// ─────────────────────────────────────────────────────────────────────────────
// App header
// ─────────────────────────────────────────────────────────────────────────────

/// Flat app header — replaces the gradient AppBar from the previous design.
/// Quiet, dense, and consistent across every screen.
PreferredSizeWidget appHeader({
  required String title,
  String? subtitle,
  Widget? leading,
  Widget? trailing,
}) {
  return _DtHeader(
    title: title,
    subtitle: subtitle,
    leading: leading,
    trailing: trailing,
  );
}

class _DtHeader extends StatelessWidget implements PreferredSizeWidget {
  const _DtHeader({
    required this.title,
    this.subtitle,
    this.leading,
    this.trailing,
  });

  final String title;
  final String? subtitle;
  final Widget? leading;
  final Widget? trailing;

  @override
  Size get preferredSize => const Size.fromHeight(72);

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return AppBar(
      automaticallyImplyLeading: false,
      toolbarHeight: 72,
      elevation: 0,
      scrolledUnderElevation: 0,
      backgroundColor: palette.canvas,
      surfaceTintColor: Colors.transparent,
      shape: Border(
        bottom: BorderSide(color: palette.borderSubtle, width: 1),
      ),
      titleSpacing: DTTokens.space20,
      title: Row(
        children: [
          if (leading != null) ...[
            leading!,
            const SizedBox(width: DTTokens.space12),
          ],
          Expanded(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: DTTokens.h1(palette.textPrimary),
                ),
                if (subtitle != null) ...[
                  const SizedBox(height: 2),
                  Text(
                    subtitle!,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: DTTokens.caption(palette.textSecondary),
                  ),
                ],
              ],
            ),
          ),
          if (trailing != null) trailing!,
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Card primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Flat card — the design-system building block. 1px subtle border,
/// no blur, no heavy shadow. Used everywhere a piece of content
/// needs to be grouped.
class DtCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;
  final Color? color;
  final Color? borderColor;
  final VoidCallback? onTap;

  const DtCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(DTTokens.cardPadding),
    this.radius = DTTokens.radiusMd,
    this.color,
    this.borderColor,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final fill = color ?? palette.surface;
    final border = borderColor ?? palette.borderSubtle;

    final card = AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      curve: Curves.easeOut,
      decoration: BoxDecoration(
        color: fill,
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: border, width: 1),
      ),
      child: Padding(padding: padding, child: child),
    );

    if (onTap == null) return card;
    return Material(
      color: Colors.transparent,
      borderRadius: BorderRadius.circular(radius),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(radius),
        child: card,
      ),
    );
  }
}

/// Backwards compatibility shim — `GlassCard` is referenced from a lot
/// of existing screens. Keeping the name means we don't have to touch
/// every callsite, and we get the polish for free.
class GlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final double radius;

  const GlassCard({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(DTTokens.cardPadding),
    this.radius = DTTokens.radiusMd,
  });

  @override
  Widget build(BuildContext context) {
    return DtCard(padding: padding, radius: radius, child: child);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Status badge & live dot — used for machine cards, dashboard headers, etc.
// ─────────────────────────────────────────────────────────────────────────────

class StatusBadge extends StatelessWidget {
  final String label;
  final Color color;
  final IconData? icon;
  final double fontSize;

  const StatusBadge({
    super.key,
    required this.label,
    required this.color,
    this.icon,
    this.fontSize = 11,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.alphaF(0.13),
        borderRadius: BorderRadius.circular(DTTokens.radiusPill),
        border: Border.all(color: color.alphaF(0.30), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[
            Icon(icon, size: fontSize + 2, color: color),
            const SizedBox(width: 4),
          ] else ...[
            Container(
              width: 6,
              height: 6,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle),
            ),
            const SizedBox(width: 6),
          ],
          Text(
            label,
            style: TextStyle(
              color: color,
              fontSize: fontSize,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.4,
            ),
          ),
        ],
      ),
    );
  }
}

/// Pulsing dot indicating live data flow. When [state] is stale or
/// offline the pulse is suppressed and the color shifts.
class LiveDot extends StatefulWidget {
  final DataState state;
  final double size;

  const LiveDot({super.key, required this.state, this.size = 8});

  @override
  State<LiveDot> createState() => _LiveDotState();
}

class _LiveDotState extends State<LiveDot>
    with SingleTickerProviderStateMixin {
  late final AnimationController _ctrl;

  @override
  void initState() {
    super.initState();
    _ctrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat();
  }

  @override
  void dispose() {
    _ctrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final color = widget.state.color;
    final shouldPulse = widget.state == DataState.live;

    return SizedBox(
      width: widget.size + 8,
      height: widget.size + 8,
      child: Stack(
        alignment: Alignment.center,
        children: [
          if (shouldPulse)
            AnimatedBuilder(
              animation: _ctrl,
              builder: (_, __) {
                final t = _ctrl.value;
                return Container(
                  width: widget.size + (t * 8),
                  height: widget.size + (t * 8),
                  decoration: BoxDecoration(
                    color: color.alphaF((1 - t) * 0.5),
                    shape: BoxShape.circle,
                  ),
                );
              },
            ),
          Container(
            width: widget.size,
            height: widget.size,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
        ],
      ),
    );
  }
}

/// Realtime data freshness state, derived from `lastUpdated` timestamps.
enum DataState { live, stale, offline, awaiting }

extension DataStateExtension on DataState {
  Color get color {
    switch (this) {
      case DataState.live:
        return DTTokens.accentLive;
      case DataState.stale:
        return DTTokens.statusWarning;
      case DataState.offline:
        return DTTokens.statusOffline;
      case DataState.awaiting:
        return DTTokens.statusOffline;
    }
  }

  String get label {
    switch (this) {
      case DataState.live:
        return 'Live';
      case DataState.stale:
        return 'Stale';
      case DataState.offline:
        return 'Offline';
      case DataState.awaiting:
        return 'Awaiting data';
    }
  }
}

/// Compute freshness state from a last-update timestamp. Tunable
/// thresholds: <15 s = live, <2 min = stale, otherwise offline.
DataState dataStateFromTimestamp(DateTime lastUpdated, {DateTime? now}) {
  final t = now ?? DateTime.now();
  final age = t.difference(lastUpdated);
  if (age.inSeconds < 15) return DataState.live;
  if (age.inSeconds < 120) return DataState.stale;
  return DataState.offline;
}

String formatAge(DateTime lastUpdated, {DateTime? now}) {
  final t = now ?? DateTime.now();
  final age = t.difference(lastUpdated);
  if (age.inSeconds < 5) return 'just now';
  if (age.inSeconds < 60) return '${age.inSeconds}s ago';
  if (age.inMinutes < 60) return '${age.inMinutes}m ago';
  if (age.inHours < 24) return '${age.inHours}h ago';
  return '${age.inDays}d ago';
}

// ─────────────────────────────────────────────────────────────────────────────
// KPI card — the dashboard hero
// ─────────────────────────────────────────────────────────────────────────────

class KpiCard extends StatelessWidget {
  final String label;
  final String value;
  final String? unit;
  final Color? valueColor;
  final IconData? icon;
  final Widget? trailing;

  const KpiCard({
    super.key,
    required this.label,
    required this.value,
    this.unit,
    this.valueColor,
    this.icon,
    this.trailing,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DtCard(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              if (icon != null) ...[
                Icon(icon, size: 14, color: palette.textTertiary),
                const SizedBox(width: 6),
              ],
              Expanded(
                child: Text(
                  label.toUpperCase(),
                  style: DTTokens.label(palette.textSecondary)
                      .copyWith(letterSpacing: 0.6),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Flexible(
                child: Text(
                  value,
                  style: DTTokens.kpi(valueColor ?? palette.textPrimary),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              if (unit != null) ...[
                const SizedBox(width: 4),
                Text(
                  unit!,
                  style: DTTokens.caption(palette.textTertiary),
                ),
              ],
              if (trailing != null) ...[
                const Spacer(),
                trailing!,
              ],
            ],
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section header (used for "Top risky", "Recent alerts", etc.)
// ─────────────────────────────────────────────────────────────────────────────

class SectionHeader extends StatelessWidget {
  final String title;
  final String? trailingText;
  final VoidCallback? onTrailingTap;

  const SectionHeader({
    super.key,
    required this.title,
    this.trailingText,
    this.onTrailingTap,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(2, 4, 2, 10),
      child: Row(
        children: [
          Expanded(
            child: Text(title, style: DTTokens.h2(palette.textPrimary)),
          ),
          if (trailingText != null)
            InkWell(
              onTap: onTrailingTap,
              borderRadius: BorderRadius.circular(DTTokens.radiusSm),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
                child: Text(
                  trailingText!,
                  style: DTTokens.caption(DTTokens.accentPrimary)
                      .copyWith(fontWeight: FontWeight.w700),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Empty view
// ─────────────────────────────────────────────────────────────────────────────

class EmptyView extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? description;
  final Widget? action;

  const EmptyView({
    super.key,
    required this.icon,
    required this.title,
    this.description,
    this.action,
  });

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(DTTokens.space24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 56,
              height: 56,
              decoration: BoxDecoration(
                color: palette.surfaceElevated,
                borderRadius: BorderRadius.circular(DTTokens.radiusLg),
                border: Border.all(color: palette.borderSubtle),
              ),
              child: Icon(icon, color: palette.textTertiary, size: 24),
            ),
            const SizedBox(height: DTTokens.space16),
            Text(
              title,
              textAlign: TextAlign.center,
              style: DTTokens.h3(palette.textPrimary),
            ),
            if (description != null) ...[
              const SizedBox(height: 6),
              Text(
                description!,
                textAlign: TextAlign.center,
                style: DTTokens.body(palette.textSecondary),
              ),
            ],
            if (action != null) ...[
              const SizedBox(height: DTTokens.space16),
              action!,
            ],
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Primary button — replaces the heavy GradientButton with something
// quieter that still reads as a primary CTA.
// ─────────────────────────────────────────────────────────────────────────────

class GradientButton extends StatelessWidget {
  final String text;
  final VoidCallback onTap;
  final IconData? icon;

  const GradientButton({
    super.key,
    required this.text,
    required this.onTap,
    this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 50,
      width: double.infinity,
      child: DecoratedBox(
        decoration: BoxDecoration(
          gradient: DT.grad,
          borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          boxShadow: [
            BoxShadow(
              color: DT.blue.alphaF(0.20),
              blurRadius: 18,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: ElevatedButton(
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.transparent,
            shadowColor: Colors.transparent,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(DTTokens.radiusMd),
            ),
          ),
          onPressed: onTap,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (icon != null) ...[
                Icon(icon, color: Colors.white, size: 18),
                const SizedBox(width: 8),
              ],
              Text(
                text,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 15,
                  fontWeight: FontWeight.w700,
                  letterSpacing: 0.2,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Outlined secondary action button (used on detail screen quick actions).
class DtActionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final Color color;
  final VoidCallback onTap;

  const DtActionButton({
    super.key,
    required this.label,
    required this.icon,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: 46,
      child: OutlinedButton.icon(
        onPressed: onTap,
        icon: Icon(icon, size: 18, color: color),
        label: Text(
          label,
          style: TextStyle(
            color: color,
            fontWeight: FontWeight.w600,
            fontSize: 14,
            letterSpacing: 0.1,
          ),
        ),
        style: OutlinedButton.styleFrom(
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(DTTokens.radiusMd),
          ),
          side: BorderSide(color: color.alphaF(0.30)),
          backgroundColor: color.alphaF(0.06),
        ),
      ),
    );
  }
}
