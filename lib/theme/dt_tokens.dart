import 'package:flutter/material.dart';

import 'dt_colors.dart';

/// Design tokens for the Digital Twin app.
///
/// This is the single source of truth for spacing, radius, shadow,
/// typography, and the semantic status palette. Use these instead of
/// magic numbers anywhere in the UI.
class DTTokens {
  DTTokens._();

  // ── Surfaces ────────────────────────────────────────────────────────────
  /// Primary canvas background (dark mode).
  static const Color canvas = Color(0xFF0B0F14);

  /// Default card surface.
  static const Color surface = Color(0xFF121821);

  /// Elevated surface for modals, sheets, hovered cards.
  static const Color surfaceElevated = Color(0xFF1A2230);

  /// Input fill in dark mode.
  static const Color inputFill = Color(0xFF0F1620);

  /// Light-mode counterparts.
  static const Color canvasLight = Color(0xFFF6F7FB);
  static const Color surfaceLight = Color(0xFFFFFFFF);
  static const Color surfaceElevatedLight = Color(0xFFF1F3F8);

  // ── Borders ─────────────────────────────────────────────────────────────
  static const Color borderSubtle = Color(0xFF1F2937);
  static const Color borderStrong = Color(0xFF334155);
  static const Color borderSubtleLight = Color(0xFFE2E6EE);
  static const Color borderStrongLight = Color(0xFFCBD2DE);

  // ── Text ────────────────────────────────────────────────────────────────
  static const Color textPrimary = Color(0xFFF1F5F9);
  static const Color textSecondary = Color(0xFF94A3B8);
  static const Color textTertiary = Color(0xFF64748B);

  static const Color textPrimaryLight = Color(0xFF0F172A);
  static const Color textSecondaryLight = Color(0xFF475569);
  static const Color textTertiaryLight = Color(0xFF94A3B8);

  // ── Status semantic palette ─────────────────────────────────────────────
  /// Reserved exclusively for machine status / data states. Never
  /// use these for decorative purposes.
  static const Color statusHealthy = Color(0xFF10B981);
  static const Color statusWarning = Color(0xFFF59E0B);
  static const Color statusCritical = Color(0xFFEF4444);
  static const Color statusOffline = Color(0xFF6B7280);
  static const Color statusMaintenance = Color(0xFFA78BFA);
  static const Color statusStopped = Color(0xFF8B5CF6);

  /// Brand accents (used sparingly, never to indicate machine state).
  static const Color accentPrimary = Color(0xFF3B82F6);
  static const Color accentLive = Color(0xFF22D3EE);

  // ── Spacing scale (4 px grid, only these values) ────────────────────────
  static const double space4 = 4;
  static const double space8 = 8;
  static const double space12 = 12;
  static const double space16 = 16;
  static const double space20 = 20;
  static const double space24 = 24;
  static const double space32 = 32;

  // ── Radius scale ────────────────────────────────────────────────────────
  static const double radiusSm = 8;
  static const double radiusMd = 12;
  static const double radiusLg = 16;
  static const double radiusXl = 20;
  static const double radiusPill = 999;

  // ── Card anatomy ────────────────────────────────────────────────────────
  static const double cardPadding = 16;
  static const double cardGap = 12;
  static const double sectionGap = 24;

  // ── Typography ──────────────────────────────────────────────────────────
  // We avoid adding a font dependency — system sans-serif with disciplined
  // weights and tracking does the heavy lifting.
  static TextStyle h1(Color color) => TextStyle(
        color: color,
        fontSize: 22,
        fontWeight: FontWeight.w700,
        height: 1.2,
        letterSpacing: -0.2,
      );

  static TextStyle h2(Color color) => TextStyle(
        color: color,
        fontSize: 18,
        fontWeight: FontWeight.w700,
        height: 1.25,
        letterSpacing: -0.1,
      );

  static TextStyle h3(Color color) => TextStyle(
        color: color,
        fontSize: 15,
        fontWeight: FontWeight.w600,
        height: 1.3,
      );

  static TextStyle body(Color color) => TextStyle(
        color: color,
        fontSize: 14,
        fontWeight: FontWeight.w500,
        height: 1.4,
      );

  static TextStyle label(Color color) => TextStyle(
        color: color,
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.4,
      );

  static TextStyle caption(Color color) => TextStyle(
        color: color,
        fontSize: 11,
        fontWeight: FontWeight.w500,
        letterSpacing: 0.2,
      );

  /// KPI / telemetry numerals. Tabular figures keep widths stable as
  /// values change in real time — the small visual feature that makes
  /// dashboards feel like instrumentation.
  static TextStyle kpi(Color color, {double size = 28}) => TextStyle(
        color: color,
        fontSize: size,
        fontWeight: FontWeight.w700,
        height: 1.0,
        letterSpacing: -0.5,
        fontFeatures: const [FontFeature.tabularFigures()],
      );

  static TextStyle mono(Color color, {double size = 13, FontWeight weight = FontWeight.w600}) =>
      TextStyle(
        color: color,
        fontSize: size,
        fontWeight: weight,
        fontFeatures: const [FontFeature.tabularFigures()],
        letterSpacing: 0.2,
      );
}

/// Helper: read text colors for the active brightness without yet
/// migrating every screen to Theme.of(context). Most of our screens
/// run in dark mode in practice.
class DTPalette {
  final bool dark;
  const DTPalette(this.dark);

  factory DTPalette.of(BuildContext context) {
    return DTPalette(Theme.of(context).brightness == Brightness.dark);
  }

  Color get canvas => dark ? DTTokens.canvas : DTTokens.canvasLight;
  Color get surface => dark ? DTTokens.surface : DTTokens.surfaceLight;
  Color get surfaceElevated =>
      dark ? DTTokens.surfaceElevated : DTTokens.surfaceElevatedLight;
  Color get borderSubtle =>
      dark ? DTTokens.borderSubtle : DTTokens.borderSubtleLight;
  Color get borderStrong =>
      dark ? DTTokens.borderStrong : DTTokens.borderStrongLight;
  Color get textPrimary =>
      dark ? DTTokens.textPrimary : DTTokens.textPrimaryLight;
  Color get textSecondary =>
      dark ? DTTokens.textSecondary : DTTokens.textSecondaryLight;
  Color get textTertiary =>
      dark ? DTTokens.textTertiary : DTTokens.textTertiaryLight;
}

/// Convenience adapter so existing code referencing `DT.green` etc.
/// keeps compiling but resolves to the new semantic palette where
/// it makes sense.
extension DTSemanticBridge on DT {
  // intentionally empty — DT's static fields remain. We re-export
  // the semantic colors via DTTokens above for new code.
}
