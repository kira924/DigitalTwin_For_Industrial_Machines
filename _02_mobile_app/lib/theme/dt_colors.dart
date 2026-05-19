import 'package:flutter/material.dart';

extension ColorAlphaFix on Color {
  Color alphaF(double alpha) => withValues(alpha: alpha.clamp(0.0, 1.0));
}

/// Legacy color tokens — kept for backwards compatibility with the
/// existing screens. Values have been retuned to the new semantic
/// palette so every screen picks up the polish automatically.
///
/// New code should reference DTTokens (theme/dt_tokens.dart) directly.
class DT {
  // ── Canvas ──────────────────────────────────────────────────────────────
  static const bg = Color(0xFF0B0F14);
  static const headerA = Color(0xFF0F1620);
  static const headerB = Color(0xFF121821);

  // ── Brand accents ───────────────────────────────────────────────────────
  static const blue = Color(0xFF3B82F6);
  static const cyan = Color(0xFF22D3EE);

  // ── Status colors (semantic — reserved for machine state) ───────────────
  // The legacy names are kept so existing screens keep compiling.
  static const green = Color(0xFF10B981); // healthy / running
  static const yellow = Color(0xFFF59E0B); // warning / maintenance
  static const red = Color(0xFFEF4444); // critical / fault
  static const purple = Color(0xFF8B5CF6); // stopped
  static const pink = Color(0xFFA78BFA); // alt accent

  // ── Surfaces & borders ──────────────────────────────────────────────────
  static Color surface([double opacity = 1.0]) =>
      const Color(0xFF121821).alphaF(opacity);

  static Color border([double opacity = 1.0]) =>
      const Color(0xFF1F2937).alphaF(opacity);

  static Color muted([double opacity = 0.65]) => Colors.white.alphaF(opacity);
  static Color dim([double opacity = 0.45]) => Colors.white.alphaF(opacity);

  /// Two-stop gradient used sparingly — only for the brand mark and the
  /// primary CTA on the login screen. Avoid using on incidental UI.
  static const grad = LinearGradient(
    colors: [blue, cyan],
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
  );
}
