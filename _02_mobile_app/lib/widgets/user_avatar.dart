import 'package:flutter/material.dart';

import '../models/app_user.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';

/// Avatar for an [AppUser].
///
/// Renders precedence:
///   1. `user.photoUrl` (uploaded to Firebase Storage)
///   2. Initials in a coloured circle (deterministic per user id)
///
/// Never shows the broken-image icon — a network failure falls back
/// to initials automatically.
class UserAvatar extends StatelessWidget {
  const UserAvatar({
    super.key,
    required this.user,
    this.size = 36,
  });

  final AppUser user;
  final double size;

  String get _initials {
    final n = user.name.trim();
    if (n.isEmpty) return '?';
    final parts = n.split(RegExp(r'\s+')).where((p) => p.isNotEmpty).toList();
    if (parts.length == 1) return parts.first.characters.first.toUpperCase();
    return (parts.first.characters.first + parts.last.characters.first)
        .toUpperCase();
  }

  Color get _bg {
    // Stable deterministic color per user id — keeps the avatar
    // recognizable across rebuilds without needing a server preference.
    final palette = <Color>[
      DTTokens.accentPrimary,
      DTTokens.accentLive,
      DTTokens.statusHealthy,
      DTTokens.statusWarning,
      DTTokens.statusStopped,
      DTTokens.statusMaintenance,
    ];
    final hash = user.id.codeUnits.fold<int>(0, (a, c) => (a + c) & 0x7fffffff);
    return palette[hash % palette.length];
  }

  @override
  Widget build(BuildContext context) {
    final url = user.photoUrl;
    final radius = size / 2;
    final placeholder = _initialsCircle();
    if (url != null && url.isNotEmpty) {
      return ClipOval(
        child: SizedBox(
          width: size,
          height: size,
          child: Image.network(
            url,
            fit: BoxFit.cover,
            errorBuilder: (_, __, ___) => placeholder,
            loadingBuilder: (_, w, p) {
              if (p == null) return w;
              return Container(
                width: size,
                height: size,
                color: DTTokens.surfaceElevated,
                alignment: Alignment.center,
                child: SizedBox(
                  width: radius * 0.6,
                  height: radius * 0.6,
                  child: const CircularProgressIndicator(strokeWidth: 2),
                ),
              );
            },
          ),
        ),
      );
    }
    return placeholder;
  }

  Widget _initialsCircle() {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: _bg.alphaF(0.18),
        shape: BoxShape.circle,
        border: Border.all(color: _bg.alphaF(0.40), width: 1),
      ),
      alignment: Alignment.center,
      child: Text(
        _initials,
        style: TextStyle(
          color: _bg,
          fontSize: size * 0.40,
          fontWeight: FontWeight.w800,
          letterSpacing: 0.4,
        ),
      ),
    );
  }
}
