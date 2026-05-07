import 'package:flutter/material.dart';

import '../models/machine.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';

/// Renders a machine's photo with the right precedence:
///   1. `imageUrl` (uploaded user photo from Firebase Storage)
///   2. `imagePath` (bundled asset fallback)
///   3. Polished placeholder (when neither resolves)
///
/// The widget never shows the broken-image icon: any load failure on
/// the network image automatically falls back to the asset, and any
/// asset failure falls back to the placeholder.
class MachineImage extends StatelessWidget {
  const MachineImage({
    super.key,
    required this.machine,
    this.fit = BoxFit.cover,
    this.borderRadius,
  });

  final Machine machine;
  final BoxFit fit;
  final BorderRadius? borderRadius;

  @override
  Widget build(BuildContext context) {
    Widget child;
    final url = machine.imageUrl;
    if (url != null && url.isNotEmpty) {
      child = Image.network(
        url,
        fit: fit,
        errorBuilder: (_, __, ___) => _AssetOrPlaceholder(machine: machine, fit: fit),
        loadingBuilder: (_, w, progress) {
          if (progress == null) return w;
          return Container(
            color: DTTokens.surface,
            alignment: Alignment.center,
            child: const SizedBox(
              width: 18,
              height: 18,
              child: CircularProgressIndicator(strokeWidth: 2),
            ),
          );
        },
      );
    } else {
      child = _AssetOrPlaceholder(machine: machine, fit: fit);
    }

    if (borderRadius != null) {
      return ClipRRect(borderRadius: borderRadius!, child: child);
    }
    return child;
  }
}

class _AssetOrPlaceholder extends StatelessWidget {
  const _AssetOrPlaceholder({required this.machine, required this.fit});
  final Machine machine;
  final BoxFit fit;

  @override
  Widget build(BuildContext context) {
    final asset = machine.imagePath;
    if (asset.isEmpty) return const _Placeholder();
    return Image.asset(
      asset,
      fit: fit,
      errorBuilder: (_, __, ___) => const _Placeholder(),
    );
  }
}

class _Placeholder extends StatelessWidget {
  const _Placeholder();

  @override
  Widget build(BuildContext context) {
    return Container(
      color: DTTokens.surfaceElevated,
      alignment: Alignment.center,
      child: Icon(
        Icons.precision_manufacturing_outlined,
        size: 28,
        color: DTTokens.textTertiary.alphaF(0.6),
      ),
    );
  }
}
