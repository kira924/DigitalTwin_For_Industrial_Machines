import 'package:flutter/material.dart';

import '../theme/dt_colors.dart';

class DTPoint {
  final String x;
  final double y;
  const DTPoint(this.x, this.y);
}

class SimpleLineChart extends StatelessWidget {
  final List<DTPoint> points;
  final double minY;
  final double maxY;
  final Color lineColor;

  const SimpleLineChart({
    super.key,
    required this.points,
    required this.minY,
    required this.maxY,
    required this.lineColor,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _LinePainter(points, minY, maxY, lineColor, fill: false),
    );
  }
}

class AreaChart extends StatelessWidget {
  final List<DTPoint> points;
  final double minY;
  final double maxY;
  final Color lineColor;

  const AreaChart({
    super.key,
    required this.points,
    required this.minY,
    required this.maxY,
    required this.lineColor,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _LinePainter(points, minY, maxY, lineColor, fill: true),
    );
  }
}

class _LinePainter extends CustomPainter {
  final List<DTPoint> points;
  final double minY;
  final double maxY;
  final Color color;
  final bool fill;

  _LinePainter(this.points, this.minY, this.maxY, this.color, {required this.fill});

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 2) return;

    const padding = EdgeInsets.fromLTRB(10, 10, 10, 16);
    final w = size.width - padding.left - padding.right;
    final h = size.height - padding.top - padding.bottom;

    final gridPaint = Paint()
      ..color = DT.border(0.35)
      ..strokeWidth = 1;

    for (int i = 0; i <= 4; i++) {
      final y = padding.top + (h * i / 4);
      canvas.drawLine(Offset(padding.left, y), Offset(padding.left + w, y), gridPaint);
    }

    Offset toPx(int i, double v) {
      final x = padding.left + (w * i / (points.length - 1));
      final t = ((v - minY) / (maxY - minY)).clamp(0.0, 1.0);
      final y = padding.top + h * (1 - t);
      return Offset(x, y);
    }

    final path = Path();
    for (int i = 0; i < points.length; i++) {
      final p = toPx(i, points[i].y);
      if (i == 0) {
        path.moveTo(p.dx, p.dy);
      } else {
        path.lineTo(p.dx, p.dy);
      }
    }

    if (fill) {
      final fillPath = Path.from(path)
        ..lineTo(padding.left + w, padding.top + h)
        ..lineTo(padding.left, padding.top + h)
        ..close();

      final shader = LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: [color.alphaF(0.30), color.alphaF(0.0)],
      ).createShader(Rect.fromLTWH(0, 0, size.width, size.height));

      canvas.drawPath(fillPath, Paint()..shader = shader);
    }

    final linePaint = Paint()
      ..color = color
      ..strokeWidth = 2.2
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    canvas.drawPath(path, linePaint);

    final dotPaint = Paint()..color = color;
    for (int i = 0; i < points.length; i++) {
      final p = toPx(i, points[i].y);
      canvas.drawCircle(p, 3.2, dotPaint);
      canvas.drawCircle(p, 6.5, Paint()..color = color.alphaF(0.12));
    }

    final labelStyle = TextStyle(color: DT.muted(0.50), fontSize: 11, fontWeight: FontWeight.w600);
    final idxs = {0, (points.length / 2).floor(), points.length - 1}.toList()..sort();
    for (final i in idxs) {
      final p = toPx(i, minY);
      final tp = TextPainter(
        text: TextSpan(text: points[i].x, style: labelStyle),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(p.dx - tp.width / 2, padding.top + h + 2));
    }
  }

  @override
  bool shouldRepaint(covariant _LinePainter oldDelegate) {
    if (oldDelegate.minY != minY ||
        oldDelegate.maxY != maxY ||
        oldDelegate.color != color ||
        oldDelegate.fill != fill ||
        oldDelegate.points.length != points.length) {
      return true;
    }

    for (int i = 0; i < points.length; i++) {
      if (oldDelegate.points[i].x != points[i].x || oldDelegate.points[i].y != points[i].y) {
        return true;
      }
    }

    return false;
  }
}
