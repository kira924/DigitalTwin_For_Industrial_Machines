import 'dart:math';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../models/engine_telemetry_payload.dart';
import '../../providers/dashboard_providers.dart';

/// Layout shell for charts, KPIs, and sensor diagnostics.
class DashboardShell extends StatelessWidget {
  /// Creates the dashboard shell.
  const DashboardShell({super.key});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (BuildContext context, BoxConstraints constraints) {
        final double horizontalPadding = constraints.maxWidth >= 1200 ? 24 : 16;

        return SingleChildScrollView(
          padding: EdgeInsets.fromLTRB(horizontalPadding, 16, horizontalPadding, 24),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              _HeaderSection(),
              SizedBox(height: 16),
              _KeyMetricsSection(),
              SizedBox(height: 16),
              _SensorsSection(),
              SizedBox(height: 16),
              _RootCausesSection(),
            ],
          ),
        );
      },
    );
  }
}

class _HeaderSection extends StatelessWidget {
  const _HeaderSection();

  @override
  Widget build(BuildContext context) {
    return const _GlassPanel(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'Industrial Digital Twin Dashboard',
            style: TextStyle(
              color: Colors.white,
              fontSize: 21,
              fontWeight: FontWeight.w800,
            ),
          ),
          SizedBox(height: 8),
          Text(
            'Real-time telemetry, predictive health, and AI root-cause insights',
            style: TextStyle(color: Color(0xFF9FB0C3), fontSize: 13),
          ),
        ],
      ),
    );
  }
}

class _KeyMetricsSection extends StatelessWidget {
  const _KeyMetricsSection();

  @override
  Widget build(BuildContext context) {
    return Consumer(
      builder: (BuildContext context, WidgetRef ref, Widget? child) {
        final EngineTelemetryPayload? data = ref.watch(latestTelemetryProvider);
        if (data == null) {
          return const _WaitingTelemetryCard();
        }

        final Color rulColor = _rulColor(data.predictedRul);

        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: <Widget>[
            _MetricCard(
              title: 'Engine ID',
              value: data.engineId.toString(),
              width: 260,
            ),
            _MetricCard(
              title: 'Current Cycle',
              valueBuilder: (BuildContext context) => _AnimatedNumber(
                value: data.currentCycle.toDouble(),
                fractionDigits: 0,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 28,
                  fontWeight: FontWeight.w700,
                ),
              ),
              width: 260,
            ),
            _MetricCard(
              title: 'Predicted RUL',
              valueBuilder: (BuildContext context) => _AnimatedNumber(
                value: data.predictedRul,
                fractionDigits: 2,
                style: TextStyle(
                  color: rulColor,
                  fontSize: 28,
                  fontWeight: FontWeight.w800,
                ),
              ),
              width: 260,
            ),
          ],
        );
      },
    );
  }
}

class _SensorsSection extends StatelessWidget {
  const _SensorsSection();

  @override
  Widget build(BuildContext context) {
    return Consumer(
      builder: (BuildContext context, WidgetRef ref, Widget? child) {
        final EngineTelemetryPayload? data = ref.watch(latestTelemetryProvider);
        if (data == null) {
          return const SizedBox.shrink();
        }

        final List<MapEntry<String, double>> sensors =
            data.currentSensorReadings.entries.take(4).toList();

        return _GlassPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              const Text(
                'Sensors Overview',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 12),
              Wrap(
                spacing: 10,
                runSpacing: 10,
                children: sensors
                    .map(
                      (MapEntry<String, double> entry) => _SensorMiniCard(
                        sensorName: entry.key,
                        sensorValue: entry.value,
                      ),
                    )
                    .toList(),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _RootCausesSection extends StatelessWidget {
  const _RootCausesSection();

  @override
  Widget build(BuildContext context) {
    return Consumer(
      builder: (BuildContext context, WidgetRef ref, Widget? child) {
        final EngineTelemetryPayload? data = ref.watch(latestTelemetryProvider);
        if (data == null) {
          return const SizedBox.shrink();
        }

        final List<MapEntry<String, double>> topCauses = data.aiRootCauses.entries
            .toList()
          ..sort((MapEntry<String, double> a, MapEntry<String, double> b) {
            return b.value.compareTo(a.value);
          });
        final List<MapEntry<String, double>> topThree = topCauses.take(3).toList();

        return _GlassPanel(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: <Widget>[
              const Text(
                'AI Root Causes (SHAP Top 3)',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 16),
              SizedBox(
                height: 260,
                child: BarChart(
                  BarChartData(
                    alignment: BarChartAlignment.spaceAround,
                    maxY: _chartMaxY(topThree),
                    titlesData: FlTitlesData(
                      topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      rightTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false),
                      ),
                      leftTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          reservedSize: 36,
                          interval: 5,
                          getTitlesWidget: (double value, TitleMeta meta) {
                            return Text(
                              value.toStringAsFixed(0),
                              style: const TextStyle(
                                color: Color(0xFF8FA2B8),
                                fontSize: 11,
                              ),
                            );
                          },
                        ),
                      ),
                      bottomTitles: AxisTitles(
                        sideTitles: SideTitles(
                          showTitles: true,
                          getTitlesWidget: (double value, TitleMeta meta) {
                            final int index = value.toInt();
                            if (index < 0 || index >= topThree.length) {
                              return const SizedBox.shrink();
                            }
                            return Padding(
                              padding: const EdgeInsets.only(top: 8),
                              child: Text(
                                topThree[index].key,
                                style: const TextStyle(
                                  color: Color(0xFF8FA2B8),
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
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
                      getDrawingHorizontalLine: (double value) {
                        return const FlLine(
                          color: Color(0xFF233447),
                          strokeWidth: 1,
                        );
                      },
                    ),
                    borderData: FlBorderData(show: false),
                    barGroups: List<BarChartGroupData>.generate(
                      topThree.length,
                      (int index) {
                        final MapEntry<String, double> entry = topThree[index];
                        return BarChartGroupData(
                          x: index,
                          barRods: <BarChartRodData>[
                            BarChartRodData(
                              toY: entry.value,
                              width: 28,
                              borderRadius: BorderRadius.circular(8),
                              gradient: const LinearGradient(
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                                colors: <Color>[
                                  Color(0xFFFFA24C),
                                  Color(0xFFFF5A5A),
                                ],
                              ),
                            ),
                          ],
                        );
                      },
                    ),
                  ),
                  swapAnimationDuration: Duration.zero,
                  swapAnimationCurve: Curves.easeInOutCubic,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _WaitingTelemetryCard extends StatelessWidget {
  const _WaitingTelemetryCard();

  @override
  Widget build(BuildContext context) {
    return const _GlassPanel(
      height: 220,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: <Widget>[
            CircularProgressIndicator(color: Color(0xFF67D7FF)),
            SizedBox(height: 12),
            Text(
              'Waiting for Live Engine Telemetry...',
              style: TextStyle(color: Color(0xFFB3C0CF), fontSize: 14),
            ),
          ],
        ),
      ),
    );
  }
}

class _MetricCard extends StatelessWidget {
  const _MetricCard({
    required this.title,
    this.value,
    this.valueBuilder,
    this.width = 250,
  });

  final String title;
  final String? value;
  final WidgetBuilder? valueBuilder;
  final double width;

  @override
  Widget build(BuildContext context) {
    return _GlassPanel(
      width: width,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            title,
            style: const TextStyle(
              color: Color(0xFF9AA8B8),
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 10),
          if (valueBuilder != null)
            valueBuilder!(context)
          else
            Text(
              value ?? '-',
              style: const TextStyle(
                color: Colors.white,
                fontSize: 28,
                fontWeight: FontWeight.w700,
              ),
            ),
        ],
      ),
    );
  }
}

class _SensorMiniCard extends ConsumerWidget {
  const _SensorMiniCard({required this.sensorName, required this.sensorValue});

  final String sensorName;
  final double sensorValue;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final List<double> history = ref.watch(
      sensorHistoryProvider.select(
        (Map<String, List<double>> values) =>
            values[sensorName] ?? const <double>[],
      ),
    );

    return _GlassPanel(
      width: 190,
      contentPadding: const EdgeInsets.all(12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            sensorName.toUpperCase(),
            style: const TextStyle(
              color: Color(0xFF8FA2B8),
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          _AnimatedNumber(
            value: sensorValue,
            fractionDigits: 4,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 44,
            child: _SensorSparkline(history: history),
          ),
        ],
      ),
    );
  }
}

class _SensorSparkline extends StatelessWidget {
  const _SensorSparkline({required this.history});

  final List<double> history;

  @override
  Widget build(BuildContext context) {
    final List<FlSpot> spots = _toSpots(history);
    if (spots.length < 2) {
      return const SizedBox.shrink();
    }

    return LineChart(
      LineChartData(
        minX: 0,
        maxX: (spots.length - 1).toDouble(),
        minY: _minY(spots),
        maxY: _maxY(spots),
        lineBarsData: <LineChartBarData>[
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.28,
            barWidth: 2.2,
            color: const Color(0xFF6BD7FF),
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              gradient: const LinearGradient(
                begin: Alignment.topCenter,
                end: Alignment.bottomCenter,
                colors: <Color>[
                  Color(0x886BD7FF),
                  Color(0x006BD7FF),
                ],
              ),
            ),
          ),
        ],
        gridData: const FlGridData(show: false),
        borderData: FlBorderData(show: false),
        titlesData: const FlTitlesData(
          leftTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          topTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
          bottomTitles: AxisTitles(sideTitles: SideTitles(showTitles: false)),
        ),
        lineTouchData: const LineTouchData(enabled: false),
      ),
      duration: Duration.zero,
      curve: Curves.easeInOutCubic,
    );
  }
}

class _AnimatedNumber extends StatelessWidget {
  const _AnimatedNumber({
    required this.value,
    required this.fractionDigits,
    required this.style,
  });

  final double value;
  final int fractionDigits;
  final TextStyle style;

  @override
  Widget build(BuildContext context) {
    return Text(
      value.toStringAsFixed(fractionDigits),
      style: style,
    );
  }
}

class _GlassPanel extends StatelessWidget {
  const _GlassPanel({
    required this.child,
    this.width,
    this.height,
    this.contentPadding = const EdgeInsets.all(16),
  });

  final Widget child;
  final double? width;
  final double? height;
  final EdgeInsetsGeometry contentPadding;

  @override
  Widget build(BuildContext context) {
    return Container(
      width: width,
      height: height,
      padding: contentPadding,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: <Color>[Color(0xFF172331), Color(0xFF101A24)],
        ),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: const Color(0xFF2B3B4F)),
        boxShadow: const <BoxShadow>[
          BoxShadow(
            color: Color(0x66050A10),
            blurRadius: 26,
            offset: Offset(0, 14),
          ),
          BoxShadow(
            color: Color(0x401A80B8),
            blurRadius: 16,
            offset: Offset(0, 0),
          ),
        ],
      ),
      child: child,
    );
  }
}

Color _rulColor(double rul) {
  if (rul < 50) {
    return const Color(0xFFFF5252);
  }
  if (rul < 100) {
    return const Color(0xFFFFB74D);
  }
  return const Color(0xFF66BB6A);
}

double _chartMaxY(List<MapEntry<String, double>> values) {
  if (values.isEmpty) {
    return 10;
  }
  final double maxValue =
      values.map((MapEntry<String, double> e) => e.value).reduce(max);
  return (maxValue * 1.2).clamp(5, 1000);
}

List<FlSpot> _toSpots(List<double> values) {
  return List<FlSpot>.generate(
    values.length,
    (int index) => FlSpot(index.toDouble(), values[index]),
  );
}

double _minY(List<FlSpot> spots) {
  final double minValue = spots.map((FlSpot point) => point.y).reduce(min);
  return minValue - 0.01;
}

double _maxY(List<FlSpot> spots) {
  final double maxValue = spots.map((FlSpot point) => point.y).reduce(max);
  return maxValue + 0.01;
}