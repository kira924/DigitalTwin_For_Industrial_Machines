import '../models/machine.dart';

/// How fresh telemetry must be to count as "live" for the automatic
/// status rule. Tuned to the publisher's 1 Hz cadence and the freshness
/// thresholds elsewhere in the app (see `dataStateFromTimestamp`).
const Duration kLiveTelemetryWindow = Duration(seconds: 30);

/// Resolved status the UI should show for a machine, derived purely
/// from inputs (last telemetry time + report existence) — no manual
/// Start/Stop toggle. Used by the dashboard, machine cards, and
/// machine detail to keep the displayed status consistent.
///
/// Rule:
///   - has live telemetry          → running
///   - no live telemetry, no report → fault
///   - no live telemetry, has report → maintenance
///
/// "Has live telemetry" means the most recent payload arrived within
/// [kLiveTelemetryWindow]. A machine that has never received
/// telemetry counts as "no live telemetry".
class MachineStatusResolver {
  const MachineStatusResolver._();

  /// Compute the automatic status for a machine.
  ///
  /// [lastSeen] — when the most recent telemetry payload arrived for
  /// this machine, or `null` if it has never received any.
  /// [hasReport] — whether the machine has at least one open report
  /// in the reports collection.
  /// [now] — overridable for testing; defaults to `DateTime.now()`.
  static MachineStatus resolve({
    required DateTime? lastSeen,
    required bool hasReport,
    DateTime? now,
  }) {
    final t = now ?? DateTime.now();
    final live = lastSeen != null &&
        t.difference(lastSeen) <= kLiveTelemetryWindow;

    if (live) return MachineStatus.running;
    if (hasReport) return MachineStatus.maintenance;
    return MachineStatus.fault;
  }

  /// Human-friendly status label used in UI badges and history events.
  static String label(MachineStatus status) {
    switch (status) {
      case MachineStatus.running:
        return 'Running';
      case MachineStatus.maintenance:
        return 'Under Maintenance';
      case MachineStatus.fault:
        return 'Fault';
      case MachineStatus.stopped:
        // Phase 3.6: "Stopped" is no longer a user-facing label.
        // The resolver never *produces* MachineStatus.stopped, but
        // map it to "Offline" defensively in case a legacy value
        // reaches this helper.
        return 'Offline';
      case MachineStatus.offline:
        return 'Offline';
    }
  }
}
