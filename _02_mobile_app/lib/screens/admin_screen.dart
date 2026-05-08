import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../app/state.dart';
import '../config/api_config.dart';
import '../models/app_user.dart';
import '../services/storage_service.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import '../theme/dt_widgets.dart';
import '../widgets/form_sheets.dart';
import '../widgets/machine_image.dart';
import '../widgets/user_avatar.dart';

/// Admin screen — adapts to the signed-in role.
///
/// - Admin: full access (Settings + Firebase + Machine + User panels).
/// - Engineer: Settings panel is read-only; only sees app behavior.
/// - Viewer: read-only on app behavior; gets a "view-only" notice.
class AdminScreen extends StatelessWidget {
  const AdminScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final perms = app.permissions;

    final canManage = perms.canManageUsers; // admin only

    return Scaffold(
      backgroundColor: palette.canvas,
      appBar: appHeader(
        title: canManage ? 'Administration' : 'Settings',
        subtitle: canManage
            ? 'Users, machines, system status'
            : 'Read-only — sign in as Admin to manage',
        trailing: IconButton(
          tooltip: 'Sign out',
          onPressed: () => context.read<AppState>().logout(),
          icon: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: DTTokens.statusCritical.alphaF(0.10),
              borderRadius: BorderRadius.circular(DTTokens.radiusMd),
              border: Border.all(
                color: DTTokens.statusCritical.alphaF(0.25),
              ),
            ),
            child: const Icon(
              Icons.logout_rounded,
              color: DTTokens.statusCritical,
              size: 16,
            ),
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 90),
        children: [
          if (!canManage) ...[
            DtCard(
              child: Row(
                children: [
                  const Icon(Icons.lock_outlined,
                      color: DTTokens.statusWarning, size: 18),
                  const SizedBox(width: DTTokens.space12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'You are signed in as ${roleLabel(app.currentUser.role)}',
                          style: DTTokens.body(palette.textPrimary).copyWith(
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          perms.role == AppRole.engineer
                              ? 'Engineers can write reports and act on machines, but cannot manage users or change global settings.'
                              : 'Viewers have read-only access. No actions are available.',
                          style: DTTokens.caption(palette.textSecondary)
                              .copyWith(height: 1.4),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: DTTokens.space16),
          ],

          // 1. My profile — every signed-in user can update their avatar.
          //    This is the "user can upload their own profile photo" path.
          const _MyProfilePanel(),
          const SizedBox(height: DTTokens.space16),

          // 2. App behavior — light personal preferences. Switches are
          //    role-gated inside the panel.
          const _AppBehaviorPanel(),
          const SizedBox(height: DTTokens.space16),

          // 3. Emergency alerts overview — visible to admin and engineer.
          //    Always shown so the operator on duty has a single place
          //    to glance at the active emergency feed.
          if (perms.canViewAlerts) ...[
            const _EmergencyAlertsOverviewPanel(),
            const SizedBox(height: DTTokens.space16),
          ],

          if (canManage) ...[
            // 4. System status — Firebase / MQTT / Storage at a glance.
            const _SystemStatusPanel(),
            const SizedBox(height: DTTokens.space16),

            // 5. Machines management — admin-only mutations.
            const _MachineAdminPanel(),
            const SizedBox(height: DTTokens.space16),

            // 6. Users management — admin-only mutations.
            const _UserAdminPanel(),
            const SizedBox(height: DTTokens.space16),

            // 7. Backend notes — clearly mark Cloud-Function-required gaps.
            const _BackendNotesPanel(),
          ],
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// My profile panel — every user can manage their own avatar
// ─────────────────────────────────────────────────────────────────────────────

class _MyProfilePanel extends StatefulWidget {
  const _MyProfilePanel();
  @override
  State<_MyProfilePanel> createState() => _MyProfilePanelState();
}

class _MyProfilePanelState extends State<_MyProfilePanel> {
  bool _busy = false;
  double _progress = 0.0;
  String? _error;

  Future<void> _pickAndUpload(BuildContext context) async {
    final app = context.read<AppState>();
    try {
      final picker = ImagePicker();
      final picked = await picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1024,
        imageQuality: 85,
      );
      if (picked == null) return; // user cancelled — nothing to do

      // Validate immediately so the user sees the error without
      // waiting for an upload that would only fail.
      try {
        final size = await picked.length();
        validateImageFile(filename: picked.name, sizeBytes: size);
      } on ImageValidationError catch (e) {
        if (mounted) setState(() => _error = e.message);
        return;
      }

      // Honest fallback: without Firebase initialized, the upload would
      // hang forever. Surface a clear message instead.
      if (!app.firebaseEnabled) {
        if (mounted) {
          setState(() => _error =
              'Firebase Storage is not initialized — cannot upload right now.');
        }
        return;
      }

      if (mounted) {
        setState(() {
          _busy = true;
          _progress = 0.0;
          _error = null;
        });
      }

      final svc = StorageService();
      UploadResult result;
      if (kIsWeb) {
        final bytes = await picked.readAsBytes();
        result = await svc.uploadUserAvatarBytes(
          uid: app.currentUser.id,
          bytes: bytes,
          filename: picked.name,
          onProgress: (p) {
            if (mounted) setState(() => _progress = p);
          },
        );
      } else {
        result = await svc.uploadUserAvatar(
          uid: app.currentUser.id,
          file: File(picked.path),
          onProgress: (p) {
            if (mounted) setState(() => _progress = p);
          },
        );
      }
      await app.setMyPhotoUrl(
        photoUrl: result.url,
        photoStoragePath: result.path,
      );
    } on TimeoutException {
      if (mounted) {
        setState(() => _error =
            'Upload timed out. Please check your connection or '
            'Firebase Storage rules.');
      }
    } on StorageUploadError catch (e) {
      if (mounted) setState(() => _error = e.message);
    } catch (e) {
      if (mounted) setState(() => _error = 'Upload failed: $e');
    } finally {
      // Defensive: always clear loading state. This is the central
      // hang-fix — previously a thrown error inside the try block
      // could leave _busy=true forever.
      if (mounted) {
        setState(() {
          _busy = false;
          _progress = 0.0;
        });
      }
    }
  }

  Future<void> _removePhoto(BuildContext context) async {
    final app = context.read<AppState>();
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Remove your profile photo?'),
        content: const Text(
          'Your photo will be replaced with your initials. The image '
          'is also deleted from Firebase Storage.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: DTTokens.statusCritical,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );
    if (ok != true) return;
    await app.setMyPhotoUrl(photoUrl: null, photoStoragePath: null);
  }

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final me = app.currentUser;
    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'My profile',
            style: DTTokens.h3(palette.textPrimary),
          ),
          const SizedBox(height: DTTokens.space12),
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              UserAvatar(user: me, size: 56),
              const SizedBox(width: DTTokens.space16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      me.name.isEmpty ? me.email : me.name,
                      style: DTTokens.body(palette.textPrimary)
                          .copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      '${me.email}  ·  ${roleLabel(me.role).toUpperCase()}',
                      style: DTTokens.caption(palette.textTertiary),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: DTTokens.space12),
          Row(
            children: [
              OutlinedButton.icon(
                onPressed: _busy ? null : () => _pickAndUpload(context),
                icon: const Icon(Icons.photo_library_outlined, size: 16),
                label: Text(me.photoUrl == null
                    ? 'Add profile photo'
                    : 'Change profile photo'),
              ),
              if (me.photoUrl != null) ...[
                const SizedBox(width: 8),
                TextButton.icon(
                  onPressed: _busy ? null : () => _removePhoto(context),
                  icon: const Icon(Icons.delete_outline,
                      size: 16, color: DTTokens.statusCritical),
                  label: const Text(
                    'Remove',
                    style: TextStyle(color: DTTokens.statusCritical),
                  ),
                ),
              ],
            ],
          ),
          if (_busy) ...[
            const SizedBox(height: 10),
            LinearProgressIndicator(
              value: _progress > 0 ? _progress : null,
            ),
            const SizedBox(height: 4),
            Text(
              'Uploading… ${(_progress * 100).toStringAsFixed(0)}%',
              style: DTTokens.caption(palette.textTertiary),
            ),
          ],
          if (_error != null) ...[
            const SizedBox(height: 8),
            Text(
              _error!,
              style: DTTokens.caption(DTTokens.statusCritical),
            ),
          ],
          const SizedBox(height: 6),
          Text(
            'Optional. Photos are stored in Firebase Storage and only your initials show if you skip it.',
            style: DTTokens.caption(palette.textTertiary)
                .copyWith(height: 1.4),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// App behavior panel
// ─────────────────────────────────────────────────────────────────────────────

class _AppBehaviorPanel extends StatelessWidget {
  const _AppBehaviorPanel();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final canChangeSettings = app.permissions.canChangeSettings;

    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'App behavior',
            style: DTTokens.h3(palette.textPrimary),
          ),
          const SizedBox(height: DTTokens.space12),
          // Theme can always be toggled — it's a personal preference, not
          // a global setting.
          _PanelSwitch(
            title: 'Dark mode',
            value: app.themeMode == ThemeMode.dark,
            onChanged: (v) => context.read<AppState>().toggleTheme(v),
          ),
          // Notifications + offline mode are global settings — admin-only.
          _PanelSwitch(
            title: 'Critical alert notifications',
            value: app.notificationsEnabled,
            onChanged: canChangeSettings
                ? (v) => context.read<AppState>().toggleNotifications(v)
                : null,
            disabled: !canChangeSettings,
          ),
          _PanelSwitch(
            title: 'Offline mode (last-known status)',
            value: app.offlineMode,
            onChanged: canChangeSettings
                ? (v) => context.read<AppState>().toggleOfflineMode(v)
                : null,
            disabled: !canChangeSettings,
          ),
        ],
      ),
    );
  }
}

class _PanelSwitch extends StatelessWidget {
  const _PanelSwitch({
    required this.title,
    required this.value,
    required this.onChanged,
    this.disabled = false,
  });

  final String title;
  final bool value;
  final ValueChanged<bool>? onChanged;
  final bool disabled;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Expanded(
            child: Text(
              title,
              style: DTTokens.body(disabled
                      ? palette.textTertiary
                      : palette.textPrimary)
                  .copyWith(
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
          Switch.adaptive(value: value, onChanged: onChanged),
        ],
      ),
    );
  }
}


// ─────────────────────────────────────────────────────────────────────────────
// System status panel — Firebase, MQTT, Storage at a glance
// ─────────────────────────────────────────────────────────────────────────────

class _SystemStatusPanel extends StatelessWidget {
  const _SystemStatusPanel();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);

    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'System status',
                  style: DTTokens.h3(palette.textPrimary),
                ),
              ),
              FilledButton.icon(
                onPressed: app.syncBusy
                    ? null
                    : () => context.read<AppState>().syncNow(),
                icon: Icon(
                  app.syncBusy
                      ? Icons.hourglass_bottom_rounded
                      : Icons.sync_rounded,
                  size: 14,
                ),
                label: Text(app.syncBusy ? 'Syncing…' : 'Sync now'),
              ),
            ],
          ),
          const SizedBox(height: DTTokens.space12),
          _StatusRow(
            label: 'Firebase Auth + Firestore',
            ok: app.firebaseEnabled,
            okText: app.firebaseStatus,
            offText: app.firebaseStatus,
          ),
          const SizedBox(height: 8),
          _StatusRow(
            label: 'Firebase Storage (machine images)',
            ok: app.firebaseEnabled,
            okText: 'Ready — admin can upload machine photos.',
            offText: 'Inactive — Firebase not initialized.',
          ),
          const SizedBox(height: 8),
          _StatusRow(
            label: 'MQTT telemetry stream',
            ok: app.firebaseEnabled,
            // We don't expose MQTT connection state on AppState publicly;
            // the dashboard owns that subscription. Be honest: report
            // configuration intent, not live socket state.
            okText: 'Configured. Dashboard subscribes on open.',
            offText: 'Demo simulator running — no broker connection.',
          ),
        ],
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  const _StatusRow({
    required this.label,
    required this.ok,
    required this.okText,
    required this.offText,
  });

  final String label;
  final bool ok;
  final String okText;
  final String offText;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    final color = ok ? DTTokens.statusHealthy : DTTokens.statusWarning;
    return Container(
      padding: const EdgeInsets.all(DTTokens.cardPadding),
      decoration: BoxDecoration(
        color: color.alphaF(0.06),
        borderRadius: BorderRadius.circular(DTTokens.radiusMd),
        border: Border.all(color: color.alphaF(0.20)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 8,
            height: 8,
            margin: const EdgeInsets.only(top: 5),
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: DTTokens.space12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: DTTokens.body(palette.textPrimary).copyWith(
                    fontWeight: FontWeight.w600,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  ok ? okText : offText,
                  style: DTTokens.caption(palette.textSecondary)
                      .copyWith(height: 1.4),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Emergency alerts overview — small read-only summary on Admin
// ─────────────────────────────────────────────────────────────────────────────

class _EmergencyAlertsOverviewPanel extends StatelessWidget {
  const _EmergencyAlertsOverviewPanel();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final alerts = app.alerts;

    final critical = alerts
        .where((a) =>
            a.color == DTTokens.statusCritical || a.color == DT.red)
        .length;
    final warning = alerts.length - critical;

    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                alerts.isEmpty
                    ? Icons.shield_outlined
                    : Icons.crisis_alert_rounded,
                size: 16,
                color: alerts.isEmpty
                    ? DTTokens.statusHealthy
                    : DTTokens.statusCritical,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Emergency alerts',
                  style: DTTokens.h3(palette.textPrimary),
                ),
              ),
              if (alerts.isEmpty)
                Text(
                  'All clear',
                  style: DTTokens.caption(DTTokens.statusHealthy)
                      .copyWith(fontWeight: FontWeight.w700),
                )
              else
                Text(
                  '${alerts.length} active',
                  style: DTTokens.caption(DTTokens.statusCritical)
                      .copyWith(fontWeight: FontWeight.w700),
                ),
            ],
          ),
          if (alerts.isNotEmpty) ...[
            const SizedBox(height: DTTokens.space12),
            Row(
              children: [
                _AlertChip(
                  label: 'Critical',
                  count: critical,
                  color: DTTokens.statusCritical,
                ),
                const SizedBox(width: 8),
                _AlertChip(
                  label: 'Warning',
                  count: warning,
                  color: DTTokens.statusWarning,
                ),
              ],
            ),
            const SizedBox(height: DTTokens.space12),
            Text(
              'Open the Alerts tab to triage. Tapping an alert opens the affected machine.',
              style: DTTokens.caption(palette.textTertiary)
                  .copyWith(height: 1.4),
            ),
          ] else ...[
            const SizedBox(height: 4),
            Text(
              'No emergency events right now. Critical incidents will appear here when machines exceed safe operating limits.',
              style: DTTokens.caption(palette.textSecondary)
                  .copyWith(height: 1.4),
            ),
          ],
        ],
      ),
    );
  }
}

class _AlertChip extends StatelessWidget {
  const _AlertChip({
    required this.label,
    required this.count,
    required this.color,
  });

  final String label;
  final int count;
  final Color color;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.alphaF(0.10),
        borderRadius: BorderRadius.circular(DTTokens.radiusSm),
        border: Border.all(color: color.alphaF(0.30)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '$count',
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w900,
              fontSize: 13,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w700,
              fontSize: 11,
              letterSpacing: 0.4,
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend notes — calls out gaps that are honestly out of scope today
// ─────────────────────────────────────────────────────────────────────────────

class _BackendNotesPanel extends StatelessWidget {
  const _BackendNotesPanel();

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.info_outline_rounded,
                  size: 14, color: DTTokens.accentPrimary),
              const SizedBox(width: 8),
              Text(
                'Backend notes',
                style: DTTokens.label(palette.textSecondary),
              ),
            ],
          ),
          const SizedBox(height: DTTokens.space12),
          const _NoteLine(
            text:
                'Creating Firebase Auth users from Admin requires a Cloud Function with the Firebase Admin SDK. Today the user form writes a Firestore-only profile.',
          ),
          const _NoteLine(
            text:
                'Push notifications for emergency alerts require Firebase Cloud Messaging backend setup.',
          ),
          const _NoteLine(
            text:
                'Live cross-device sync of reports/history is partial: writes flow to Firestore subcollections; the detail screen still reads from the local in-app cache.',
          ),
        ],
      ),
    );
  }
}

class _NoteLine extends StatelessWidget {
  const _NoteLine({required this.text});
  final String text;

  @override
  Widget build(BuildContext context) {
    final palette = DTPalette.of(context);
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.only(top: 5),
            child: Container(
              width: 4,
              height: 4,
              decoration: BoxDecoration(
                color: palette.textTertiary,
                shape: BoxShape.circle,
              ),
            ),
          ),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              text,
              style: DTTokens.caption(palette.textSecondary)
                  .copyWith(height: 1.45),
            ),
          ),
        ],
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Machine admin panel — admin only
// ─────────────────────────────────────────────────────────────────────────────

class _MachineAdminPanel extends StatelessWidget {
  const _MachineAdminPanel();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final perms = app.permissions;

    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Machine management',
                  style: DTTokens.h3(palette.textPrimary),
                ),
              ),
              if (perms.canAddMachines)
                FilledButton.icon(
                  onPressed: () => showMachineFormSheet(context),
                  icon: const Icon(Icons.add_rounded, size: 14),
                  label: const Text('Add machine'),
                ),
            ],
          ),
          const SizedBox(height: DTTokens.space12),
          ...app.machines.asMap().entries.map((entry) {
            final i = entry.key;
            final machine = entry.value;
            final isLast = i == app.machines.length - 1;
            return Padding(
              padding: EdgeInsets.only(bottom: isLast ? 0 : 8),
              child: Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: palette.surfaceElevated,
                  borderRadius: BorderRadius.circular(DTTokens.radiusSm),
                  border: Border.all(color: palette.borderSubtle),
                ),
                child: Row(
                  children: [
                    SizedBox(
                      width: 44,
                      height: 44,
                      child: MachineImage(
                        machine: machine,
                        fit: BoxFit.cover,
                        borderRadius:
                            BorderRadius.circular(DTTokens.radiusSm),
                      ),
                    ),
                    const SizedBox(width: DTTokens.space12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            machine.name,
                            style: DTTokens.body(palette.textPrimary)
                                .copyWith(fontWeight: FontWeight.w600),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 2),
                          Text(
                            '${machine.code} · ${machine.location}',
                            style: DTTokens.caption(palette.textTertiary),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                    if (perms.canEditMachines)
                      IconButton(
                        tooltip: 'Edit',
                        onPressed: () =>
                            showMachineFormSheet(context, machine: machine),
                        icon: const Icon(
                          Icons.edit_outlined,
                          color: DTTokens.accentLive,
                          size: 18,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                    if (perms.canDeleteMachines)
                      IconButton(
                        tooltip: 'Delete',
                        onPressed: () =>
                            _confirmDeleteMachine(context, machine.id, machine.name),
                        icon: const Icon(
                          Icons.delete_outline_rounded,
                          color: DTTokens.statusCritical,
                          size: 18,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                  ],
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  Future<void> _confirmDeleteMachine(
    BuildContext context,
    String id,
    String name,
  ) async {
    final palette = DTPalette.of(context);
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: palette.surface,
        title: const Text('Delete machine?'),
        content: Text(
          '$name will be removed from the factory roster. This cannot be undone.',
          style: DTTokens.body(palette.textSecondary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: DTTokens.statusCritical,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (ok == true && context.mounted) {
      context.read<AppState>().deleteMachine(id);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// User admin panel — admin only
// ─────────────────────────────────────────────────────────────────────────────

class _UserAdminPanel extends StatelessWidget {
  const _UserAdminPanel();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);
    final perms = app.permissions;
    // Only active, non-deleted users appear in the admin list.
    final visibleUsers =
        app.users.where((u) => u.isActive).toList(growable: false);

    return DtCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'User management',
                  style: DTTokens.h3(palette.textPrimary),
                ),
              ),
              if (perms.canManageUsers)
                FilledButton.icon(
                  onPressed: () => showUserFormSheet(context),
                  icon: const Icon(Icons.person_add_alt_1_rounded, size: 14),
                  label: const Text('Add user'),
                ),
            ],
          ),
          const SizedBox(height: DTTokens.space12),
          // visibleUsers filters out soft-deleted / inactive accounts.
          ...visibleUsers.asMap().entries.map<Widget>((entry) {
            final i = entry.key;
            final user = entry.value;
            final isLast = i == visibleUsers.length - 1;
            final isSelf = user.id == app.currentUser.id;
            return Padding(
              padding: EdgeInsets.only(bottom: isLast ? 0 : 8),
              child: Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: palette.surfaceElevated,
                  borderRadius: BorderRadius.circular(DTTokens.radiusSm),
                  border: Border.all(color: palette.borderSubtle),
                ),
                child: Row(
                  children: [
                    UserAvatar(user: user, size: 36),
                    const SizedBox(width: DTTokens.space12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            user.name,
                            style: DTTokens.body(palette.textPrimary)
                                .copyWith(fontWeight: FontWeight.w600),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          const SizedBox(height: 2),
                          Text(
                            '${user.email} · ${roleLabel(user.role).toUpperCase()}',
                            style: DTTokens.caption(palette.textTertiary),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                        ],
                      ),
                    ),
                    if (perms.canManageUsers)
                      IconButton(
                        tooltip: 'Edit',
                        onPressed: () =>
                            showUserFormSheet(context, user: user),
                        icon: const Icon(
                          Icons.edit_outlined,
                          color: DTTokens.accentLive,
                          size: 18,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                    if (perms.canDeleteUsers)
                      IconButton(
                        tooltip: isSelf ? 'Cannot delete yourself' : 'Delete',
                        onPressed: isSelf
                            ? null
                            : () => _confirmDeleteUser(
                                  context,
                                  user,
                                ),
                        icon: Icon(
                          Icons.delete_outline_rounded,
                          size: 18,
                          color: isSelf
                              ? palette.textTertiary.alphaF(0.4)
                              : DTTokens.statusCritical,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                  ],
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  Future<void> _confirmDeleteUser(
    BuildContext context,
    AppUser target,
  ) async {
    final app = context.read<AppState>();
    final palette = DTPalette.of(context);

    // ── Last-admin guard (fast, client-side) ────────────────────────────
    if (target.role == AppRole.admin) {
      final adminCount =
          app.users.where((u) => u.role == AppRole.admin).length;
      if (adminCount <= 1) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Cannot remove the last admin account.'),
          ),
        );
        return;
      }
    }

    // ── Confirmation dialog ─────────────────────────────────────────────
    final ok = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        backgroundColor: palette.surface,
        title: const Text('Remove user?'),
        content: Text(
          '${target.name} will lose access immediately. '
          'Their Firebase Auth account will be deleted.',
          style: DTTokens.body(palette.textSecondary),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            style: FilledButton.styleFrom(
              backgroundColor: DTTokens.statusCritical,
            ),
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );

    if (ok != true || !context.mounted) return;

    // ── Loading indicator ───────────────────────────────────────────────
    showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (_) => AlertDialog(
        backgroundColor: palette.surface,
        content: Row(
          children: [
            const CircularProgressIndicator(),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                'Removing ${target.name}…',
                style: DTTokens.body(palette.textPrimary),
              ),
            ),
          ],
        ),
      ),
    );

    // ── Call backend ────────────────────────────────────────────────────
    final error = await _deleteUserViaBackend(target.id);

    if (!context.mounted) return;
    Navigator.of(context).pop(); // close loading dialog

    if (error != null) {
      // ── Error dialog ──────────────────────────────────────────────────
      showDialog<void>(
        context: context,
        builder: (_) => AlertDialog(
          backgroundColor: palette.surface,
          title: const Text('Could not remove user'),
          content: Text(error, style: DTTokens.body(palette.textSecondary)),
          actions: [
            FilledButton(
              onPressed: () => Navigator.of(context).pop(),
              child: const Text('OK'),
            ),
          ],
        ),
      );
      return;
    }

    // ── Success: update local state ─────────────────────────────────────
    app.deleteUser(target.id);
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('${target.name} removed.')),
    );
  }

  /// POST /admin/delete-user — returns null on success, error string on failure.
  Future<String?> _deleteUserViaBackend(String uid) async {
    try {
      final response = await http
          .post(
            Uri.parse(
              '${ApiConfig.baseUrl}${ApiConfig.deleteUserEndpoint}',
            ),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({'uid': uid, 'softDelete': true}),
          )
          .timeout(ApiConfig.deleteUserTimeout);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        // 'partial' means Auth was deleted but Firestore update failed.
        // Treat as success for the UI; Firestore will be cleaned up by the
        // sync service's fallback soft-delete on the next deleteUser() call.
        if (data['status'] == 'partial') {
          return 'User removed from Authentication but Firestore update '
              'failed: ${data['warning']}. The account is inaccessible.';
        }
        return null;
      }

      String detail = 'Server error ${response.statusCode}.';
      try {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        detail = (body['detail'] as String?) ?? detail;
      } catch (_) {}
      return detail;
    } on TimeoutException {
      return 'Request timed out. Check the backend is running.';
    } catch (e) {
      return 'Could not reach the backend: $e';
    }
  }
}
