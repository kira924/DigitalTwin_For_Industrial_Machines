import 'package:cloud_firestore/cloud_firestore.dart';

/// User identity, role, and permission gating.
///
/// Roles are intentionally limited to 3:
///   - admin: full control (machines, users, settings, reports, deletes)
///   - engineer: read everything; can write reports/notes; can act on
///     machines (start/stop/maintenance/issue) but not delete machines
///     or manage users.
///   - viewer: strictly read-only. No mutations of any kind.
///
/// Old roles (`operator`, `maintenance`) are migrated to `engineer` on
/// load — see [roleFromStoredName].
enum AppRole {
  admin,
  engineer,
  viewer,
}

/// Email allowlists used to seed a user's role on first login.
///
/// Only emails listed here get elevated roles automatically. Anything
/// else defaults to viewer. Never default unknown users to admin.
const Set<String> kAdminEmails = {
  'belal19lol@gmail.com',
  'admin2@example.com',
};

const Set<String> kEngineerEmails = {
  'engineer@example.com',
};

const Set<String> kViewerEmails = {
  'viewer@example.com',
};

/// Resolves the role to assign when a Firebase Auth user signs in for
/// the first time and there is no `users/{uid}` document yet.
AppRole resolveSeedRole(String email) {
  final n = email.trim().toLowerCase();
  if (kAdminEmails.contains(n)) return AppRole.admin;
  if (kEngineerEmails.contains(n)) return AppRole.engineer;
  if (kViewerEmails.contains(n)) return AppRole.viewer;
  // Unknown emails default to viewer. Never admin.
  return AppRole.viewer;
}

/// Maps a stored role name (which may be a legacy value) to the current
/// 3-role taxonomy. Unknown values fall back to `viewer` (the safest,
/// lowest-privilege role).
AppRole roleFromStoredName(String? raw) {
  switch ((raw ?? '').toLowerCase()) {
    case 'admin':
      return AppRole.admin;
    case 'engineer':
      return AppRole.engineer;
    // Legacy roles → engineer
    case 'operator':
    case 'maintenance':
      return AppRole.engineer;
    case 'viewer':
    default:
      return AppRole.viewer;
  }
}

/// Human-readable label for a role.
String roleLabel(AppRole role) {
  switch (role) {
    case AppRole.admin:
      return 'Admin';
    case AppRole.engineer:
      return 'Engineer';
    case AppRole.viewer:
      return 'Viewer';
  }
}

/// Permission flags. The single source of truth for "can this user do X?".
/// Every action button on every screen consults this before rendering or
/// executing — viewer cannot mutate anything anywhere.
///
/// A disabled user (`!isActive`) gets effectively viewer-level permissions
/// regardless of their stored role, so deactivation is enforceable in
/// software (Firestore rules should enforce it server-side too).
class Permissions {
  final AppRole role;
  final bool isActive;
  const Permissions(this.role, {this.isActive = true});

  factory Permissions.of(AppUser user) =>
      Permissions(user.role, isActive: user.isActive);

  AppRole get _effectiveRole => isActive ? role : AppRole.viewer;

  // ── Machine fleet ────────────────────────────────────────────────────────
  bool get canAddMachines => _effectiveRole == AppRole.admin;
  bool get canEditMachines => _effectiveRole == AppRole.admin;
  bool get canDeleteMachines => _effectiveRole == AppRole.admin;

  /// Only admin can upload/change/delete machine images.
  bool get canUploadMachineImages => _effectiveRole == AppRole.admin;

  // ── Machine actions on the detail page ───────────────────────────────────
  bool get canActOnMachine =>
      _effectiveRole == AppRole.admin || _effectiveRole == AppRole.engineer;

  // ── Reports & engineering notes ──────────────────────────────────────────
  bool get canWriteReports =>
      _effectiveRole == AppRole.admin || _effectiveRole == AppRole.engineer;

  bool get canEditAnyReport => _effectiveRole == AppRole.admin;
  bool get canDeleteReports => _effectiveRole == AppRole.admin;

  bool canEditOwnReport(String authorUserId, String currentUserId) {
    if (_effectiveRole == AppRole.admin) return true;
    if (_effectiveRole == AppRole.engineer && authorUserId == currentUserId) {
      return true;
    }
    return false;
  }

  /// Engineers can attach photos to their own reports; admins can attach
  /// photos to any report.
  bool get canAttachReportPhotos =>
      _effectiveRole == AppRole.admin || _effectiveRole == AppRole.engineer;

  // ── Alerts ───────────────────────────────────────────────────────────────
  bool get canAcknowledgeAlerts =>
      _effectiveRole == AppRole.admin || _effectiveRole == AppRole.engineer;

  // ── User management ──────────────────────────────────────────────────────
  bool get canManageUsers => _effectiveRole == AppRole.admin;
  bool get canDeleteUsers => _effectiveRole == AppRole.admin;
  bool get canActivateUsers => _effectiveRole == AppRole.admin;

  // ── Settings ─────────────────────────────────────────────────────────────
  bool get canChangeSettings => _effectiveRole == AppRole.admin;

  // ── Read access ──────────────────────────────────────────────────────────
  // All roles can read everything; explicit getters so callers don't need
  // to know the role hierarchy.
  bool get canViewDashboard => true;
  bool get canViewMachines => true;
  bool get canViewMachineDetail => true;
  bool get canViewAlerts => true;
  bool get canViewHistory => true;
  bool get canViewReports => true;
}

/// User profile, mirrored from Firestore `users/{uid}`.
///
/// `id` holds the Firebase Auth uid for users created post-Phase-2.
/// Legacy demo users have `u-XX` ids; treated as their own uid.
class AppUser {
  final String id;
  final String name;
  final String email;
  final AppRole role;
  final String assignedArea;
  final bool isActive;

  /// Optional user avatar URL (uploaded to Firebase Storage).
  /// When null/empty, the UI shows initials in a coloured circle.
  final String? photoUrl;

  /// Firebase Storage object path for the avatar, e.g.
  /// `user-avatars/{uid}/{ts}.jpg`. Only the path can address the
  /// blob for deletion — the URL is one-way.
  final String? photoStoragePath;

  // Audit fields
  final DateTime? createdAt;
  final DateTime? updatedAt;
  final DateTime? lastLoginAt;
  final String? createdBy;

  const AppUser({
    required this.id,
    required this.name,
    required this.email,
    required this.role,
    required this.assignedArea,
    this.isActive = true,
    this.photoUrl,
    this.photoStoragePath,
    this.createdAt,
    this.updatedAt,
    this.lastLoginAt,
    this.createdBy,
  });

  /// Convenience: legacy code asks for `enabled` — keep both names mapped
  /// to the same underlying field.
  bool get enabled => isActive;

  AppUser copyWith({
    String? id,
    String? name,
    String? email,
    AppRole? role,
    String? assignedArea,
    bool? isActive,
    String? photoUrl,
    bool clearPhotoUrl = false,
    String? photoStoragePath,
    bool clearPhotoStoragePath = false,
    DateTime? createdAt,
    DateTime? updatedAt,
    DateTime? lastLoginAt,
    String? createdBy,
  }) {
    return AppUser(
      id: id ?? this.id,
      name: name ?? this.name,
      email: email ?? this.email,
      role: role ?? this.role,
      assignedArea: assignedArea ?? this.assignedArea,
      isActive: isActive ?? this.isActive,
      photoUrl: clearPhotoUrl ? null : (photoUrl ?? this.photoUrl),
      photoStoragePath: clearPhotoStoragePath
          ? null
          : (photoStoragePath ?? this.photoStoragePath),
      createdAt: createdAt ?? this.createdAt,
      updatedAt: updatedAt ?? this.updatedAt,
      lastLoginAt: lastLoginAt ?? this.lastLoginAt,
      createdBy: createdBy ?? this.createdBy,
    );
  }

  Map<String, dynamic> toMap() {
    return <String, dynamic>{
      'id': id,
      'name': name,
      'email': email.trim().toLowerCase(),
      'role': role.name,
      'assignedArea': assignedArea,
      'isActive': isActive,
      // Keep `enabled` as a back-compat field; some older callers read it.
      'enabled': isActive,
      'photoUrl': photoUrl,
      'photoStoragePath': photoStoragePath,
      'createdAt': createdAt?.toIso8601String(),
      'updatedAt': updatedAt?.toIso8601String(),
      'lastLoginAt': lastLoginAt?.toIso8601String(),
      'createdBy': createdBy,
    };
  }

  factory AppUser.fromMap(Map<String, dynamic> map, {String? fallbackId}) {
    return AppUser(
      id: (map['id'] ?? fallbackId ?? '').toString(),
      name: (map['name'] ?? '').toString(),
      email: (map['email'] ?? '').toString(),
      role: roleFromStoredName(map['role']?.toString()),
      assignedArea: (map['assignedArea'] ?? '').toString(),
      // Both new and legacy field names accepted.
      isActive: (map['isActive'] ?? map['enabled'] ?? true) == true,
      photoUrl: (map['photoUrl'] as String?)?.isNotEmpty == true
          ? map['photoUrl'] as String
          : null,
      photoStoragePath:
          (map['photoStoragePath'] as String?)?.isNotEmpty == true
              ? map['photoStoragePath'] as String
              : null,
      createdAt: _parseTime(map['createdAt']),
      updatedAt: _parseTime(map['updatedAt']),
      lastLoginAt: _parseTime(map['lastLoginAt']),
      createdBy: map['createdBy']?.toString(),
    );
  }
}

DateTime? _parseTime(Object? value) {
  if (value == null) return null;
  if (value is DateTime) return value;
  if (value is Timestamp) return value.toDate();
  if (value is String && value.isNotEmpty) {
    return DateTime.tryParse(value);
  }
  return null;
}
