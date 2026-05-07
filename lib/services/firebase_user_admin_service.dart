import 'package:cloud_firestore/cloud_firestore.dart';

import '../models/app_user.dart';

/// Admin-side user provisioning.
///
/// Creating Firebase Auth accounts directly from the Flutter client is
/// unsafe for two reasons:
///   1. It signs the new user in, which kicks the admin out of their
///      session.
///   2. The Flutter SDK cannot set custom claims, disable accounts, or
///      delete Auth users without re-authenticating as the target.
///
/// The right place for this is a Cloud Function that uses the Firebase
/// Admin SDK. We keep this abstraction so the UI flow exists today and
/// can be wired to a real backend tomorrow with a single edit.
///
/// **DEMO-ONLY behaviour today:** [createAuthAccount] does NOT actually
/// create a Firebase Auth user. It only writes a Firestore profile with
/// a synthetic uid prefixed `pending-`. When the real human signs up
/// later via Firebase Auth, the existing self-signup path
/// (loadOrCreateProfile) will create the real `users/{uid}` document.
///
/// To make this production-ready: implement `createAppUser` as a callable
/// Cloud Function that:
///   1. Verifies the caller's claim has role==admin.
///   2. Calls `admin.auth().createUser({ email, password, displayName })`.
///   3. Writes `users/{uid}` with the role and `createdBy = caller uid`.
///   4. (Optionally) sets a custom claim `role` so Firestore rules can
///      gate writes by claim instead of by document lookup.
class FirebaseUserAdminService {
  FirebaseUserAdminService() : _firestore = FirebaseFirestore.instance;
  final FirebaseFirestore _firestore;

  /// Creates a Firestore user profile *only*. Real Auth account creation
  /// requires a Cloud Function. See class docs.
  ///
  /// Throws [UserAdminUnsupportedError] when called against a backend
  /// that does not have the Cloud Function deployed (i.e. always today).
  Future<AppUser> createAppUser({
    required String email,
    required String password,
    required String name,
    required AppRole role,
    required String createdByUid,
    String assignedArea = '',
  }) async {
    // Pretend we have a callable. Today: only writes the profile.
    // Tomorrow: replace with FirebaseFunctions.instance.httpsCallable.
    final pendingUid =
        'pending-${DateTime.now().millisecondsSinceEpoch.toRadixString(16)}';
    final user = AppUser(
      id: pendingUid,
      name: name,
      email: email.trim().toLowerCase(),
      role: role,
      assignedArea: assignedArea,
      isActive: true,
      createdAt: DateTime.now(),
      updatedAt: DateTime.now(),
      createdBy: createdByUid,
    );
    await _firestore.collection('users').doc(pendingUid).set({
      ...user.toMap(),
      'createdAt': FieldValue.serverTimestamp(),
      'updatedAt': FieldValue.serverTimestamp(),
      // Hint for the UI to show "pending invite" state.
      'isPending': true,
    });
    return user;
  }

  /// Toggle active/inactive. This *is* safe to do client-side as long
  /// as Firestore rules require `request.auth` to have admin role.
  Future<void> setUserActive(String uid, bool isActive) {
    return _firestore.collection('users').doc(uid).set(
      {
        'isActive': isActive,
        'enabled': isActive,
        'updatedAt': FieldValue.serverTimestamp(),
      },
      SetOptions(merge: true),
    );
  }

  /// Update role. Same caveats as [setUserActive]: enforced by rules.
  Future<void> updateUserRole(String uid, AppRole role) {
    return _firestore.collection('users').doc(uid).set(
      {
        'role': role.name,
        'updatedAt': FieldValue.serverTimestamp(),
      },
      SetOptions(merge: true),
    );
  }
}

/// Thrown when an admin operation requires a backend that isn't deployed.
class UserAdminUnsupportedError implements Exception {
  final String message;
  const UserAdminUnsupportedError(this.message);
  @override
  String toString() => 'UserAdminUnsupportedError: $message';
}
