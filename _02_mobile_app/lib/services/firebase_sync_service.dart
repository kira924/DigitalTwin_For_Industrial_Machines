import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../models/app_user.dart';
import '../models/machine.dart';
import '../models/machine_report.dart';

/// Wraps Firebase Auth + Firestore access for the app.
///
/// Notes:
///  - Login is uid-based. After [signIn], we look up `users/{uid}`. If
///    missing, we auto-create the profile using [resolveSeedRole] from
///    the email allowlist.
///  - Machine subcollections (`machines/{id}/history`, `.../reports`)
///    are watched per-machine on demand by [watchHistory] /
///    [watchReports]. The detail screen subscribes when the user is
///    looking at that machine and unsubscribes when they leave.
///  - Image URL update is a thin Firestore write — actual upload to
///    Firebase Storage is in [StorageService].
class FirebaseSyncService {
  FirebaseSyncService()
      : _firestore = FirebaseFirestore.instance,
        _auth = FirebaseAuth.instance;

  final FirebaseFirestore _firestore;
  final FirebaseAuth _auth;

  // ── Auth ──────────────────────────────────────────────────────────────────

  /// Signs in with email + password. Returns the Firebase Auth uid.
  Future<String> signIn(String email, String password) async {
    final cred = await _auth.signInWithEmailAndPassword(
      email: email.trim(),
      password: password,
    );
    final uid = cred.user?.uid;
    if (uid == null) {
      throw FirebaseAuthException(
        code: 'no-uid',
        message: 'Firebase did not return a uid for this user.',
      );
    }
    return uid;
  }

  Future<void> signOut() => _auth.signOut();

  /// Sends a Firebase-hosted password-reset email to [email].
  /// Throws [FirebaseAuthException] on failure; callers should catch it.
  Future<void> sendPasswordResetEmail(String email) =>
      _auth.sendPasswordResetEmail(email: email.trim().toLowerCase());

  User? get currentAuthUser => _auth.currentUser;

  /// Loads `users/{uid}`. Returns null if the document does not exist.
  Future<AppUser?> fetchUserByUid(String uid) async {
    final doc = await _firestore.collection('users').doc(uid).get();
    if (!doc.exists) return null;
    final data = doc.data();
    if (data == null) return null;
    return AppUser.fromMap(data, fallbackId: uid);
  }

  /// Legacy: fetch by email (used when migrating older accounts).
  Future<AppUser?> fetchUserByEmail(String email) async {
    final result = await _firestore
        .collection('users')
        .where('email', isEqualTo: email.trim().toLowerCase())
        .limit(1)
        .get();
    if (result.docs.isEmpty) return null;
    return AppUser.fromMap(result.docs.first.data(),
        fallbackId: result.docs.first.id);
  }

  /// Loads or auto-creates the user profile for the just-authenticated
  /// Firebase Auth user. Always sets `lastLoginAt` to server-time.
  Future<AppUser> loadOrCreateProfile({
    required String uid,
    required String email,
    String? displayName,
  }) async {
    final ref = _firestore.collection('users').doc(uid);
    final snap = await ref.get();
    final now = DateTime.now();

    if (snap.exists && snap.data() != null) {
      // Existing profile — touch lastLoginAt and return.
      final user = AppUser.fromMap(snap.data()!, fallbackId: uid)
          .copyWith(lastLoginAt: now, updatedAt: now);
      await ref.set(
        {
          'lastLoginAt': FieldValue.serverTimestamp(),
          'updatedAt': FieldValue.serverTimestamp(),
        },
        SetOptions(merge: true),
      );
      return user;
    }

    // Auto-create profile using the email allowlist.
    final role = resolveSeedRole(email);
    final created = AppUser(
      id: uid,
      name: displayName?.trim().isNotEmpty == true
          ? displayName!.trim()
          : email.split('@').first,
      email: email.trim().toLowerCase(),
      role: role,
      assignedArea: '',
      isActive: true,
      createdAt: now,
      updatedAt: now,
      lastLoginAt: now,
      createdBy: 'self-signup',
    );
    final asMap = created.toMap()
      ..['createdAt'] = FieldValue.serverTimestamp()
      ..['updatedAt'] = FieldValue.serverTimestamp()
      ..['lastLoginAt'] = FieldValue.serverTimestamp();
    await ref.set(asMap);
    return created;
  }

  // ── Users ─────────────────────────────────────────────────────────────────

  Stream<List<AppUser>> watchUsers() {
    return _firestore.collection('users').snapshots().map((snapshot) {
      return snapshot.docs
          .map((doc) => AppUser.fromMap(doc.data(), fallbackId: doc.id))
          .toList(growable: false)
        ..sort((a, b) => a.name.compareTo(b.name));
    });
  }

  Future<void> upsertUser(AppUser user) {
    final asMap = user.toMap()
      ..['updatedAt'] = FieldValue.serverTimestamp();
    return _firestore.collection('users').doc(user.id).set(
          asMap,
          SetOptions(merge: true),
        );
  }

  Future<void> deleteUser(String userId) {
    // The backend /admin/delete-user endpoint already deleted the Firebase
    // Auth account and soft-deleted the Firestore document.  This call is a
    // fallback merge-write that marks the document inactive in case the
    // Firestore step is retried or the backend was unavailable.
    return _firestore.collection('users').doc(userId).set(
      {
        'isActive': false,
        'isDeleted': true,
        'deletedAt': FieldValue.serverTimestamp(),
      },
      SetOptions(merge: true),
    );
  }

  // ── Machines ──────────────────────────────────────────────────────────────

  Stream<List<Machine>> watchMachines() {
    return _firestore.collection('machines').snapshots().map((snapshot) {
      return snapshot.docs
          .map((doc) => Machine.fromMap(doc.data(), fallbackId: doc.id))
          .toList(growable: false)
        ..sort((a, b) => a.code.compareTo(b.code));
    });
  }

  Future<void> upsertMachine(Machine machine) {
    return _firestore
        .collection('machines')
        .doc(machine.id)
        .set(machine.toMap(), SetOptions(merge: true));
  }

  Future<void> deleteMachine(String machineId) {
    return _firestore.collection('machines').doc(machineId).delete();
  }

  /// Updates only the `imageUrl` field on a machine.
  Future<void> updateMachineImageUrl(String machineId, String imageUrl) {
    return _firestore.collection('machines').doc(machineId).set(
      {'imageUrl': imageUrl, 'updatedAt': FieldValue.serverTimestamp()},
      SetOptions(merge: true),
    );
  }

  Future<void> seedIfEmpty(List<Machine> machines, List<AppUser> users) async {
    final machineCollection = _firestore.collection('machines');
    final userCollection = _firestore.collection('users');

    final machineSnapshot = await machineCollection.limit(1).get();
    if (machineSnapshot.docs.isEmpty) {
      final batch = _firestore.batch();
      for (final machine in machines) {
        batch.set(machineCollection.doc(machine.id), machine.toMap());
      }
      await batch.commit();
    }

    final userSnapshot = await userCollection.limit(1).get();
    if (userSnapshot.docs.isEmpty) {
      final batch = _firestore.batch();
      for (final user in users) {
        batch.set(userCollection.doc(user.id), user.toMap());
      }
      await batch.commit();
    }
  }

  // ── Per-machine reports ───────────────────────────────────────────────────

  Stream<List<MachineReport>> watchReports(String machineId) {
    return _firestore
        .collection('machines')
        .doc(machineId)
        .collection('reports')
        .orderBy('createdAt', descending: true)
        .snapshots()
        .map((s) => s.docs
            .map((d) => MachineReport.fromMap(_withId(d.data(), d.id)))
            .toList(growable: false));
  }

  Future<void> upsertReport(MachineReport report) {
    final ref = _firestore
        .collection('machines')
        .doc(report.machineId)
        .collection('reports')
        .doc(report.id);
    final asMap = report.toMap()
      ..['createdAt'] = FieldValue.serverTimestamp()
      ..['updatedAt'] = FieldValue.serverTimestamp();
    return ref.set(asMap, SetOptions(merge: true));
  }

  Future<void> deleteReport(String machineId, String reportId) {
    return _firestore
        .collection('machines')
        .doc(machineId)
        .collection('reports')
        .doc(reportId)
        .delete();
  }

  // ── Per-machine history ───────────────────────────────────────────────────

  /// Each history entry is a generic event against a machine. The Flutter
  /// side serializes [HistoryLog]-like maps; we don't enforce a Dart type
  /// here so non-Dart writers (cloud functions, IoT pipelines) can append
  /// without coupling to our model.
  Stream<List<Map<String, dynamic>>> watchHistory(String machineId) {
    return _firestore
        .collection('machines')
        .doc(machineId)
        .collection('history')
        .orderBy('createdAt', descending: true)
        .limit(200)
        .snapshots()
        .map((s) => s.docs.map((d) => _withId(d.data(), d.id)).toList());
  }

  Future<void> appendHistoryEvent(
    String machineId,
    Map<String, dynamic> event,
  ) {
    final ref = _firestore
        .collection('machines')
        .doc(machineId)
        .collection('history')
        .doc();
    return ref.set({
      ...event,
      'id': ref.id,
      'machineId': machineId,
      'createdAt': FieldValue.serverTimestamp(),
    });
  }

  // ── Settings ──────────────────────────────────────────────────────────────

  Future<Map<String, dynamic>?> loadFactorySettings() async {
    final snapshot = await _firestore.collection('config').doc('app').get();
    return snapshot.data();
  }

  Future<void> saveFactorySettings(Map<String, dynamic> data) {
    return _firestore.collection('config').doc('app').set(
          data,
          SetOptions(merge: true),
        );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  Map<String, dynamic> _withId(Map<String, dynamic> data, String id) {
    return {...data, 'id': data['id'] ?? id};
  }
}
