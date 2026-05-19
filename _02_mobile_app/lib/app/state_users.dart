part of 'state.dart';

extension AppStateUsers on AppState {
  void addUser(AppUser user) {
    if (_blockWrite(permissions.canManageUsers, 'addUser')) return;
    _users = [..._users, user];
    _pushHistory('User added', '${user.name} joined as ${user.role.name}.',
        Icons.person_add_alt_1_rounded, DT.cyan);
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertUser(user).catchError((_) {});
    }
    _notify();
  }

  Future<String?> addUserAsync(AppUser user) async {
    if (_blockWrite(permissions.canManageUsers, 'addUserAsync')) {
      return 'You do not have permission to add users.';
    }
    _users = [..._users, user];
    _pushHistory(
      'User added',
      '${user.name} joined as ${user.role.name}.',
      Icons.person_add_alt_1_rounded,
      DT.cyan,
    );
    _persistCachedState();
    _notify();

    if (_firebaseEnabled && _firebase != null) {
      try {
        await _firebase!.upsertUser(user);
        return null;
      } catch (e) {
        return 'User saved locally but Firestore write failed: $e';
      }
    }
    return null;
  }

  void updateUser(AppUser user) {
    final isSelf = user.id == _currentUser.id;
    if (_blockWrite(permissions.canManageUsers || isSelf, 'updateUser')) {
      return;
    }
    _users = _users
        .map((item) => item.id == user.id ? user : item)
        .toList(growable: false);
    if (_currentUser.id == user.id) {
      _currentUser = user;
    }
    _pushHistory('User updated', '${user.name} profile was updated.',
        Icons.manage_accounts_rounded, DT.blue);
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.upsertUser(user).catchError((_) {});
    }
    _notify();
  }

  void deleteUser(String userId) {
    if (_blockWrite(permissions.canDeleteUsers, 'deleteUser')) return;
    if (_currentUser.id == userId) return;

    final target = _users.firstWhereOrNull((user) => user.id == userId);

    if (target?.role == AppRole.admin) {
      final remainingAdmins = _users
          .where((u) => u.role == AppRole.admin && u.id != userId)
          .length;
      if (remainingAdmins == 0) return;
    }

    _users = _users.where((user) => user.id != userId).toList(growable: false);
    if (target != null) {
      _pushHistory(
        'User removed',
        '${target.name} was removed from access control.',
        Icons.person_remove_alt_1_rounded,
        DT.red,
      );
    }
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.deleteUser(userId).catchError((_) {});
    }
    _notify();
  }

  Future<void> setMyPhotoUrl({
    required String? photoUrl,
    required String? photoStoragePath,
  }) async {
    final oldStoragePath = _currentUser.photoStoragePath;
    final updated = _currentUser.copyWith(
      photoUrl: photoUrl,
      clearPhotoUrl: photoUrl == null,
      photoStoragePath: photoStoragePath,
      clearPhotoStoragePath: photoStoragePath == null,
      updatedAt: DateTime.now(),
    );
    _currentUser = updated;
    _users = _users
        .map((u) => u.id == updated.id ? updated : u)
        .toList(growable: false);
    _persistCachedState();
    if (_firebaseEnabled && _firebase != null) {
      if (photoUrl == null) {
        _firebase!.clearUserPhotoFields(_currentUser.id).catchError((_) {});
      } else {
        _firebase!.upsertUser(updated).catchError((_) {});
      }
    }
    if (oldStoragePath != null &&
        oldStoragePath.isNotEmpty &&
        oldStoragePath != photoStoragePath) {
      // ignore: unawaited_futures
      StorageService().deleteByPath(oldStoragePath);
    }
    _notify();
  }
}
