part of 'state.dart';

extension AppStateAuth on AppState {
  Permissions get permissions => Permissions.of(_currentUser);
  bool get isAdmin => _currentUser.role == AppRole.admin;
  bool get isEngineer => _currentUser.role == AppRole.engineer;
  bool get isViewer => _currentUser.role == AppRole.viewer;

  /// Defense-in-depth gate for write actions. Returns `true` if the
  /// action should be BLOCKED.
  bool _blockWrite(bool capability, String action) {
    if (!capability) {
      debugPrint(
        'AppState: blocked "$action" — '
        '${_currentUser.email} (${roleLabel(_currentUser.role)}, '
        'active=${_currentUser.isActive}) lacks permission.',
      );
      return true;
    }
    return false;
  }

  Future<String?> login({required String email, required String password}) async {
    final normalizedEmail = email.trim().toLowerCase();
    _authBusy = true;
    _notify();

    try {
      if (_firebaseEnabled && _firebase != null) {
        final uid = await _firebase!.signIn(normalizedEmail, password);

        final profile = await _firebase!.loadOrCreateProfile(
          uid: uid,
          email: normalizedEmail,
          displayName: _firebase!.currentAuthUser?.displayName,
        );

        if (!profile.isActive) {
          await _firebase!.signOut();
          return 'This account has been disabled. Contact your administrator.';
        }

        _currentUser = profile;
        final existingIndex = _users
            .indexWhere((u) => u.id == profile.id || u.email == profile.email);
        if (existingIndex >= 0) {
          final next = List<AppUser>.from(_users);
          next[existingIndex] = profile;
          _users = next;
        } else {
          _users = [..._users, profile];
        }

        _firebaseStatus =
            'Signed in as ${profile.email} (${roleLabel(profile.role)})';
      } else {
        final localUser = _users.firstWhereOrNull(
          (user) => user.email.toLowerCase() == normalizedEmail,
        );
        if (localUser != null) {
          _currentUser = localUser.copyWith(lastLoginAt: DateTime.now());
        } else {
          final role = resolveSeedRole(normalizedEmail);
          _currentUser = AppUser(
            id: 'demo-${DateTime.now().millisecondsSinceEpoch}',
            name: normalizedEmail.split('@').first,
            email: normalizedEmail,
            role: role,
            assignedArea: '',
            isActive: true,
            createdAt: DateTime.now(),
            updatedAt: DateTime.now(),
            lastLoginAt: DateTime.now(),
            createdBy: 'demo-mode',
          );
          _users = [..._users, _currentUser];
        }
        _firebaseStatus =
            'Demo sign-in (${roleLabel(_currentUser.role)}). Firebase will activate once real options are wired in.';
      }

      _isLoggedIn = true;
      _currentTab = AppTab.dashboard;
      if (_firebaseEnabled) {
        _simulator?.cancel();
        // Rebind Firestore streams now that the user is authenticated.
        // The streams started in _initialize() were rejected by Firestore
        // security rules (no auth at that point) and are effectively dead.
        // Rebinding here ensures machines and users load immediately after
        // login without requiring an app restart.
        _bindRemoteStreams();
      } else {
        _startSimulator();
      }
      await _persistCachedState();
      return null;
    } on FirebaseAuthException catch (error) {
      switch (error.code) {
        case 'user-not-found':
        case 'invalid-credential':
        case 'invalid-email':
          return 'No account found for that email. Contact admin if you need access.';
        case 'wrong-password':
          return 'Incorrect password. If you forgot it, contact admin.';
        case 'user-disabled':
          return 'This account has been disabled. Contact your administrator.';
        case 'too-many-requests':
          return 'Too many sign-in attempts. Please try again in a few minutes.';
        case 'network-request-failed':
          return 'Network error. Check your connection and try again.';
        default:
          return error.message ?? 'Authentication failed.';
      }
    } catch (error) {
      return 'Login failed: $error';
    } finally {
      _authBusy = false;
      _notify();
    }
  }

  Future<String?> resetPassword({required String email}) async {
    final normalizedEmail = email.trim().toLowerCase();

    if (!_firebaseEnabled || _firebase == null) {
      return 'Password reset requires a live Firebase connection. '
          'Make sure Firebase is configured and you are online.';
    }

    try {
      await _firebase!.sendPasswordResetEmail(normalizedEmail);
      return null;
    } on FirebaseAuthException catch (e) {
      switch (e.code) {
        case 'user-not-found':
        case 'invalid-credential':
          return 'No account found for that email address.';
        case 'invalid-email':
          return 'Enter a valid email address.';
        case 'too-many-requests':
          return 'Too many requests. Please try again later.';
        case 'network-request-failed':
          return 'Network error. Check your connection and try again.';
        default:
          return e.message ?? 'Failed to send reset email.';
      }
    } catch (e) {
      return 'Failed to send reset email: $e';
    }
  }

  void logout() {
    _isLoggedIn = false;
    _selectedMachineId = null;
    _currentTab = AppTab.dashboard;
    _simulator?.cancel();
    if (_firebaseEnabled && _firebase != null) {
      _firebase!.signOut().catchError((_) {});
    }
    _notify();
  }
}
