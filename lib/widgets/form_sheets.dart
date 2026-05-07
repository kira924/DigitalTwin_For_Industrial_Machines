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
import '../models/machine.dart';
import '../models/machine_report.dart';
import '../services/storage_service.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';

Future<void> showMachineFormSheet(BuildContext context, {Machine? machine}) {
  final name = TextEditingController(text: machine?.name ?? '');
  final code = TextEditingController(text: machine?.code ?? '');
  final type = TextEditingController(text: machine?.type ?? '');
  final location = TextEditingController(text: machine?.location ?? '');

  // Internal-only placeholder used when no real photo has been uploaded.
  // The user CANNOT pick this from a dropdown — assets are no longer
  // selectable as machine photos. They exist only as a visual fallback.
  const String placeholderAsset = 'assets/images/machine_cnc.png';

  MachineStatus selectedStatus = machine?.status ?? MachineStatus.running;

  // Image-upload state. We do the actual upload AFTER the machine is
  // saved (so we have a definite machineId for the storage path).
  XFile? pickedImage;
  String? currentImageUrl = machine?.imageUrl;
  bool uploadInFlight = false;
  double uploadProgress = 0.0;
  String? uploadError;

  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF07111F),
    builder: (_) {
      return StatefulBuilder(
        builder: (context, setState) {
          return Padding(
            padding: EdgeInsets.fromLTRB(
              18,
              18,
              18,
              18 + MediaQuery.of(context).viewInsets.bottom,
            ),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    machine == null ? 'Add Machine' : 'Edit Machine',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 24,
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextField(
                    controller: name,
                    decoration: const InputDecoration(labelText: 'Machine name'),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: code,
                    decoration: const InputDecoration(labelText: 'Machine ID / Code'),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: type,
                    decoration: const InputDecoration(labelText: 'Machine type'),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: location,
                    decoration: const InputDecoration(labelText: 'Location'),
                  ),
                  const SizedBox(height: 12),
                  DropdownButtonFormField<MachineStatus>(
                    initialValue: selectedStatus == MachineStatus.stopped
                        ? MachineStatus.offline
                        : selectedStatus,
                    decoration: const InputDecoration(labelText: 'Status'),
                    items: MachineStatus.values
                        // Phase 3.6: hide "stopped" from the UI; the
                        // resolver derives status automatically from
                        // telemetry + reports, and the dashboard maps
                        // any legacy stopped value to Offline.
                        .where((s) => s != MachineStatus.stopped)
                        .map(
                          (status) => DropdownMenuItem(
                            value: status,
                            child: Text(_machineStatusText(status)),
                          ),
                        )
                        .toList(growable: false),
                    onChanged: (value) => setState(
                      () => selectedStatus = value ?? MachineStatus.running,
                    ),
                  ),
                  const SizedBox(height: 12),
                  // ── Machine photo (optional) ──
                  // Admin can attach a real photo from the device. Photo
                  // is OPTIONAL — saving without one is fine. The model
                  // falls back to a polished placeholder (handled by
                  // MachineImage) when imageUrl is empty.
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: const Color(0xFF0E1A2A),
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(color: Colors.white.alphaF(0.06)),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Machine photo (optional)',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.w700,
                            fontSize: 13,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'Add a real photo from your device. Stored in Firebase Storage. Admins only — Engineers and Viewers can view but not change machine photos. Saving without a photo is fine; a placeholder is shown instead.',
                          style: TextStyle(
                            color: Colors.white.alphaF(0.55),
                            fontSize: 11,
                            height: 1.4,
                          ),
                        ),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            Container(
                              width: 60,
                              height: 60,
                              decoration: BoxDecoration(
                                color: Colors.white.alphaF(0.04),
                                borderRadius: BorderRadius.circular(8),
                                border:
                                    Border.all(color: Colors.white.alphaF(0.08)),
                              ),
                              clipBehavior: Clip.antiAlias,
                              child: pickedImage != null
                                  ? (kIsWeb
                                      ? Image.network(
                                          pickedImage!.path,
                                          fit: BoxFit.cover,
                                        )
                                      : Image.file(
                                          File(pickedImage!.path),
                                          fit: BoxFit.cover,
                                        ))
                                  : (currentImageUrl != null &&
                                          currentImageUrl!.isNotEmpty
                                      ? Image.network(
                                          currentImageUrl!,
                                          fit: BoxFit.cover,
                                          errorBuilder: (_, __, ___) =>
                                              const Icon(
                                            Icons.image_outlined,
                                            color: Colors.white54,
                                          ),
                                        )
                                      : Icon(
                                          Icons.add_photo_alternate_outlined,
                                          color: Colors.white.alphaF(0.4),
                                        )),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  OutlinedButton.icon(
                                    onPressed: uploadInFlight
                                        ? null
                                        : () async {
                                            try {
                                              final picker = ImagePicker();
                                              final picked = await picker.pickImage(
                                                source: ImageSource.gallery,
                                                maxWidth: 1600,
                                                imageQuality: 85,
                                              );
                                              if (picked == null) return;

                                              // Validate immediately so the
                                              // user sees the error now,
                                              // not on submit.
                                              try {
                                                final size =
                                                    await picked.length();
                                                validateImageFile(
                                                  filename: picked.name,
                                                  sizeBytes: size,
                                                );
                                              } on ImageValidationError catch (e) {
                                                setState(() =>
                                                    uploadError = e.message);
                                                return;
                                              }

                                              setState(() {
                                                pickedImage = picked;
                                                uploadError = null;
                                              });
                                            } catch (e) {
                                              setState(() {
                                                uploadError = 'Picker error: $e';
                                              });
                                            }
                                          },
                                    icon: const Icon(Icons.photo_library_outlined,
                                        size: 16),
                                    label: Text(pickedImage != null
                                        ? 'Replace photo'
                                        : (currentImageUrl != null
                                            ? 'Change photo'
                                            : 'Pick from gallery')),
                                  ),
                                  if (currentImageUrl != null &&
                                      pickedImage == null) ...[
                                    const SizedBox(height: 4),
                                    TextButton.icon(
                                      onPressed: uploadInFlight
                                          ? null
                                          : () => setState(() {
                                                currentImageUrl = null;
                                              }),
                                      icon: const Icon(Icons.delete_outline,
                                          size: 16, color: Colors.redAccent),
                                      label: const Text(
                                        'Remove photo',
                                        style: TextStyle(color: Colors.redAccent),
                                      ),
                                    ),
                                  ],
                                ],
                              ),
                            ),
                          ],
                        ),
                        if (uploadInFlight) ...[
                          const SizedBox(height: 10),
                          LinearProgressIndicator(
                            value: uploadProgress > 0 ? uploadProgress : null,
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Uploading… ${(uploadProgress * 100).toStringAsFixed(0)}%',
                            style: TextStyle(
                              color: Colors.white.alphaF(0.6),
                              fontSize: 11,
                            ),
                          ),
                        ],
                        if (uploadError != null) ...[
                          const SizedBox(height: 8),
                          Text(
                            uploadError!,
                            style: const TextStyle(
                              color: Colors.redAccent,
                              fontSize: 11,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: uploadInFlight
                          ? null
                          : () async {
                        final app = context.read<AppState>();
                        // Build the Machine. We pass through whatever
                        // imageUrl the user currently sees — it'll be
                        // replaced by setMachineImageUrl below if a
                        // new file is uploaded, or cleared if removed.
                        final item = Machine(
                          id: machine?.id ??
                              code.text.trim().toLowerCase().replaceAll(' ', '-'),
                          name: name.text.trim(),
                          type: type.text.trim(),
                          code: code.text.trim(),
                          location: location.text.trim(),
                          imagePath: machine?.imagePath ?? placeholderAsset,
                          imageUrl: currentImageUrl,
                          imageStoragePath: machine?.imageStoragePath,
                          status: selectedStatus,
                          efficiency: machine?.efficiency ?? 88,
                          temperature: machine?.temperature ?? 42,
                          speed: machine?.speed ?? 120,
                          vibration: machine?.vibration ?? 0.18,
                          power: machine?.power ?? 9.8,
                          pressure: machine?.pressure ?? 8,
                          productionCount: machine?.productionCount ?? 0,
                          productionRate: machine?.productionRate ?? 84,
                          activeAlert: machine?.activeAlert,
                          lastUpdated: DateTime.now(),
                          performanceSeries: machine?.performanceSeries ?? const [
                            MachineMetricPoint(label: '10:00', value: 82),
                            MachineMetricPoint(label: '10:15', value: 86),
                            MachineMetricPoint(label: '10:30', value: 88),
                            MachineMetricPoint(label: '10:45', value: 91),
                          ],
                          sensors: machine?.sensors ?? const [
                            MachineSensor(
                              label: 'Vibration',
                              unit: 'mm/s',
                              value: 0.18,
                              color: DT.green,
                            ),
                            MachineSensor(
                              label: 'Noise Level',
                              unit: 'dB',
                              value: 72,
                              color: DT.blue,
                            ),
                            MachineSensor(
                              label: 'Oil Pressure',
                              unit: 'PSI',
                              value: 45,
                              color: DT.cyan,
                            ),
                          ],
                          timeline: machine?.timeline ?? [
                            MachineTimelineEvent(
                              time: DateTime.now(),
                              title: 'Machine created',
                              detail: 'Machine record was created by admin.',
                              color: DT.cyan,
                            ),
                          ],
                        );
                        if (machine == null) {
                          app.addMachine(item);
                        } else {
                          app.updateMachine(item);
                        }

                        // ── Image side-effects ──────────────────────
                        // Three possible cases:
                        // 1) New file picked → upload + persist
                        // 2) Removed (was set, now null) → persist null
                        // 3) Unchanged → just close

                        // Case 2: removed photo
                        if (pickedImage == null &&
                            machine != null &&
                            machine.imageUrl != null &&
                            currentImageUrl == null) {
                          try {
                            await app.setMachineImageUrl(
                              item.id,
                              imageUrl: null,
                              imageStoragePath: null,
                            );
                          } catch (_) {/* best-effort */}
                        }

                        // Case 3: nothing to upload
                        if (pickedImage == null) {
                          if (context.mounted) Navigator.pop(context);
                          return;
                        }

                        // Case 1: new image picked.
                        if (!app.firebaseEnabled) {
                          // Honest fail — without Firebase initialized,
                          // a Storage upload would hang. Surface a
                          // clear message instead of pretending.
                          setState(() {
                            uploadError =
                                'Cannot upload right now: Firebase Storage '
                                'is not initialized. Save the machine without '
                                'a photo, or configure Firebase.';
                          });
                          return;
                        }

                        setState(() {
                          uploadInFlight = true;
                          uploadProgress = 0.0;
                          uploadError = null;
                        });
                        try {
                          // File-size validation (web variant validates
                          // inside StorageService since size lives on
                          // Uint8List directly).
                          final svc = StorageService();
                          UploadResult result;
                          if (kIsWeb) {
                            final bytes = await pickedImage!.readAsBytes();
                            result = await svc.uploadMachineImageBytes(
                              machineId: item.id,
                              bytes: bytes,
                              filename: pickedImage!.name,
                              onProgress: (p) {
                                if (context.mounted) {
                                  setState(() => uploadProgress = p);
                                }
                              },
                            );
                          } else {
                            result = await svc.uploadMachineImage(
                              machineId: item.id,
                              file: File(pickedImage!.path),
                              onProgress: (p) {
                                if (context.mounted) {
                                  setState(() => uploadProgress = p);
                                }
                              },
                            );
                          }
                          // Persist URL + path to Firestore + local state.
                          await app.setMachineImageUrl(
                            item.id,
                            imageUrl: result.url,
                            imageStoragePath: result.path,
                          );
                          if (context.mounted) Navigator.pop(context);
                        } on ImageValidationError catch (e) {
                          if (context.mounted) {
                            setState(() => uploadError = e.message);
                          }
                        } on TimeoutException {
                          if (context.mounted) {
                            setState(() => uploadError =
                                'Upload timed out. Please check your '
                                'connection or Firebase Storage rules.');
                          }
                        } on StorageUploadError catch (e) {
                          if (context.mounted) {
                            setState(() => uploadError = e.message);
                          }
                        } catch (e) {
                          if (context.mounted) {
                            setState(() => uploadError = 'Upload failed: $e');
                          }
                        } finally {
                          // ALWAYS clear the loading state — this is the
                          // central fix for the hang. Previously the flag
                          // was only flipped on the error path.
                          if (context.mounted) {
                            setState(() {
                              uploadInFlight = false;
                              uploadProgress = 0.0;
                            });
                          }
                        }
                      },
                      child: Text(
                        machine == null ? 'Save machine' : 'Update machine',
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      );
    },
  );
}

Future<void> showUserFormSheet(BuildContext context, {AppUser? user}) {
  final name = TextEditingController(text: user?.name ?? '');
  final email = TextEditingController(text: user?.email ?? '');
  final password = TextEditingController(); // new users only; never stored
  final area = TextEditingController(text: user?.assignedArea ?? '');
  AppRole selectedRole = user?.role ?? AppRole.viewer;

  XFile? pickedAvatar;
  String? formError;
  bool busy = false;
  String? nameError;
  String? emailError;
  String? passwordError;
  bool passwordVisible = false;

  final isNew = user == null;

  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF07111F),
    builder: (sheetCtx) {
      return StatefulBuilder(
        builder: (context, setState) {
          final app = context.read<AppState>();

          Future<void> submit() async {
            final n = name.text.trim();
            final e = email.text.trim();
            final pw = password.text; // intentionally not trimmed

            // ── Validation ────────────────────────────────────────────────
            setState(() {
              nameError = n.isEmpty ? 'Name is required.' : null;
              emailError = !RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(e)
                  ? 'Enter a valid email address.'
                  : null;
              passwordError = isNew && pw.length < 6
                  ? 'Password must be at least 6 characters.'
                  : null;
              formError = null;
            });
            if (nameError != null || emailError != null || passwordError != null) {
              return;
            }

            // Refuse duplicate emails for new users.
            if (isNew) {
              final dup = app.users.any(
                (u) => u.email.toLowerCase() == e.toLowerCase(),
              );
              if (dup) {
                setState(() => formError = 'A user with that email already exists.');
                return;
              }
            }

            setState(() {
              busy = true;
              formError = null;
            });

            try {
              // Temporary id used only for the avatar Cloudinary upload path.
              // For new users this is replaced by the real Firebase uid once
              // the backend responds. For edits it stays as the existing id.
              final tempId = user?.id ?? 'tmp-${DateTime.now().millisecondsSinceEpoch}';

              AppUser item = AppUser(
                id: tempId,
                name: n,
                email: e,
                role: selectedRole,
                assignedArea: area.text.trim(),
                isActive: user?.isActive ?? true,
                photoUrl: user?.photoUrl,
                photoStoragePath: user?.photoStoragePath,
                createdAt: user?.createdAt ?? DateTime.now(),
                updatedAt: DateTime.now(),
                createdBy: user?.createdBy ?? app.currentUser.id,
              );

              // ── Avatar upload (optional, before persisting) ────────────
              if (pickedAvatar != null) {
                try {
                  final size = await pickedAvatar!.length();
                  validateImageFile(filename: pickedAvatar!.name, sizeBytes: size);

                  final svc = ImageUploadApiService();
                  UploadResult result;
                  if (kIsWeb) {
                    final bytes = await pickedAvatar!.readAsBytes();
                    result = await svc.uploadUserAvatarBytes(
                      uid: tempId,
                      bytes: bytes,
                      filename: pickedAvatar!.name,
                    );
                  } else {
                    result = await svc.uploadUserAvatar(
                      uid: tempId,
                      file: File(pickedAvatar!.path),
                    );
                  }
                  item = item.copyWith(
                    photoUrl: result.url,
                    photoStoragePath: result.path,
                  );
                } on ImageValidationError catch (err) {
                  setState(() { busy = false; formError = err.message; });
                  return;
                } on TimeoutException {
                  setState(() {
                    busy = false;
                    formError = 'Avatar upload timed out. Check the image '
                        'upload backend is running.';
                  });
                  return;
                } on StorageUploadError catch (err) {
                  setState(() { busy = false; formError = err.message; });
                  return;
                } catch (err) {
                  setState(() { busy = false; formError = 'Avatar upload failed: $err'; });
                  return;
                }
              }

              // ── Persist ────────────────────────────────────────────────
              if (isNew) {
                // POST to FastAPI → Firebase Admin SDK creates Auth account
                // and Firestore profile atomically, then returns the real uid.
                final uri = Uri.parse(
                  '${ApiConfig.baseUrl}${ApiConfig.createUserEndpoint}',
                );
                final http.Response response;
                try {
                  response = await http
                      .post(
                        uri,
                        headers: {'Content-Type': 'application/json'},
                        body: jsonEncode({
                          'name': n,
                          'email': e,
                          'password': pw,
                          'role': selectedRole.name,
                          'assignedArea': area.text.trim(),
                          'photoUrl': item.photoUrl ?? '',
                          'createdByUid': app.currentUser.id,
                        }),
                      )
                      .timeout(ApiConfig.createUserTimeout);
                } on TimeoutException {
                  setState(() {
                    busy = false;
                    formError = 'Request timed out. Check the backend is running.';
                  });
                  return;
                } catch (err) {
                  setState(() {
                    busy = false;
                    formError = 'Could not reach the backend: $err';
                  });
                  return;
                }

                if (response.statusCode != 200) {
                  String detail = 'Server error ${response.statusCode}.';
                  try {
                    final body = jsonDecode(response.body) as Map<String, dynamic>;
                    detail = (body['detail'] as String?) ?? detail;
                  } catch (_) {}
                  setState(() { busy = false; formError = detail; });
                  return;
                }

                // Backend success — use the real Firebase uid.
                final data = jsonDecode(response.body) as Map<String, dynamic>;
                final realUid = data['uid'] as String;
                // Replace temp id with real uid before adding to local state.
                final created = item.copyWith(id: realUid);
                // addUser updates the local list and history.
                // The backend already wrote Firestore; the merge write here is
                // harmless (SetOptions.merge — does not touch isPending:false).
                app.addUser(created);
              } else {
                // Editing an existing user: local + Firestore update only.
                app.updateUser(item);
              }

              if (!context.mounted) return;
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                    isNew ? 'User created: $n' : 'User updated: $n',
                  ),
                ),
              );
            } catch (err) {
              setState(() {
                busy = false;
                formError = 'Could not save user: $err';
              });
            }
          }

          Future<void> pickAvatar() async {
            try {
              final picker = ImagePicker();
              final picked = await picker.pickImage(
                source: ImageSource.gallery,
                maxWidth: 1024,
                imageQuality: 85,
              );
              if (picked == null) return;
              try {
                final size = await picked.length();
                validateImageFile(filename: picked.name, sizeBytes: size);
              } on ImageValidationError catch (err) {
                setState(() => formError = err.message);
                return;
              }
              setState(() {
                pickedAvatar = picked;
                formError = null;
              });
            } catch (err) {
              setState(() => formError = 'Picker error: $err');
            }
          }

          return Padding(
            padding: EdgeInsets.fromLTRB(
              18,
              18,
              18,
              18 + MediaQuery.of(context).viewInsets.bottom,
            ),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    isNew ? 'Add User' : 'Edit User',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 24,
                    ),
                  ),
                  const SizedBox(height: 14),

                  // ── Name ─────────────────────────────────────────────────
                  TextField(
                    controller: name,
                    enabled: !busy,
                    decoration: InputDecoration(
                      labelText: 'Full name',
                      errorText: nameError,
                    ),
                  ),
                  const SizedBox(height: 10),

                  // ── Email ────────────────────────────────────────────────
                  TextField(
                    controller: email,
                    enabled: !busy && isNew, // email is immutable on edit
                    keyboardType: TextInputType.emailAddress,
                    decoration: InputDecoration(
                      labelText: 'Email',
                      errorText: emailError,
                    ),
                  ),
                  const SizedBox(height: 10),

                  // ── Password (new users only) ─────────────────────────────
                  if (isNew) ...[
                    TextField(
                      controller: password,
                      enabled: !busy,
                      obscureText: !passwordVisible,
                      decoration: InputDecoration(
                        labelText: 'Password',
                        helperText: 'Minimum 6 characters.',
                        errorText: passwordError,
                        suffixIcon: IconButton(
                          icon: Icon(
                            passwordVisible
                                ? Icons.visibility_off_outlined
                                : Icons.visibility_outlined,
                            size: 18,
                          ),
                          onPressed: () =>
                              setState(() => passwordVisible = !passwordVisible),
                          tooltip: passwordVisible ? 'Hide' : 'Show',
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),
                  ],

                  // ── Assigned area ─────────────────────────────────────────
                  TextField(
                    controller: area,
                    enabled: !busy,
                    decoration: const InputDecoration(
                      labelText: 'Assigned area (optional)',
                    ),
                  ),
                  const SizedBox(height: 10),

                  // ── Role ──────────────────────────────────────────────────
                  DropdownButtonFormField<AppRole>(
                    initialValue: selectedRole,
                    decoration: const InputDecoration(labelText: 'Role'),
                    items: const [
                      DropdownMenuItem(value: AppRole.admin, child: Text('Admin')),
                      DropdownMenuItem(value: AppRole.engineer, child: Text('Engineer')),
                      DropdownMenuItem(value: AppRole.viewer, child: Text('Viewer')),
                    ],
                    onChanged: busy
                        ? null
                        : (value) =>
                            setState(() => selectedRole = value ?? AppRole.viewer),
                  ),
                  const SizedBox(height: 14),

                  // ── Avatar ────────────────────────────────────────────────
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.white.alphaF(0.03),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.white.alphaF(0.08)),
                    ),
                    child: Row(
                      children: [
                        const Icon(Icons.account_circle_outlined,
                            color: Colors.white70, size: 28),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                pickedAvatar != null
                                    ? 'Avatar selected: ${pickedAvatar!.name}'
                                    : (user?.photoUrl != null
                                        ? 'Avatar already set (existing photo).'
                                        : 'Avatar (optional)'),
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 12,
                                  fontWeight: FontWeight.w600,
                                ),
                              ),
                              const SizedBox(height: 2),
                              Text(
                                'JPG / PNG / WEBP up to 5 MB. '
                                'Uploaded to Cloudinary via FastAPI.',
                                style: TextStyle(
                                  color: Colors.white.alphaF(0.55),
                                  fontSize: 11,
                                  height: 1.3,
                                ),
                              ),
                            ],
                          ),
                        ),
                        TextButton.icon(
                          onPressed: busy ? null : pickAvatar,
                          icon: const Icon(Icons.photo_library_outlined, size: 16),
                          label: Text(pickedAvatar != null ? 'Replace' : 'Pick'),
                        ),
                      ],
                    ),
                  ),

                  // ── Progress / error ──────────────────────────────────────
                  if (busy) ...[
                    const SizedBox(height: 14),
                    const LinearProgressIndicator(),
                    const SizedBox(height: 6),
                    Text(
                      isNew
                          ? 'Creating user account…'
                          : 'Saving changes…',
                      style: const TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                  ],
                  if (formError != null) ...[
                    const SizedBox(height: 12),
                    Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: DTTokens.statusCritical.alphaF(0.10),
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                            color: DTTokens.statusCritical.alphaF(0.30)),
                      ),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Icon(Icons.error_outline,
                              color: DTTokens.statusCritical, size: 16),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              formError!,
                              style: const TextStyle(
                                color: DTTokens.statusCritical,
                                fontSize: 12,
                                height: 1.4,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],

                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: busy ? null : submit,
                      child: Text(
                        busy
                            ? (isNew ? 'Creating…' : 'Saving…')
                            : (isNew ? 'Create user' : 'Update user'),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      );
    },
  );
}

String _machineStatusText(MachineStatus status) => switch (status) {
      MachineStatus.running => 'Running',
      MachineStatus.maintenance => 'Under Maintenance',
      MachineStatus.fault => 'Fault',
      MachineStatus.stopped => 'Offline',
      MachineStatus.offline => 'Offline',
    };

/// Bottom sheet to file or edit a machine report.
///
/// The CALLER is responsible for permission gating — this sheet trusts
/// that the engineer/admin has already cleared the [Permissions] check.
/// We re-check at the top anyway as a defense-in-depth measure.
Future<void> showReportSheet(
  BuildContext context, {
  required Machine machine,
  MachineReport? existing,
}) {
  final app = context.read<AppState>();
  final perms = app.permissions;
  final isEdit = existing != null;

  // Defense in depth: even if a viewer hits this somehow, refuse.
  if (!perms.canWriteReports) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('You do not have permission to write reports.'),
      ),
    );
    return Future.value();
  }
  if (isEdit && !perms.canEditOwnReport(existing.authorId, app.currentUser.id)) {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Only the author or an admin can edit this report.'),
      ),
    );
    return Future.value();
  }

  final title = TextEditingController(text: existing?.title ?? '');
  final desc = TextEditingController(text: existing?.description ?? '');
  final action =
      TextEditingController(text: existing?.recommendedAction ?? '');
  ReportSeverity severity = existing?.severity ?? ReportSeverity.medium;
  ReportStatus status = existing?.status ?? ReportStatus.open;

  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    backgroundColor: const Color(0xFF07111F),
    builder: (sheetContext) {
      return StatefulBuilder(
        builder: (sheetContext, setState) {
          return Padding(
            padding: EdgeInsets.fromLTRB(
              18,
              18,
              18,
              18 + MediaQuery.of(sheetContext).viewInsets.bottom,
            ),
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    isEdit ? 'Edit Report' : 'Write Report',
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                      fontSize: 22,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${machine.name} · ${machine.code}',
                    style: TextStyle(
                      color: Colors.white.alphaF(0.55),
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 14),
                  TextField(
                    controller: title,
                    decoration: const InputDecoration(labelText: 'Title'),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: desc,
                    minLines: 3,
                    maxLines: 6,
                    decoration:
                        const InputDecoration(labelText: 'Description'),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: action,
                    minLines: 2,
                    maxLines: 4,
                    decoration: const InputDecoration(
                      labelText: 'Recommended action (optional)',
                    ),
                  ),
                  const SizedBox(height: 14),
                  DropdownButtonFormField<ReportSeverity>(
                    initialValue: severity,
                    decoration:
                        const InputDecoration(labelText: 'Severity'),
                    items: ReportSeverity.values
                        .map(
                          (s) => DropdownMenuItem(
                            value: s,
                            child: Text(s.label),
                          ),
                        )
                        .toList(growable: false),
                    onChanged: (v) =>
                        setState(() => severity = v ?? ReportSeverity.medium),
                  ),
                  const SizedBox(height: 10),
                  DropdownButtonFormField<ReportStatus>(
                    initialValue: status,
                    decoration: const InputDecoration(labelText: 'Status'),
                    items: ReportStatus.values
                        .map(
                          (s) => DropdownMenuItem(
                            value: s,
                            child: Text(s.label),
                          ),
                        )
                        .toList(growable: false),
                    onChanged: (v) =>
                        setState(() => status = v ?? ReportStatus.open),
                  ),
                  const SizedBox(height: 18),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton(
                      onPressed: () {
                        if (title.text.trim().isEmpty) {
                          ScaffoldMessenger.of(sheetContext).showSnackBar(
                            const SnackBar(
                              content: Text('Title is required.'),
                            ),
                          );
                          return;
                        }
                        if (isEdit) {
                          app.updateReport(
                            existing.copyWith(
                              title: title.text.trim(),
                              description: desc.text.trim(),
                              recommendedAction: action.text.trim(),
                              severity: severity,
                              status: status,
                            ),
                          );
                        } else {
                          app.addReport(
                            machineId: machine.id,
                            title: title.text.trim(),
                            description: desc.text.trim(),
                            recommendedAction: action.text.trim(),
                            severity: severity,
                            status: status,
                          );
                        }
                        Navigator.pop(sheetContext);
                      },
                      child: Text(isEdit ? 'Save changes' : 'File report'),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      );
    },
  );
}