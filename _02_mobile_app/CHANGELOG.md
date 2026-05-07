# Changelog — Phase 3.6: Dashboard layout, status cleanup, Add User reliability

This pass closes the 8 items reported against Phase 3.5: Dashboard
layout matches the sketch, every "Stopped" surface reads as "Offline"
or is hidden, the Machine Detail Start/Stop buttons stay gone, and
Add User has full validation / loading / success / error states with
optional Cloudinary avatar upload.

---

## Item-by-item summary

### 1. Energy widget completely removed from Dashboard ✅

`lib/screens/dashboard_screen.dart` — no `Energy` KPI tile, no
`energyConsumption` reference, no energy section. Verified with
`grep -rn 'energy\|Energy' lib/screens/ lib/widgets/` returning only
a trail comment marking the removal.

### 2. Top Risks table/card present ✅

New `_TopRisksCard` widget in `dashboard_screen.dart` (lines ~504+):
- Header reads "Top Risks" with a critical-colored priority icon and a
  count badge.
- Up to 5 rows of `_AtRiskRow`, each showing a status dot, machine
  name, code · location, current efficiency %, and predicted RUL.
- Empty state: "No machines flagged. Everything is healthy."
- Each row is wrapped in `InkWell` with `onTap: Navigator.push(...)
  → MachineDetailScreen(machineId: machine.id)`, so clicking a row
  jumps to that machine's detail page.

### 3. Top Risks card placed beside the metric tiles ✅

`dashboard_screen.dart` uses `LayoutBuilder` around the KPI section
to do responsive layout matching the sketch:

```dart
LayoutBuilder(builder: (ctx, c) {
  final wide = c.maxWidth >= 720;
  if (wide) {
    return Row(children: [
      Expanded(flex: 5, child: _KpiMiniGrid(...)),
      Expanded(flex: 4, child: _TopRisksCard(...)),
    ]);
  }
  return Column(children: [_KpiMiniGrid(...), _TopRisksCard(...)]);
});
```

- **Wide (≥ 720 px)** — Online / OEE / Alerts / Faults grid on the
  left (5/9 of the row), Top Risks on the right (4/9). Matches the
  sketch exactly.
- **Narrow** — same widgets stacked vertically, Top Risks below.

### 4. "Stopped" status removed from all visible UI ✅

The `MachineStatus.stopped` enum value is preserved (Dart needs it for
exhaustive switches), but every place it could surface in user-facing
text now reads "Offline":

| Site | Before | After |
|---|---|---|
| `dashboard_screen.dart` `_statusLabel` (was) | `'STOPPED'` | `'OFFLINE'` |
| `dashboard_screen.dart` `_statusText` (was) | `'Stopped'` | `'Offline'` |
| `machine_detail_screen.dart` `_statusLabel` | `'STOPPED'` | `'OFFLINE'` |
| `machines_screen.dart` `statusText` | `'Stopped'` | `'Offline'` |
| `machine_status_resolver.dart` `label` | `'Stopped'` | `'Offline'` |

The Stopped filter chip on the Machines screen was already removed
in Phase 3.5; remains absent.

The machine form's Status dropdown filters `MachineStatus.stopped`
out of its options (`.where((s) => s != MachineStatus.stopped)`)
and remaps any legacy stored value to `MachineStatus.offline` on
display. So the user can never *select* Stopped, and any pre-existing
stored value renders as "Offline".

### 5. Machine Start / Machine Stop hidden from Machine Details ✅

`machine_detail_screen.dart` — `grep "Start machine\|Stop machine\|
startMachine\|stopMachine"` returns zero matches in any screen or
widget file. The Start/Stop button was removed in Phase 3.5; this
pass confirms it stays gone.

The `AppState.startMachine` / `stopMachine` methods are preserved
because the simulator (which runs when Firebase is in offline/demo
mode) calls them internally to drive demo machine state. UI cannot
reach them.

### 6. Add User reliability with Firebase + Cloudinary ✅

`lib/widgets/form_sheets.dart` `showUserFormSheet(...)` covers all 7
reliability requirements from the brief:

1. **Field validation** — name and email validated on submit.
   `nameError` shows "Name is required." Email regex validates
   `RFC-5322`-shaped addresses; "Enter a valid email address."
   Duplicate-email check against `app.users` rejects with
   "A user with that email already exists." for new users.

2. **Saved correctly to Firebase** — uses `app.addUserAsync(item)`
   which performs the optimistic local insert + history event +
   `_firebase.upsertUser(user)` write. Firebase write errors are
   surfaced as form errors: "User saved locally but Firestore
   write failed: {detail}".

3. **Optional Cloudinary avatar upload** — when an image is picked,
   the form posts to `/upload/user-avatar` via
   `ImageUploadApiService` *before* the Firestore write. Returned
   `photoUrl` and `photoStoragePath` are written into the
   `AppUser` record in the same save.

4. **Partial-failure safety** — if the avatar upload fails (timeout,
   validation, backend down), the user record is **not** created.
   This avoids a half-baked user pointing at a non-existent photo.
   Firebase write errors after a successful upload do leave a local
   record but flag the failure clearly in the form's error banner.

5. **Loading state** — `bool busy` controls a top-of-form
   `LinearProgressIndicator` + "Saving user…" caption. All
   `TextField`s use `enabled: !busy`. Role dropdown's `onChanged`
   becomes null during busy. Submit button text flips to "Saving…"
   and `onPressed: null`.

6. **Success state** — on success, the sheet pops and a snackbar
   reads "User added: {name}" or "User updated: {name}".

7. **Error state** — every error path lands in a single inline
   error banner at the bottom of the form (red, with icon and
   detailed message). The form stays open so the user can correct
   and retry without losing their input.

Specific error messages cover:
- Image validation (size, type)
- Image upload timeout / connection
- Image upload backend unavailable
- Cloudinary returned non-2xx
- Firestore write failure
- Generic exception: "Could not save user: {error}"

Note on Auth-side limitation (unchanged from earlier phases): the
form writes to Firestore but does **not** create a Firebase Auth
user; that requires the `createAppUser` Cloud Function with the
Admin SDK. The synthetic `pending-…` uid is used until the real
person signs in for the first time, at which point
`loadOrCreateProfile` reconciles. This was already documented in
Phase 3.0 and remains unchanged.

### 7. flutter analyze ✅ (heuristic equivalent)

I have no Flutter SDK in this container, so I cannot run
`flutter analyze` directly. Instead I ran the Dart-syntax-aware
checks I rely on as a substitute:

- **Brace balance** (string-aware, comment-aware): all 38+ Dart
  files balance.
- **Unused imports / unused locals** (heuristic across files):
  zero real issues. The seven false positives the heuristic raised
  are all in `dt_colors.dart` (color tokens used in other files
  the script can't trace through).
- **Switch exhaustiveness** for `MachineStatus`: every `switch`
  covers `running`, `maintenance`, `fault`, `stopped`, `offline`.
- **`stopped` user-facing surface**: zero remaining 'Stopped' /
  'STOPPED' display strings. The enum value still exists in
  switches as defensive cases, all returning "Offline" /
  "OFFLINE".
- **`startMachine` / `stopMachine` UI references**: zero matches
  in `lib/screens/`, `lib/widgets/`.
- **Energy on dashboard**: zero matches (one trail comment only).

If `flutter analyze` flags something I can't see from a static
read, the most likely sites are imports added in this pass; let
me know and I'll adjust.

### 8. ZIP delivered

See `DigitalTwin-Phase3-6.zip`.

---

## Files changed this pass

Most of the work for items 1–6 was already on disk from Phase 3.6
prep. This pass made one additional code change and a thorough
verification sweep:

- `lib/services/machine_status_resolver.dart` — `label(MachineStatus.stopped)`
  changed from `'Stopped'` to `'Offline'` for consistency with the
  rest of the UI.
- `CHANGELOG.md` — rewritten to honestly document the verification
  + the one additional fix.

---

## Untouched (verified)

- All MQTT services and the multi-machine publisher
- All Firebase services (`firebase_sync_service.dart`,
  `firebase_user_admin_service.dart`)
- Image upload (FastAPI + Cloudinary, `image_upload_api_service.dart`)
- All theme files
- `routes.dart`, `app.dart`, `main.dart`, `pubspec.yaml`,
  `firestore.rules`
- All other models
- Login flow (Phase 3.5 friendly error mapping preserved)
- 15 `_blockWrite` defense-in-depth gates on AppState mutations
- `MachineStatusResolver` 30 s window + ingest/addReport hooks
- Periodic 15 s status sweep + history transition recording
- Sensor metadata catalog (16 sensors)
- "More sensors" sheet on Machine Detail
- "AI Root Causes (SHAP Top 3)" promoted to the slot above Live
  Telemetry
- "Predictive Main." KPI label

---

## Backend-required (unchanged from earlier phases)

| Feature | Status |
|---|---|
| Real Auth user creation from Add User | Cloud Function with Admin SDK |
| Push notifications | FCM |
| Camera capture | One-line change in form_sheets |
| Image compression beyond `imageQuality: 85` | Add `flutter_image_compress` |
| AI/RUL real inference | Wire Flutter → `backend/main.py` `/predict` |
| Forgot-password / no-account → admin request queue | Firestore collection + admin Inbox UI + Cloud Function |
