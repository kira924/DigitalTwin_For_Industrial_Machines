// Compatibility shim. Phase 3.1 had a `storage_service.dart` backed by
// Firebase Storage. Phase 3.2 swaps the binary blob path to a
// FastAPI + Cloudinary backend. The implementation moved to
// `image_upload_api_service.dart` and is re-exported here so the
// existing imports in `state.dart`, `form_sheets.dart`, and
// `admin_screen.dart` keep compiling unchanged.
//
// New code should import `image_upload_api_service.dart` directly.

export 'image_upload_api_service.dart';
