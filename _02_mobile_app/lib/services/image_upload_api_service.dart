import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart';

import '../config/api_config.dart';

/// Maximum allowed image size in bytes: 5 MB.
const int kMaxImageSizeBytes = 5 * 1024 * 1024;

/// Backwards-compatible timeout name.
/// The old service used Firebase Storage, but now this timeout is for
/// the FastAPI + Cloudinary upload backend.
const Duration kStorageUploadTimeout = ApiConfig.uploadTimeout;

const Set<String> kAllowedImageExtensions = {
  '.jpg',
  '.jpeg',
  '.png',
  '.webp',
};

const Set<String> _allowedMimeTypes = {
  'image/jpeg',
  'image/png',
  'image/webp',
};

class UploadResult {
  final String url;
  final String path;

  const UploadResult({
    required this.url,
    required this.path,
  });
}

class ImageValidationError implements Exception {
  final String message;

  const ImageValidationError(this.message);

  @override
  String toString() => message;
}

class StorageUploadError implements Exception {
  final String message;

  const StorageUploadError(this.message);

  @override
  String toString() => message;
}

/// Validates image size and extension before upload.
void validateImageFile({
  required String filename,
  required int sizeBytes,
}) {
  final lower = filename.toLowerCase().trim();
  final dotIndex = lower.lastIndexOf('.');
  final ext = dotIndex >= 0 ? lower.substring(dotIndex) : '';

  if (!kAllowedImageExtensions.contains(ext)) {
    throw const ImageValidationError(
      'Unsupported image type. Please choose JPG, PNG, or WEBP.',
    );
  }

  if (sizeBytes > kMaxImageSizeBytes) {
    throw const ImageValidationError(
      'Image is too large. Please choose an image under 5MB.',
    );
  }
}

/// Detects MIME type from file bytes first, then filename fallback.
///
/// This is important for Flutter Web/Desktop because selected files may be
/// sent as application/octet-stream unless we explicitly set contentType.
String _detectMimeType({
  required String filename,
  required Uint8List bytes,
}) {
  final detected = lookupMimeType(
    filename,
    headerBytes: bytes,
  );

  if (detected != null && _allowedMimeTypes.contains(detected)) {
    return detected;
  }

  final lower = filename.toLowerCase().trim();

  if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) {
    return 'image/jpeg';
  }

  if (lower.endsWith('.png')) {
    return 'image/png';
  }

  if (lower.endsWith('.webp')) {
    return 'image/webp';
  }

  throw const ImageValidationError(
    'Unsupported image type. Please choose JPG, PNG, or WEBP.',
  );
}

MediaType _mediaTypeFromMime(String mimeType) {
  final parts = mimeType.split('/');

  if (parts.length != 2) {
    throw const ImageValidationError(
      'Unsupported image type. Please choose JPG, PNG, or WEBP.',
    );
  }

  return MediaType(parts[0], parts[1]);
}

/// Image upload service backed by the FastAPI + Cloudinary backend.
///
/// Firebase Storage is not used here.
/// Firebase Auth / Firestore can remain as they are; this service only
/// handles binary image upload/delete through the backend API.
class ImageUploadApiService {
  ImageUploadApiService({http.Client? client}) : _client = client ?? http.Client();

  final http.Client _client;

  // ── Machine images ───────────────────────────────────────────────────────

  Future<UploadResult> uploadMachineImage({
    required String machineId,
    required File file,
    void Function(double progress01)? onProgress,
  }) async {
    final size = await file.length();
    final filename = file.uri.pathSegments.isNotEmpty
        ? file.uri.pathSegments.last
        : 'machine-image.png';

    validateImageFile(
      filename: filename,
      sizeBytes: size,
    );

    final bytes = await file.readAsBytes();

    return _postMachineImage(
      machineId: machineId,
      bytes: bytes,
      filename: filename,
      onProgress: onProgress,
    );
  }

  Future<UploadResult> uploadMachineImageBytes({
    required String machineId,
    required Uint8List bytes,
    String filename = 'machine-image.png',
    void Function(double progress01)? onProgress,
  }) async {
    validateImageFile(
      filename: filename,
      sizeBytes: bytes.length,
    );

    return _postMachineImage(
      machineId: machineId,
      bytes: bytes,
      filename: filename,
      onProgress: onProgress,
    );
  }

  // ── User avatars ─────────────────────────────────────────────────────────

  Future<UploadResult> uploadUserAvatar({
    required String uid,
    required File file,
    void Function(double progress01)? onProgress,
  }) async {
    final size = await file.length();
    final filename = file.uri.pathSegments.isNotEmpty
        ? file.uri.pathSegments.last
        : 'avatar.png';

    validateImageFile(
      filename: filename,
      sizeBytes: size,
    );

    final bytes = await file.readAsBytes();

    return _postUserAvatar(
      uid: uid,
      bytes: bytes,
      filename: filename,
      onProgress: onProgress,
    );
  }

  Future<UploadResult> uploadUserAvatarBytes({
    required String uid,
    required Uint8List bytes,
    String filename = 'avatar.png',
    void Function(double progress01)? onProgress,
  }) async {
    validateImageFile(
      filename: filename,
      sizeBytes: bytes.length,
    );

    return _postUserAvatar(
      uid: uid,
      bytes: bytes,
      filename: filename,
      onProgress: onProgress,
    );
  }

  // ── Report attachments ───────────────────────────────────────────────────

  Future<UploadResult> uploadReportAttachment({
    required String machineId,
    required String reportId,
    required File file,
    void Function(double progress01)? onProgress,
  }) async {
    final size = await file.length();
    final filename = file.uri.pathSegments.isNotEmpty
        ? file.uri.pathSegments.last
        : 'report-attachment.png';

    validateImageFile(
      filename: filename,
      sizeBytes: size,
    );

    final bytes = await file.readAsBytes();

    return _postReportAttachment(
      machineId: machineId,
      reportId: reportId,
      bytes: bytes,
      filename: filename,
      onProgress: onProgress,
    );
  }

  Future<UploadResult> uploadReportAttachmentBytes({
    required String machineId,
    required String reportId,
    required Uint8List bytes,
    String filename = 'report-attachment.png',
    void Function(double progress01)? onProgress,
  }) async {
    validateImageFile(
      filename: filename,
      sizeBytes: bytes.length,
    );

    return _postReportAttachment(
      machineId: machineId,
      reportId: reportId,
      bytes: bytes,
      filename: filename,
      onProgress: onProgress,
    );
  }

  // ── Delete image ─────────────────────────────────────────────────────────

  Future<bool> deleteByPath(String path) async {
    if (path.trim().isEmpty) return false;

    try {
      final uri = Uri.parse(
        '${ApiConfig.baseUrl}${ApiConfig.deleteImageEndpoint}'
        '?public_id=${Uri.encodeQueryComponent(path)}',
      );

      final res = await _client.delete(uri).timeout(ApiConfig.deleteTimeout);

      if (res.statusCode >= 200 && res.statusCode < 300) {
        try {
          final parsed = jsonDecode(res.body);
          if (parsed is Map && parsed['deleted'] is bool) {
            return parsed['deleted'] as bool;
          }
        } catch (_) {
          return true;
        }

        return true;
      }

      return false;
    } catch (_) {
      return false;
    }
  }

  Future<bool> deleteByUrl(String url) async => false;

  // ── Internal upload methods ──────────────────────────────────────────────

  Future<UploadResult> _postMachineImage({
    required String machineId,
    required Uint8List bytes,
    required String filename,
    void Function(double progress01)? onProgress,
  }) {
    return _postUpload(
      uri: Uri.parse('${ApiConfig.baseUrl}${ApiConfig.machineImageEndpoint}'),
      fields: {'machine_id': machineId},
      bytes: bytes,
      filename: filename,
      urlField: 'imageUrl',
      pathField: 'imagePath',
      onProgress: onProgress,
    );
  }

  Future<UploadResult> _postUserAvatar({
    required String uid,
    required Uint8List bytes,
    required String filename,
    void Function(double progress01)? onProgress,
  }) {
    return _postUpload(
      uri: Uri.parse('${ApiConfig.baseUrl}${ApiConfig.userAvatarEndpoint}'),
      fields: {'user_id': uid},
      bytes: bytes,
      filename: filename,
      urlField: 'photoUrl',
      pathField: 'photoPath',
      onProgress: onProgress,
    );
  }

  Future<UploadResult> _postReportAttachment({
    required String machineId,
    required String reportId,
    required Uint8List bytes,
    required String filename,
    void Function(double progress01)? onProgress,
  }) {
    return _postUpload(
      uri: Uri.parse('${ApiConfig.baseUrl}/upload/report-attachment'),
      fields: {
        'machine_id': machineId,
        'report_id': reportId,
      },
      bytes: bytes,
      filename: filename,
      urlField: 'attachmentUrl',
      pathField: 'attachmentPath',
      onProgress: onProgress,
    );
  }

  Future<UploadResult> _postUpload({
    required Uri uri,
    required Map<String, String> fields,
    required Uint8List bytes,
    required String filename,
    required String urlField,
    required String pathField,
    void Function(double progress01)? onProgress,
  }) async {
    onProgress?.call(0.0);

    final mimeType = _detectMimeType(
      filename: filename,
      bytes: bytes,
    );

    final req = http.MultipartRequest('POST', uri);
    req.fields.addAll(fields);

    req.files.add(
      http.MultipartFile.fromBytes(
        'file',
        bytes,
        filename: filename,
        contentType: _mediaTypeFromMime(mimeType),
      ),
    );

    http.StreamedResponse streamed;

    try {
      streamed = await _client.send(req).timeout(
        ApiConfig.uploadTimeout,
        onTimeout: () {
          throw TimeoutException(
            'Upload timed out. Please check your connection or '
            'whether the image upload backend is running.',
            ApiConfig.uploadTimeout,
          );
        },
      );
    } on TimeoutException catch (e) {
      throw StorageUploadError(e.message ?? 'Upload timed out.');
    } on SocketException catch (e) {
      throw StorageUploadError(
        'Could not reach the image upload backend at ${uri.host}:${uri.port}. '
        'Make sure FastAPI is running. (${e.osError?.message ?? e.message})',
      );
    } on http.ClientException catch (e) {
      throw StorageUploadError('Upload failed: ${e.message}');
    } catch (e) {
      throw StorageUploadError('Upload failed: $e');
    }

    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      String detail = 'HTTP ${streamed.statusCode}';

      try {
        final parsed = jsonDecode(body);

        if (parsed is Map && parsed['detail'] != null) {
          detail = parsed['detail'].toString();
        }
      } catch (_) {
        // Keep default detail.
      }

      throw StorageUploadError('Upload failed: $detail');
    }

    onProgress?.call(1.0);

    Map<String, dynamic> parsed;

    try {
      parsed = jsonDecode(body) as Map<String, dynamic>;
    } catch (_) {
      throw const StorageUploadError(
        'Upload completed but the backend response could not be parsed.',
      );
    }

    final url = parsed[urlField]?.toString();
    final path = parsed[pathField]?.toString();

    if (url == null || url.isEmpty || path == null || path.isEmpty) {
      throw const StorageUploadError(
        'Upload completed but the backend did not return a URL/path.',
      );
    }

    return UploadResult(
      url: url,
      path: path,
    );
  }
}

/// Backwards-compatible alias.
/// Existing code that still calls StorageService() will keep working.
typedef StorageService = ImageUploadApiService;