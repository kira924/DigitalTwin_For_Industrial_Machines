import os
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import cloudinary
import cloudinary.uploader
import firebase_admin
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as fb_auth
from firebase_admin import credentials
from firebase_admin import firestore as fb_firestore
from pydantic import BaseModel

# Always load backend/.env from the same folder as this file
ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=ENV_PATH, override=True)

CLOUDINARY_URL = (os.getenv("CLOUDINARY_URL") or "").strip()

if not CLOUDINARY_URL:
    raise RuntimeError("Missing CLOUDINARY_URL in backend/.env")

# Expected format:
# CLOUDINARY_URL=cloudinary://API_KEY:API_SECRET@CLOUD_NAME
parsed = urlparse(CLOUDINARY_URL)

if parsed.scheme != "cloudinary":
    raise RuntimeError("Invalid CLOUDINARY_URL. It must start with cloudinary://")

CLOUDINARY_API_KEY = unquote(parsed.username or "").strip()
CLOUDINARY_API_SECRET = unquote(parsed.password or "").strip()
CLOUDINARY_CLOUD_NAME = unquote(parsed.hostname or "").strip()

if not CLOUDINARY_CLOUD_NAME:
    raise RuntimeError("Missing cloud name inside CLOUDINARY_URL")

if not CLOUDINARY_API_KEY:
    raise RuntimeError("Missing API key inside CLOUDINARY_URL")

if not CLOUDINARY_API_SECRET:
    raise RuntimeError("Missing API secret inside CLOUDINARY_URL")

cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET,
    secure=True,
)

app = FastAPI(
    title="Digital Twin Image Upload API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Development only. Restrict later for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Digital Twin Image Upload API is running",
        "docs": "http://127.0.0.1:8000/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "digital-twin-image-upload",
    }


@app.get("/debug-cloudinary")
def debug_cloudinary():
    return {
        "cloud_name": repr(CLOUDINARY_CLOUD_NAME),
        "api_key_present": bool(CLOUDINARY_API_KEY),
        "api_secret_present": bool(CLOUDINARY_API_SECRET),
        "cloudinary_url_present": bool(CLOUDINARY_URL),
        "env_path": str(ENV_PATH),
    }


def validate_image(file: UploadFile, data: bytes) -> None:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Please choose JPG, PNG, or WEBP.",
        )

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image is too large. Please choose an image under 5MB.",
        )


def safe_public_id(prefix: str, owner_id: str) -> str:
    clean_owner_id = (
        owner_id.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )

    timestamp = int(time.time())
    return f"{prefix}_{clean_owner_id}_{timestamp}"


async def upload_to_cloudinary(
    *,
    file: UploadFile,
    folder: str,
    public_id: str,
) -> dict:
    data = await file.read()
    validate_image(file, data)

    try:
        result = cloudinary.uploader.upload(
            BytesIO(data),
            folder=folder,
            public_id=public_id,
            resource_type="image",
            overwrite=True,
            unique_filename=False,
        )

        secure_url = result.get("secure_url")
        uploaded_public_id = result.get("public_id")

        if not secure_url or not uploaded_public_id:
            raise HTTPException(
                status_code=500,
                detail="Cloudinary upload succeeded but did not return a valid URL/path.",
            )

        return {
            "url": secure_url,
            "path": uploaded_public_id,
            "bytes": result.get("bytes"),
            "format": result.get("format"),
            "width": result.get("width"),
            "height": result.get("height"),
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cloudinary upload failed: {str(e)}",
        )


@app.post("/upload/machine-image")
async def upload_machine_image(
    machine_id: str = Form(...),
    file: UploadFile = File(...),
):
    machine_id = machine_id.strip()

    if not machine_id:
        raise HTTPException(status_code=400, detail="machine_id is required")

    public_id = safe_public_id("machine", machine_id)

    upload_result = await upload_to_cloudinary(
        file=file,
        folder=f"digital-twin/machine-images/{machine_id}",
        public_id=public_id,
    )

    return {
        "machineId": machine_id,
        "imageUrl": upload_result["url"],
        "imagePath": upload_result["path"],
        "metadata": {
            "bytes": upload_result.get("bytes"),
            "format": upload_result.get("format"),
            "width": upload_result.get("width"),
            "height": upload_result.get("height"),
        },
    }


@app.post("/upload/user-avatar")
async def upload_user_avatar(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    user_id = user_id.strip()

    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    public_id = safe_public_id("avatar", user_id)

    upload_result = await upload_to_cloudinary(
        file=file,
        folder=f"digital-twin/user-avatars/{user_id}",
        public_id=public_id,
    )

    return {
        "userId": user_id,
        "photoUrl": upload_result["url"],
        "photoPath": upload_result["path"],
        "metadata": {
            "bytes": upload_result.get("bytes"),
            "format": upload_result.get("format"),
            "width": upload_result.get("width"),
            "height": upload_result.get("height"),
        },
    }


@app.post("/upload/report-attachment")
async def upload_report_attachment(
    machine_id: str = Form(...),
    report_id: str = Form(...),
    file: UploadFile = File(...),
):
    machine_id = machine_id.strip()
    report_id = report_id.strip()

    if not machine_id:
        raise HTTPException(status_code=400, detail="machine_id is required")

    if not report_id:
        raise HTTPException(status_code=400, detail="report_id is required")

    public_id = safe_public_id("report", f"{machine_id}_{report_id}")

    upload_result = await upload_to_cloudinary(
        file=file,
        folder=f"digital-twin/report-attachments/{machine_id}/{report_id}",
        public_id=public_id,
    )

    return {
        "machineId": machine_id,
        "reportId": report_id,
        "attachmentUrl": upload_result["url"],
        "attachmentPath": upload_result["path"],
        "metadata": {
            "bytes": upload_result.get("bytes"),
            "format": upload_result.get("format"),
            "width": upload_result.get("width"),
            "height": upload_result.get("height"),
        },
    }


@app.delete("/delete-image")
async def delete_image(
    public_id: str = Query(..., description="Cloudinary public_id / imagePath"),
):
    public_id = public_id.strip()

    if not public_id:
        raise HTTPException(status_code=400, detail="public_id is required")

    try:
        result = cloudinary.uploader.destroy(public_id)

        return {
            "publicId": public_id,
            "deleted": result.get("result") == "ok",
            "result": result,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cloudinary delete failed: {str(e)}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Firebase Admin SDK — lazy, safe initialisation
# ─────────────────────────────────────────────────────────────────────────────
# Set  FIREBASE_SERVICE_ACCOUNT_PATH  in backend/.env to the absolute (or
# relative-to-backend/) path of your Firebase service-account JSON file.
# See backend/.env.example for a template.

_firebase_ready = False


def _ensure_firebase() -> None:
    """Initialise the Firebase Admin SDK once per process lifetime.

    Raises HTTPException(503) if the env-var or file is missing so the
    caller gets a clear error rather than a traceback.
    """
    global _firebase_ready
    if _firebase_ready:
        return

    sa_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "").strip()
    if not sa_path:
        raise HTTPException(
            status_code=503,
            detail=(
                "Firebase Admin SDK is not configured on this server. "
                "Set FIREBASE_SERVICE_ACCOUNT_PATH in backend/.env to the "
                "path of your Firebase service-account JSON file."
            ),
        )

    resolved = Path(sa_path)
    # Allow paths relative to the backend/ directory.
    if not resolved.is_absolute():
        resolved = Path(__file__).parent / resolved

    if not resolved.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Service-account file not found: {resolved}",
        )

    try:
        # Guard against double-init on uvicorn --reload.
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(str(resolved))
            firebase_admin.initialize_app(cred)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Firebase Admin SDK init failed: {exc}",
        )

    _firebase_ready = True


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/create-user
# ─────────────────────────────────────────────────────────────────────────────

_VALID_ROLES = {"admin", "engineer", "viewer"}


class CreateUserRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str
    assignedArea: str = ""
    photoUrl: str = ""
    createdByUid: str = ""


@app.post("/admin/create-user")
async def create_user(body: CreateUserRequest):
    """Create a Firebase Auth account + Firestore profile for a new app user.

    Called by the Flutter admin panel.  The password is used only to create
    the Auth account and is never written to Firestore.

    On Firestore write failure the Auth account is deleted as a rollback so
    the system is never left in a half-created state.
    """
    # ── Input validation ────────────────────────────────────────────────────
    name = body.name.strip()
    email = body.email.strip().lower()
    password = body.password          # never logged or stored
    role = body.role.strip().lower()

    if not name:
        raise HTTPException(status_code=400, detail="name is required.")
    if not email:
        raise HTTPException(status_code=400, detail="email is required.")
    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="password must be at least 6 characters.",
        )
    if role not in _VALID_ROLES:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of: {', '.join(sorted(_VALID_ROLES))}.",
        )

    # ── Firebase Admin init ─────────────────────────────────────────────────
    _ensure_firebase()

    # ── 1. Create Firebase Auth user ────────────────────────────────────────
    try:
        auth_user = fb_auth.create_user(
            email=email,
            password=password,
            display_name=name,
        )
    except fb_auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=409,
            detail=f"A Firebase Auth account for '{email}' already exists.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase Auth error: {exc}",
        )

    uid = auth_user.uid
    now = datetime.now(timezone.utc)

    # ── 2. Write Firestore profile ──────────────────────────────────────────
    user_doc = {
        "id": uid,
        "name": name,
        "email": email,
        "role": role,
        "assignedArea": body.assignedArea.strip(),
        "photoUrl": body.photoUrl.strip(),
        "photoStoragePath": "",
        "isActive": True,
        "isPending": False,
        "createdAt": now,
        "updatedAt": now,
        "createdBy": body.createdByUid.strip(),
        "lastLoginAt": None,
    }

    try:
        db = fb_firestore.client()
        db.collection("users").document(uid).set(user_doc)
    except Exception as exc:
        # ── Rollback: delete the Auth account we just created ──────────────
        try:
            fb_auth.delete_user(uid)
        except Exception:
            pass  # best-effort; log if you have a logger
        raise HTTPException(
            status_code=500,
            detail=(
                f"Firestore write failed — Auth account rolled back. "
                f"Details: {exc}"
            ),
        )

    return {
        "uid": uid,
        "email": email,
        "name": name,
        "role": role,
        "status": "created",
    }


# ─────────────────────────────────────────────────────────────────────────────
# POST /admin/delete-user
# ─────────────────────────────────────────────────────────────────────────────

class DeleteUserRequest(BaseModel):
    uid: str
    softDelete: bool = True


@app.post("/admin/delete-user")
async def delete_user(body: DeleteUserRequest):
    """Remove a user's Firebase Auth account and soft-delete their Firestore
    profile.

    Behaviour:
    - Auth account is deleted unconditionally (if it exists).
    - If the Auth account is not found, we continue gracefully and still
      mark the Firestore document as deleted.
    - Firestore document is soft-deleted by default (isActive=False,
      isDeleted=True, deletedAt set).  Pass softDelete=false to hard-delete
      the document instead.
    - If the Firestore update fails after a successful Auth delete a
      'partial' status is returned with a warning so the caller can
      surface a clear message and retry if needed.
    """
    uid = body.uid.strip()
    if not uid:
        raise HTTPException(status_code=400, detail="uid is required.")

    _ensure_firebase()

    # ── 1. Delete Firebase Auth user ────────────────────────────────────────
    auth_deleted = False
    auth_not_found = False
    try:
        fb_auth.delete_user(uid)
        auth_deleted = True
    except fb_auth.UserNotFoundError:
        # Auth account may never have been created (e.g. pending- uid) —
        # treat this as non-fatal and still clean up Firestore.
        auth_not_found = True
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Firebase Auth delete failed: {exc}",
        )

    # ── 2. Update Firestore document ────────────────────────────────────────
    now = datetime.now(timezone.utc)
    db = fb_firestore.client()
    doc_ref = db.collection("users").document(uid)

    try:
        if body.softDelete:
            doc_ref.set(
                {
                    "isActive": False,
                    "isDeleted": True,
                    "deletedAt": now,
                },
                merge=True,
            )
        else:
            doc_ref.delete()
    except Exception as exc:
        # Auth account was already deleted — return partial success so the
        # Flutter client can surface a descriptive message.
        return {
            "uid": uid,
            "authDeleted": auth_deleted,
            "authNotFound": auth_not_found,
            "firestoreUpdated": False,
            "status": "partial",
            "warning": (
                f"Auth account {'deleted' if auth_deleted else 'not found'}. "
                f"Firestore update failed: {exc}"
            ),
        }

    return {
        "uid": uid,
        "authDeleted": auth_deleted,
        "authNotFound": auth_not_found,
        "firestoreUpdated": True,
        "status": "deleted",
    }