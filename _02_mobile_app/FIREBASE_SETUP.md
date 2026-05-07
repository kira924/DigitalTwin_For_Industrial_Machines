# Firebase setup for Smart Factory app

This project is now wired for:
- Firebase Authentication (email/password)
- Cloud Firestore sync for machines and users
- Firestore-backed factory settings
- Local cache fallback with SharedPreferences

## What was added
- `firebase_core`
- `firebase_auth`
- `cloud_firestore`
- `shared_preferences`
- `lib/firebase/firebase_options.dart`
- `lib/firebase/firebase_bootstrap.dart`
- `lib/services/firebase_sync_service.dart`

## Important
The app is shipped with **placeholder Firebase values**.
Until you replace them, the app runs in **demo mode** and shows a Firebase status message.

## 1) Create your Firebase project
1. Open Firebase Console
2. Create a project
3. Enable **Authentication > Email/Password**
4. Create **Cloud Firestore** in production or test mode

## 2) Register your apps
Register:
- Android app
- iOS app
- Web app (optional)
- macOS app (optional)

## 3) Copy your Firebase options
Open:
- `lib/firebase/firebase_options.dart`

Replace the placeholder values:
- `YOUR_PROJECT_ID`
- `YOUR_SENDER_ID`
- `YOUR_ANDROID_API_KEY`
- `YOUR_IOS_API_KEY`
- `YOUR_WEB_API_KEY`
- app ids for each platform

## 4) Create auth users
Create users in Firebase Authentication.
Then add matching profiles in Firestore collection:
- `users`

Each user document should look like:

```json
{
  "id": "u-01",
  "name": "Belal Saqer",
  "email": "belal19lol@gmail.com",
  "role": "admin",
  "assignedArea": "Main Factory",
  "enabled": true
}
```

## 5) Firestore collections used
- `machines`
- `users`
- `config/app`

## 6) First sync
After login as admin:
1. open Admin
2. use **Sync Now**
3. the app pushes current machines/users/settings into Firestore

## 7) Platform note
Desktop builds can still run in demo mode.
For real Firebase usage, Android / iOS / macOS / Web are the intended platforms in this step.
