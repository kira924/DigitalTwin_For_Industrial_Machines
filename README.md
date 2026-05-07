# Smart Factory Digital Twin App

This package is a rebuilt Flutter UI for factory machine monitoring and management.

## Implemented product areas

- Responsive **Machines dashboard**
  - search
  - status filters
  - machine cards with live summary
  - admin-only add machine action
- **Machine detail screen**
  - hero image
  - status summary
  - KPI tiles
  - performance chart
  - sensor list
  - status timeline
  - quick actions: report issue, schedule maintenance, start/stop, edit machine
- **Factory dashboard**
  - time filters: 24h, 7d, 30d, 90d, 1y, custom
  - OEE, energy, operating time, downtime
  - machine-state breakdown
  - downtime reasons breakdown
  - top 5 downtime machines
  - top 5 energy machines
- **Admin / Settings**
  - add, edit, delete machine
  - add, edit, delete user
  - role switching preview
  - dark/light mode toggle
  - Arabic RTL toggle
  - offline mode toggle
  - notifications toggle
  - threshold controls
- **Alerts** screen
- **History** screen
- Local industrial image assets under `assets/images/`

## Notes

- This environment did not have Flutter SDK installed, so the project could not be compiled here.
- The UI and state structure were rebuilt directly in source code and packaged for local testing.
- Real backend auth, push notifications, and persistent offline storage are scaffolded at the UI/state level but still need production service wiring.

## Main files changed

- `lib/app/app.dart`
- `lib/app/routes.dart`
- `lib/app/state.dart`
- `lib/models/machine.dart`
- `lib/models/app_user.dart`
- `lib/data/machine_catalog.dart`
- `lib/data/user_catalog.dart`
- `lib/screens/login_screen.dart`
- `lib/screens/machines_screen.dart`
- `lib/screens/dashboard_screen.dart`
- `lib/screens/alerts_screen.dart`
- `lib/screens/history_screen.dart`
- `lib/screens/admin_screen.dart`
- `assets/images/*`
- `pubspec.yaml`
