import '../models/app_user.dart';

/// Seed users for the demo. Roles match the 3-role product taxonomy.
/// Old `operator` and `maintenance` accounts have been consolidated
/// under `engineer`.
List<AppUser> buildSeedUsers() => const <AppUser>[
  AppUser(
    id: 'u-01',
    name: 'Belal Saqer',
    email: 'belal19lol@gmail.com',
    role: AppRole.admin,
    assignedArea: 'Main Factory',
    isActive: true,
  ),
  AppUser(
    id: 'u-02',
    name: 'Alaa Engineer',
    email: 'alaa@factory.local',
    role: AppRole.engineer,
    assignedArea: 'Hall A',
    isActive: true,
  ),
  AppUser(
    id: 'u-03',
    name: 'Mariam Engineer',
    email: 'mariam@factory.local',
    role: AppRole.engineer,
    assignedArea: 'Hall B',
    isActive: true,
  ),
  AppUser(
    id: 'u-04',
    name: 'Observer View',
    email: 'viewer@factory.local',
    role: AppRole.viewer,
    assignedArea: 'Executive Floor',
    isActive: true,
  ),
];
