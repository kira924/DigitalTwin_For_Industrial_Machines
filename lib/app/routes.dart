import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../screens/admin_screen.dart';
import '../screens/alerts_screen.dart';
import '../screens/dashboard_screen.dart';
import '../screens/history_screen.dart';
import '../screens/machines_screen.dart';
import '../theme/dt_colors.dart';
import '../theme/dt_tokens.dart';
import 'state.dart';

class AppShell extends StatelessWidget {
  const AppShell({super.key});

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final screen = switch (app.currentTab) {
      AppTab.dashboard => const DashboardScreen(),
      AppTab.history => const HistoryScreen(),
      AppTab.machines => const MachineListScreen(),
      AppTab.admin => const AdminScreen(),
      AppTab.alerts => const AlertsScreen(),
    };

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(child: screen),
          const Positioned(left: 0, right: 0, bottom: 0, child: _BottomNav()),
        ],
      ),
    );
  }
}

class _BottomNav extends StatelessWidget {
  const _BottomNav();

  @override
  Widget build(BuildContext context) {
    final app = context.watch<AppState>();
    final palette = DTPalette.of(context);

    final items = <({AppTab tab, IconData icon, String label})>[
      (
        tab: AppTab.dashboard,
        icon: Icons.dashboard_outlined,
        label: 'Dashboard',
      ),
      (
        tab: AppTab.machines,
        icon: Icons.precision_manufacturing_outlined,
        label: 'Machines',
      ),
      (
        tab: AppTab.alerts,
        icon: Icons.notifications_outlined,
        label: 'Alerts',
      ),
      (
        tab: AppTab.history,
        icon: Icons.history_rounded,
        label: 'History',
      ),
      (
        tab: AppTab.admin,
        icon: Icons.tune_rounded,
        // Bottom nav label is "Settings" for ALL roles. The page itself
        // still contains admin-only sections when the signed-in user is
        // Admin — but the nav label never reads "Admin".
        label: 'Settings',
      ),
    ];

    return SafeArea(
      top: false,
      child: Container(
        decoration: BoxDecoration(
          color: palette.canvas,
          border: Border(
            top: BorderSide(color: palette.borderSubtle, width: 1),
          ),
        ),
        child: SizedBox(
          height: 62,
          child: Row(
            children: items.map((item) {
              final active = app.currentTab == item.tab;
              final color = active
                  ? DTTokens.accentLive
                  : palette.textTertiary;
              return Expanded(
                child: InkWell(
                  onTap: () => context.read<AppState>().setTab(item.tab),
                  splashColor: DT.cyan.alphaF(0.05),
                  highlightColor: Colors.transparent,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      // Top accent bar (3px) for active tab
                      AnimatedContainer(
                        duration: const Duration(milliseconds: 180),
                        width: active ? 28 : 0,
                        height: 2,
                        decoration: BoxDecoration(
                          color: DTTokens.accentLive,
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Icon(item.icon, color: color, size: 22),
                      const SizedBox(height: 3),
                      Text(
                        item.label,
                        style: TextStyle(
                          color: color,
                          fontSize: 10,
                          fontWeight: active ? FontWeight.w700 : FontWeight.w500,
                          letterSpacing: 0.2,
                        ),
                      ),
                    ],
                  ),
                ),
              );
            }).toList(growable: false),
          ),
        ),
      ),
    );
  }
}
