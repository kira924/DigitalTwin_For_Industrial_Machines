import 'package:flutter/material.dart';

import '../widgets/dashboard_shell.dart';

/// Primary industrial dashboard route.
class DashboardScreen extends StatelessWidget {
  /// Creates the dashboard screen.
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0B1118),
      appBar: AppBar(
        title: const Text('Industrial Digital Twin'),
        centerTitle: false,
        backgroundColor: const Color(0xFF121A23),
        foregroundColor: Colors.white,
        elevation: 0,
      ),
      body: const DashboardShell(),
    );
  }
}