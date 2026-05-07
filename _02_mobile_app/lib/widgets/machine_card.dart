import 'package:flutter/material.dart';

import '../models/machine.dart';
import '../theme/dt_colors.dart';

class MachineCard extends StatelessWidget {
  final Machine machine;
  final VoidCallback? onTap;
  const MachineCard({super.key, required this.machine, this.onTap});

  @override
  Widget build(BuildContext context) {
    return Card(
      color: DT.surface(0.18),
      child: ListTile(
        onTap: onTap,
        title: Text(machine.name, style: const TextStyle(color: Colors.white)),
        subtitle: Text(machine.code, style: TextStyle(color: DT.muted(0.55))),
      ),
    );
  }
}
