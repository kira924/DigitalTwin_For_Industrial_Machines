import 'package:flutter/material.dart';

import '../models/machine.dart';
import '../theme/dt_colors.dart';
import '../widgets/machine_image.dart';

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
        leading: ClipRRect(
          borderRadius: BorderRadius.circular(6),
          child: SizedBox(
            width: 48,
            height: 48,
            child: MachineImage(machine: machine, fit: BoxFit.cover),
          ),
        ),
        title: Text(machine.name, style: const TextStyle(color: Colors.white)),
        subtitle: Text(machine.code, style: TextStyle(color: DT.muted(0.55))),
      ),
    );
  }
}
