import 'package:flutter/material.dart';

class AppAlert {
  final String id;
  final String machineId;
  final String title;
  final String message;
  final DateTime time;
  final Color color;

  const AppAlert({
    required this.id,
    required this.machineId,
    required this.title,
    required this.message,
    required this.time,
    required this.color,
  });
}
