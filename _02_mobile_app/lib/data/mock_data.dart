enum ReportKind { performance, alerts, maintenance, summary }

class ReportItem {
  final String id;
  final String title;
  final ReportKind kind;
  final String period;
  final String date;
  final String size;

  const ReportItem({
    required this.id,
    required this.title,
    required this.kind,
    required this.period,
    required this.date,
    required this.size,
  });
}