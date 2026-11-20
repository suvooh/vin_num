import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../crash.dart';

/// Shows the previous run's crash report (if any) with a copy button. Used at app start.
Future<void> showPendingCrashReport(BuildContext context) async {
  final report = await CrashLog.pendingReport();
  if (report == null || !context.mounted) return;
  await showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (ctx) => AlertDialog(
      icon: const Icon(Icons.bug_report, color: Colors.red),
      title: const Text('The app closed unexpectedly last time'),
      content: SizedBox(
        width: double.maxFinite,
        child: SingleChildScrollView(child: SelectableText(report, style: const TextStyle(fontFamily: 'monospace', fontSize: 11))),
      ),
      actions: [
        TextButton(
          onPressed: () async {
            await Clipboard.setData(ClipboardData(text: report));
            if (ctx.mounted) ScaffoldMessenger.of(ctx).showSnackBar(const SnackBar(content: Text('Copied to clipboard')));
          },
          child: const Text('Copy'),
        ),
        FilledButton(
          onPressed: () async {
            await CrashLog.clear();
            if (ctx.mounted) Navigator.of(ctx).pop();
          },
          child: const Text('Dismiss'),
        ),
      ],
    ),
  );
}
