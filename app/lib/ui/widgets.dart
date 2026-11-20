import 'package:flutter/material.dart';

import '../data/scanner.dart';
import 'theme.dart';

/// An identifier shown character by character. A `?` is a character the pixels did not support:
/// it is drawn as a red box, never replaced by the decoder's guess.
class MaskedText extends StatelessWidget {
  const MaskedText(this.text, {super.key, this.size = 28, this.color, this.hints = const {}});
  final String text;
  final double size;
  final Color? color;

  /// position -> the character the recogniser's evidence AND the plate / VIN format point to for a `?`.
  /// Drawn as an amber, clearly unconfirmed letter in that slot instead of the red unknown box.
  final Map<int, String> hints;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final base = TextStyle(fontFamily: 'monospace', fontWeight: FontWeight.w700, fontSize: size, letterSpacing: 1.5, color: color ?? cs.onSurface);
    final chars = text.characters.toList();
    return Semantics(
      label: [for (var i = 0; i < chars.length; i++) chars[i] == '?' ? (hints[i] != null ? ' probably ${hints[i]} ' : ' unknown ') : chars[i]].join(),
      child: ExcludeSemantics(
        child: Wrap(crossAxisAlignment: WrapCrossAlignment.center, children: [
          for (var i = 0; i < chars.length; i++)
            if (chars[i] == '?' && hints[i] != null)
              Container(
                margin: const EdgeInsets.symmetric(horizontal: 1.5),
                padding: EdgeInsets.symmetric(horizontal: size * 0.12),
                decoration: BoxDecoration(color: Colors.amber.withValues(alpha: 0.18), borderRadius: BorderRadius.circular(5), border: Border.all(color: Colors.amber.shade800, width: 1.4)),
                child: Text(hints[i]!, key: const ValueKey('suggested-char'), style: base.copyWith(color: Colors.amber.shade800)),
              )
            else if (chars[i] == '?')
                Container(
                    margin: const EdgeInsets.symmetric(horizontal: 1.5),
                    padding: EdgeInsets.symmetric(horizontal: size * 0.18),
                    decoration: BoxDecoration(color: cs.errorContainer, borderRadius: BorderRadius.circular(5), border: Border.all(color: cs.error, width: 1.2)),
                    child: Text('?', key: const ValueKey('unknown-char'), style: base.copyWith(color: cs.error)),
                  )
            else
                Text(chars[i], style: base),
        ]),
      ),
    );
  }
}

class StatusChip extends StatelessWidget {
  const StatusChip(this.statusKey, {super.key, this.label, this.dense = false});
  final String statusKey;
  final String? label;
  final bool dense;

  @override
  Widget build(BuildContext context) {
    final t = toneFor(context, statusKey);
    return Container(
      padding: EdgeInsets.symmetric(horizontal: dense ? 8 : 10, vertical: dense ? 3 : 5),
      decoration: BoxDecoration(color: t.bg, borderRadius: BorderRadius.circular(20)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(t.icon, size: dense ? 14 : 16, color: t.fg),
        const SizedBox(width: 5),
        Flexible(child: Text(label ?? t.label, maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(color: t.fg, fontSize: dense ? 12 : 13, fontWeight: FontWeight.w600))),
      ]),
    );
  }
}

class InfoPill extends StatelessWidget {
  const InfoPill(this.text, {super.key, this.icon});
  final String text;
  final IconData? icon;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
      decoration: BoxDecoration(color: cs.surfaceContainerHighest, borderRadius: BorderRadius.circular(20)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        if (icon != null) ...[Icon(icon, size: 13, color: cs.onSurfaceVariant), const SizedBox(width: 4)],
        Flexible(child: Text(text, maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontSize: 12, color: cs.onSurfaceVariant))),
      ]),
    );
  }
}

class BulletList extends StatelessWidget {
  const BulletList(this.items, {super.key, this.icon = Icons.circle, this.color, this.size = 13});
  final List<String> items;
  final IconData icon;
  final Color? color;
  final double size;

  @override
  Widget build(BuildContext context) {
    final c = color ?? Theme.of(context).colorScheme.onSurfaceVariant;
    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      for (final s in items)
        Padding(
          padding: const EdgeInsets.only(bottom: 6),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Padding(padding: const EdgeInsets.only(top: 4), child: Icon(icon, size: icon == Icons.circle ? 6 : 15, color: c)),
            const SizedBox(width: 8),
            Expanded(child: Text(s, style: TextStyle(fontSize: size, height: 1.35))),
          ]),
        ),
    ]);
  }
}

// ------------------------------------------------------------------------------ dialogs
Future<bool> showConfirmDialog(
  BuildContext context, {
  required String title,
  required String message,
  String confirm = 'Continue',
  String cancel = 'Cancel',
  bool destructive = false,
  IconData? icon,
}) async {
  final cs = Theme.of(context).colorScheme;
  final r = await showDialog<bool>(
    context: context,
    builder: (ctx) => AlertDialog(
      icon: icon == null ? null : Icon(icon, color: destructive ? cs.error : cs.primary, size: 30),
      title: Text(title),
      content: Text(message),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, false), child: Text(cancel)),
        FilledButton(
          style: destructive ? FilledButton.styleFrom(backgroundColor: cs.error, foregroundColor: cs.onError) : null,
          onPressed: () => Navigator.pop(ctx, true),
          child: Text(confirm),
        ),
      ],
    ),
  );
  return r ?? false;
}

Future<void> showMessageDialog(BuildContext context, {required String title, required String message, IconData icon = Icons.info_outline, String button = 'OK'}) {
  return showDialog<void>(
    context: context,
    builder: (ctx) => AlertDialog(
      icon: Icon(icon, size: 30, color: Theme.of(ctx).colorScheme.primary),
      title: Text(title),
      content: Text(message),
      actions: [FilledButton(onPressed: () => Navigator.pop(ctx), child: Text(button))],
    ),
  );
}

enum ErrorAction { retry, close }

/// A readable explanation of why a scan failed, and what to do about it. Everything runs on the
/// phone, so the causes are the photos, the engine, or running out of memory - never a network.
Future<ErrorAction> showScanErrorDialog(BuildContext context, ScanException e, {bool canRetry = true}) async {
  final (title, help) = switch (e.kind) {
    ScanErrorKind.badImage => ("A photo couldn't be opened", 'Retake the photo and try again.'),
    ScanErrorKind.engine => ("The reading engine can't start", 'The models that read plates could not be loaded on this phone. Closing other apps frees memory; if it keeps happening, reinstall the app.'),
    ScanErrorKind.failed => ('The scan did not finish', 'Something went wrong while reading these photos. Try again; fewer photos use less memory.'),
    ScanErrorKind.cancelled => ('Cancelled', 'Nothing was kept.'),
  };
  final r = await showDialog<ErrorAction>(
    context: context,
    builder: (ctx) => AlertDialog(
      icon: Icon(e.kind == ScanErrorKind.cancelled ? Icons.info_outline : Icons.error_outline, size: 30, color: Theme.of(ctx).colorScheme.error),
      title: Text(title),
      content: SingleChildScrollView(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, mainAxisSize: MainAxisSize.min, children: [
          Text(help),
          if (e.detail != null && e.detail!.isNotEmpty)
            ExpansionTile(
              tilePadding: EdgeInsets.zero,
              title: const Text('Technical details', style: TextStyle(fontSize: 13)),
              children: [SelectableText(e.detail!, style: const TextStyle(fontFamily: 'monospace', fontSize: 11))],
            ),
        ]),
      ),
      actions: [
        TextButton(onPressed: () => Navigator.pop(ctx, ErrorAction.close), child: const Text('Close')),
        if (canRetry && e.kind != ScanErrorKind.cancelled) FilledButton(onPressed: () => Navigator.pop(ctx, ErrorAction.retry), child: const Text('Try again')),
      ],
    ),
  );
  return r ?? ErrorAction.close;
}
