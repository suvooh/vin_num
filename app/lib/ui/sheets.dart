
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../data/models.dart';
import 'explain.dart';
import 'theme.dart';
import 'widgets.dart';

// ======================================================================== field review sheet
/// Opens the review sheet for one reading. Returns the value the person confirmed, or null.
Future<String?> showFieldSheet(BuildContext context, FieldResult f, {String? confirmed}) {
  return showModalBottomSheet<String>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    builder: (ctx) => Padding(
      padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(ctx).bottom),
      child: _FieldSheet(f: f, confirmed: confirmed),
    ),
  );
}

class _FieldSheet extends StatefulWidget {
  const _FieldSheet({required this.f, this.confirmed});
  final FieldResult f;
  final String? confirmed;
  @override
  State<_FieldSheet> createState() => _FieldSheetState();
}

class _FieldSheetState extends State<_FieldSheet> {
  late final TextEditingController ctl;
  String? error;

  @override
  void initState() {
    super.initState();
    // never pre-fill an unknown: the person types what they can see on the vehicle
    final start = widget.confirmed ?? (widget.f.isAccepted ? (widget.f.value ?? '') : '');
    ctl = TextEditingController(text: start);
  }

  @override
  void dispose() {
    ctl.dispose();
    super.dispose();
  }

  void _submit() {
    final e = validateIdentifier(ctl.text, plate: widget.f.isPlate);
    if (e != null) {
      setState(() => error = e);
      return;
    }
    Navigator.pop(context, ctl.text.trim().toUpperCase());
  }

  @override
  Widget build(BuildContext context) {
    final f = widget.f;
    final cs = Theme.of(context).colorScheme;
    final reasons = humanReasons(f.reasons);
    final alts = f.alternatives.where((a) => a.text != f.value && !a.text.contains('?')).take(4).toList();
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        Row(children: [
          Expanded(child: Text(f.isPlate ? 'Number plate' : 'VIN / chassis number', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700))),
          StatusChip(f.status, dense: true),
        ]),
        const SizedBox(height: 12),
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(color: cs.surfaceContainerHighest, borderRadius: BorderRadius.circular(14)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            MaskedText(f.shown, size: f.isPlate ? 30 : 20),
            const SizedBox(height: 8),
            Text('${(f.confidence * 100).round()}% confidence · read from photo ${f.image + 1}${f.views > 1 ? ' · ${f.views} views combined' : ''}', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
            if (f.unknownCount > 0)
              Padding(padding: const EdgeInsets.only(top: 6), child: Text('${f.unknownCount} unknown character${f.unknownCount == 1 ? '' : 's'} (?) could not be read from the photo and were not guessed.', style: TextStyle(color: cs.error, fontSize: 13))),
          ]),
        ),
        if (reasons.isNotEmpty) ...[
          const SizedBox(height: 16),
          Text(f.isAccepted ? 'Notes' : 'Why this needs review', style: const TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          BulletList(reasons, icon: Icons.info_outline, color: cs.primary, size: 14),
        ],
        if (f.guidance.isNotEmpty) ...[
          const SizedBox(height: 8),
          Text('For a better photo', style: const TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          BulletList(f.guidance, icon: Icons.tips_and_updates_outlined, color: Colors.amber.shade800, size: 14),
        ],
        for (final slot in f.unknownSlots.where((s) => s.options.isNotEmpty).take(4)) ...[
          const SizedBox(height: 12),
          Text('Character ${slot.position + 1} could be', style: const TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          Text(
            f.isPlate ? 'Only characters a plate can have in this position are listed. Tap one to use it, then check it against the plate.' : 'Only characters a VIN can have in this position are listed. Tap one to use it, then check it against the vehicle.',
            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13),
          ),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 8, children: [
            for (final o in slot.options)
              ActionChip(
                key: ValueKey('slot-${slot.position}-${o.char}'),
                avatar: slot.suggested == o.char ? const Icon(Icons.star, size: 16) : null,
                label: Text('${o.char}  ${(o.p * 100).round()}%', style: const TextStyle(fontFamily: 'monospace', fontWeight: FontWeight.w700)),
                onPressed: () => setState(() {
                  final base = f.value ?? f.masked;
                  final cur = ctl.text.length == base.length ? ctl.text : base; // successive taps compose
                  final chars = cur.split('');
                  if (slot.position < chars.length) chars[slot.position] = o.char;
                  ctl.text = chars.join();
                  error = null;
                }),
              ),
          ]),
        ],
        if (alts.isNotEmpty) ...[
          const SizedBox(height: 12),
          const Text('Other possible readings', style: TextStyle(fontWeight: FontWeight.w700)),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 8, children: [
            for (final a in alts)
              ActionChip(
                label: Text('${a.text}  ${(a.p * 100).round()}%', style: const TextStyle(fontFamily: 'monospace')),
                onPressed: () => setState(() {
                  ctl.text = a.text;
                  error = null;
                }),
              ),
          ]),
        ],
        const SizedBox(height: 18),
        Text(f.isAccepted ? 'Correct it if it is wrong' : 'Confirm the real value', style: const TextStyle(fontWeight: FontWeight.w700)),
        const SizedBox(height: 4),
        Text('Look at the vehicle and type exactly what is printed. Your entry is stored as confirmed by you.', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
        const SizedBox(height: 10),
        TextField(
          key: const ValueKey('confirm-field'),
          controller: ctl,
          textCapitalization: TextCapitalization.characters,
          autocorrect: false,
          enableSuggestions: false,
          inputFormatters: [FilteringTextInputFormatter.allow(RegExp(r'[A-Za-z0-9]')), LengthLimitingTextInputFormatter(17), _UpperCase()],
          style: const TextStyle(fontFamily: 'monospace', fontSize: 20, letterSpacing: 2, fontWeight: FontWeight.w700),
          decoration: InputDecoration(hintText: f.isPlate ? 'MH12AB1234' : '17-character VIN', errorText: error),
          onChanged: (_) => error == null ? null : setState(() => error = null),
          onSubmitted: (_) => _submit(),
        ),
        const SizedBox(height: 14),
        FilledButton.icon(key: const ValueKey('confirm-button'), onPressed: _submit, icon: const Icon(Icons.check), label: const Text('Confirm this value')),
        const SizedBox(height: 8),
        TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
      ]),
    );
  }
}

class _UpperCase extends TextInputFormatter {
  @override
  TextEditingValue formatEditUpdate(TextEditingValue o, TextEditingValue n) => n.copyWith(text: n.text.toUpperCase());
}

// ==================================================================== authenticity sheet
Future<void> showAuthenticitySheet(BuildContext context, Authenticity a, {ScreenContext? screenContext}) {
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    builder: (ctx) => DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.8,
      minChildSize: 0.4,
      maxChildSize: 0.95,
      builder: (ctx, scroll) => _AuthSheet(a: a, scroll: scroll, screenContext: screenContext),
    ),
  );
}

class _AuthSheet extends StatelessWidget {
  const _AuthSheet({required this.a, required this.scroll, this.screenContext});
  final Authenticity a;
  final ScrollController scroll;
  final ScreenContext? screenContext;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final reasons = humanReasons(a.reasons);
    final tips = improvements(a.notRun);
    Widget section(String t, Widget child) => Padding(padding: const EdgeInsets.only(top: 18), child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(t, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 15)), const SizedBox(height: 8), child]));
    return ListView(controller: scroll, padding: const EdgeInsets.fromLTRB(20, 0, 20, 28), children: [
      Text('Is this a real car?', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
      const SizedBox(height: 10),
      Wrap(spacing: 8, runSpacing: 8, children: [StatusChip(a.liveness), StatusChip(a.status)]),
      const SizedBox(height: 12),
      Text(
        switch (a.liveness) {
          'confirmed' => 'There is physical evidence that a real, three-dimensional vehicle was photographed.',
          'suspected_spoof' => 'The photos look like a picture of a screen or a print, not a real vehicle. Treat this capture as untrusted.',
          _ => 'Nothing looked wrong, but there is also no positive evidence that a real car was photographed. A good replay of a photo with no flash picture raises no flag at all.',
        },
        style: TextStyle(color: cs.onSurfaceVariant, height: 1.4),
      ),
      if (screenContext != null && screenContext!.score > 0 && screenContext!.reasons.isNotEmpty)
        section('What the photo shows', BulletList(screenContext!.reasons, icon: Icons.desktop_windows_outlined, color: toneFor(context, 'suspected_spoof').fg, size: 14)),
      if (a.positive.isNotEmpty) section('Evidence for a live scene', BulletList([for (final p in a.positive) humanPositive(p)], icon: Icons.check_circle, color: toneFor(context, 'accept').fg, size: 14)),
      if (reasons.isNotEmpty) section('Concerns and notes', BulletList(reasons, icon: Icons.warning_amber_rounded, color: toneFor(context, 'review').fg, size: 14)),
      if (a.notRun.isNotEmpty)
        section(
          'Not checked',
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            for (final k in a.notRun)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Icon(Icons.remove_circle_outline, size: 16, color: cs.onSurfaceVariant),
                  const SizedBox(width: 8),
                  Expanded(child: Text(humanCheck(k), style: const TextStyle(fontSize: 14))),
                ]),
              ),
          ]),
        ),
      if (tips.isNotEmpty) section('To get a stronger result', BulletList(tips, icon: Icons.tips_and_updates_outlined, color: Colors.amber.shade800, size: 14)),
      const SizedBox(height: 18),
      Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: cs.surfaceContainerHighest, borderRadius: BorderRadius.circular(12)),
        child: Text('These checks have been tested on simulated attacks only, not on real devices, and their thresholds are provisional. "No red flags" is never proof that a photo is genuine.',
            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12.5, height: 1.4)),
      ),
      const SizedBox(height: 14),
      FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Got it')),
    ]);
  }
}

// ========================================================================== photo viewer
/// Full-screen photo with the region the text was read from highlighted. [size] is the photo's
/// size as the reader saw it (the box coordinates are in that frame).
Future<void> showPhotoViewer(BuildContext context, {required Uint8List bytes, required String title, required Size size, List<(String, List<int>)> boxes = const []}) {
  return showDialog<void>(
    context: context,
    builder: (ctx) => Dialog.fullscreen(
      backgroundColor: Colors.black,
      child: Stack(children: [
        Center(
          child: InteractiveViewer(
            maxScale: 6,
            child: AspectRatio(
              aspectRatio: size.width > 0 ? size.width / size.height : 1,
              child: LayoutBuilder(builder: (ctx, c) {
                return Stack(fit: StackFit.expand, children: [
                  Image.memory(bytes, fit: BoxFit.fill),
                  for (final (label, b) in boxes)
                    if (size.width > 0)
                      Positioned(
                        left: c.maxWidth * b[0] / size.width,
                        top: c.maxHeight * b[1] / size.height,
                        width: c.maxWidth * (b[2] - b[0]) / size.width,
                        height: c.maxHeight * (b[3] - b[1]) / size.height,
                        child: Container(
                          decoration: BoxDecoration(border: Border.all(color: brandTeal, width: 2.5), borderRadius: BorderRadius.circular(3)),
                          alignment: Alignment.topLeft,
                          child: Transform.translate(offset: const Offset(0, -18), child: Text(label, style: const TextStyle(color: brandTeal, fontSize: 12, fontWeight: FontWeight.w700, backgroundColor: Colors.black54))),
                        ),
                      ),
                ]);
              }),
            ),
          ),
        ),
        Positioned(
          top: 8,
          left: 8,
          right: 8,
          child: SafeArea(
            child: Row(children: [
              IconButton(onPressed: () => Navigator.pop(ctx), icon: const Icon(Icons.close, color: Colors.white)),
              Expanded(child: Text(title, style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w600))),
            ]),
          ),
        ),
      ]),
    ),
  );
}
