
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_services.dart';
import '../data/history.dart';
import '../data/models.dart';
import 'plate_types.dart';
import 'sheets.dart';
import 'theme.dart';
import 'widgets.dart';

/// The outcome of one scan. A field that is not "read with confidence" has to be confirmed by the
/// person (type what is on the vehicle) before the scan is finished - the app never silently turns
/// a guess into a record.
class ResultPage extends StatefulWidget {
  const ResultPage({super.key, required this.result, required this.photos, this.entry});
  final ScanResult result;

  /// Photo bytes by photo index. Missing entries are simply not shown.
  final Map<int, Uint8List> photos;

  /// Set when reopened from history: nothing is saved again.
  final HistoryEntry? entry;

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  String? plate, vin; // the values the person has confirmed (accepted reads are pre-confirmed)
  String? categoryOverride; // a vehicle type the person picked (or confirmed) instead of the reader's guess
  bool saving = false;

  ScanResult get r => widget.result;

  /// The vehicle type in force: the person's choice, else what the reader worked out (null = not a plate / old scan).
  String? get categoryKey => categoryOverride ?? r.plate?.category?.key;

  @override
  void initState() {
    super.initState();
    plate = widget.entry?.confirmedPlate ?? (r.plate?.isAccepted == true ? r.plate!.value : null);
    vin = widget.entry?.confirmedVin ?? (r.vin?.isAccepted == true ? r.vin!.value : null);
    categoryOverride = widget.entry?.confirmedCategory;
  }

  int get pending => [if (r.plate != null && plate == null) 1, if (r.vin != null && vin == null) 1].length;

  Future<void> _pickCategory() async {
    final c = r.plate?.category;
    final k = await showPlateTypesSheet(context, detected: c?.key, selected: categoryKey, alternatives: c?.alternatives ?? const [], pick: true);
    if (k == null || !mounted || k == categoryOverride) return;
    setState(() => categoryOverride = k);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Vehicle type: ${plateTypeOf(k).label}')));
  }

  Future<void> _open(FieldResult f) async {
    final v = await showFieldSheet(context, f, confirmed: f.isPlate ? plate : vin);
    if (v == null || !mounted) return;
    setState(() => f.isPlate ? plate = v : vin = v);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('${f.isPlate ? 'Plate' : 'VIN'} confirmed: $v')));
  }

  Future<void> _done() async {
    if (pending > 0) {
      final ok = await showConfirmDialog(context,
          title: 'Finish with $pending unconfirmed?',
          message: 'The ${pending == 1 ? 'reading that still needs' : 'readings that still need'} review will be saved as "not confirmed". You can reopen it from History later.',
          confirm: 'Finish anyway',
          cancel: 'Keep reviewing',
          icon: Icons.rule);
      if (!ok || !mounted) return;
    }
    if (widget.entry == null) {
      final services = AppScope.of(context);
      if (services.settings.saveHistory) {
        setState(() => saving = true);
        try {
          await services.history.add(r, [for (var i = 0; i < r.photos.length; i++) widget.photos[i] ?? Uint8List(0)].where((b) => b.isNotEmpty).toList(),
              confirmedPlate: plate, confirmedVin: vin, confirmedCategory: categoryOverride);
        } catch (_) {
          if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text("Couldn't save to history.")));
        }
        if (!mounted) return;
        setState(() => saving = false);
      }
    }
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  Future<void> _back() async {
    if (widget.entry != null || (pending == 0 && !r.foundAnything)) {
      Navigator.of(context).pop();
      return;
    }
    final ok = await showConfirmDialog(context,
        title: 'Leave without saving?', message: 'This scan has not been saved. Leaving now discards it.', confirm: 'Discard', cancel: 'Stay', destructive: true, icon: Icons.delete_outline);
    if (ok && mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final asked = r.photos.map((p) => p.role).toSet();
    final askedPlate = asked.any((x) => x == 'plate' || x == 'auto' || x == 'angle');
    final askedVin = asked.any((x) => x == 'vin' || x == 'auto');
    final fromHistory = widget.entry != null;
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) _back();
      },
      child: Scaffold(
        appBar: AppBar(
          leading: IconButton(icon: const Icon(Icons.arrow_back), tooltip: 'Back', onPressed: _back),
          title: Text(fromHistory ? 'Saved scan' : 'Result'),
          actions: [
            if (r.foundAnything)
              IconButton(
                tooltip: 'Copy',
                icon: const Icon(Icons.copy_all_outlined),
                onPressed: () {
                  final lines = [
                    if (plate != null) 'Plate: $plate',
                    if (plate != null && categoryKey != null && categoryKey != 'unknown') 'Type: ${plateTypeOf(categoryKey).label}',
                    if (vin != null) 'VIN: $vin',
                  ];
                  if (lines.isEmpty) return;
                  Clipboard.setData(ClipboardData(text: lines.join('\n')));
                  ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Copied to the clipboard')));
                },
              ),
          ],
        ),
        body: ListView(padding: const EdgeInsets.fromLTRB(16, 4, 16, 24), children: [
          _Summary(r: r, pending: pending),
          const SizedBox(height: 12),
          if (r.plate != null)
            _FieldCard(
              f: r.plate!,
              confirmed: plate,
              onTap: () => _open(r.plate!),
              categoryKey: categoryKey,
              categoryChosen: categoryOverride != null,
              onPickCategory: fromHistory ? null : _pickCategory,
              colourTrusted: !r.auth.isSpoofSuspected,
            ),
          if (r.plate == null && askedPlate) const _MissingCard(title: 'Number plate', text: 'No plate was found in these photos. Retake with the plate square-on and filling about a third of the frame.'),
          const SizedBox(height: 12),
          if (r.vin != null) _FieldCard(f: r.vin!, confirmed: vin, onTap: () => _open(r.vin!)),
          if (r.vin == null && askedVin && !(askedPlate && !asked.contains('vin') && !asked.contains('auto')))
            const _MissingCard(title: 'VIN', text: 'No VIN was found. Get closer, keep the whole 17-character line in frame and tilt the phone so reflections leave the characters.'),
          if (r.conflicts.isNotEmpty) ...[
            const SizedBox(height: 12),
            Card(
              color: toneFor(context, 'review').bg,
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Row(children: [Icon(Icons.compare_arrows, color: toneFor(context, 'review').fg), const SizedBox(width: 8), Text('Photos disagree', style: TextStyle(fontWeight: FontWeight.w700, color: toneFor(context, 'review').fg))]),
                  const SizedBox(height: 6),
                  for (final c in r.conflicts) Text(c, style: TextStyle(color: toneFor(context, 'review').fg, fontSize: 13)),
                ]),
              ),
            ),
          ],
          const SizedBox(height: 12),
          _AuthCard(a: r.auth, onTap: () => showAuthenticitySheet(context, r.auth, screenContext: r.screenContext)),
          if (widget.photos.isNotEmpty) ...[
            const SizedBox(height: 18),
            Text('Photos', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            SizedBox(
              height: 112,
              child: ListView(scrollDirection: Axis.horizontal, children: [
                for (final p in r.photos)
                  if (widget.photos[p.index] != null)
                    Padding(
                      padding: const EdgeInsets.only(right: 10),
                      child: InkWell(
                        borderRadius: BorderRadius.circular(12),
                        onTap: () => showPhotoViewer(context,
                            bytes: widget.photos[p.index]!,
                            title: 'Photo ${p.index + 1} · ${_photoLabel(p)}',
                            size: Size(p.width.toDouble(), p.height.toDouble()),
                            boxes: [
                              if (r.plate != null && r.plate!.image == p.index && r.plate!.box != null) ('plate', r.plate!.box!),
                              if (r.vin != null && r.vin!.image == p.index && r.vin!.box != null) ('VIN', r.vin!.box!),
                            ]),
                        child: Column(children: [
                          Container(
                            width: 112,
                            height: 86,
                            decoration: BoxDecoration(borderRadius: BorderRadius.circular(12), border: Border.all(color: cs.outlineVariant)),
                            clipBehavior: Clip.antiAlias,
                            child: Image.memory(widget.photos[p.index]!, fit: BoxFit.cover, cacheWidth: 300),
                          ),
                          const SizedBox(height: 4),
                          Text(_photoLabel(p), style: const TextStyle(fontSize: 12)),
                        ]),
                      ),
                    ),
              ]),
            ),
          ],
        ]),
        bottomNavigationBar: SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
            child: fromHistory
                ? FilledButton(onPressed: () => Navigator.of(context).pop(), child: const Text('Close'))
                : Row(children: [
                    Expanded(child: OutlinedButton(onPressed: saving ? null : _back, child: const Text('Discard'))),
                    const SizedBox(width: 12),
                    Expanded(
                      flex: 2,
                      child: FilledButton.icon(
                        key: const ValueKey('done'),
                        onPressed: saving ? null : _done,
                        icon: saving ? const SizedBox(width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.check),
                        label: Text(pending > 0 ? 'Done ($pending to review)' : 'Done'),
                      ),
                    ),
                  ]),
          ),
        ),
      ),
    );
  }
}

/// A person-readable name for a photo: the capture step it came from (front / left / right ...).
String _photoLabel(PhotoInfo p) => switch (p.name.replaceAll('.jpg', '')) {
      'front' => 'Front',
      'front_flash' => 'Front + torch',
      'left' => 'Left',
      'right' => 'Right',
      'vin' => 'VIN',
      _ => switch (p.role) { 'plate' => 'Plate', 'plate_flash' => 'Torch', 'angle' => 'Side', 'vin' => 'VIN', _ => 'Photo' },
    };

// =================================================================================== cards
class _Summary extends StatelessWidget {
  const _Summary({required this.r, required this.pending});
  final ScanResult r;
  final int pending;

  @override
  Widget build(BuildContext context) {
    final String key;
    final String title;
    final String sub;
    if (!r.foundAnything) {
      key = 'reject';
      title = 'Nothing could be read';
      sub = 'No plate or VIN was found. Retake the photos in better light, closer, and square-on.';
    } else if (r.auth.isSpoofSuspected) {
      key = 'suspected_spoof';
      title = 'Do not trust this capture';
      sub = 'The photos look like a picture of a screen or a print. Review the details below.';
    } else if (pending > 0) {
      key = 'review';
      title = '$pending reading${pending == 1 ? '' : 's'} need${pending == 1 ? 's' : ''} your check';
      sub = 'Tap a card, compare it with the vehicle, and confirm the real value.';
    } else {
      key = 'accept';
      title = 'Read successfully';
      sub = r.auth.isConfirmed ? 'A real vehicle was photographed (liveness confirmed).' : 'Check it against the vehicle before relying on it.';
    }
    final t = toneFor(context, key);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: t.bg, borderRadius: BorderRadius.circular(18)),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Icon(t.icon, color: t.fg, size: 30),
        const SizedBox(width: 12),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(title, style: TextStyle(color: t.fg, fontWeight: FontWeight.w800, fontSize: 18)),
            const SizedBox(height: 4),
            Text(sub, style: TextStyle(color: t.fg, height: 1.35)),
          ]),
        ),
      ]),
    );
  }
}

class _FieldCard extends StatelessWidget {
  const _FieldCard(
      {required this.f, required this.confirmed, required this.onTap, this.categoryKey, this.categoryChosen = false, this.onPickCategory, this.colourTrusted = true});
  final FieldResult f;
  final String? confirmed;
  final VoidCallback onTap;

  /// False when the photos look like a picture of a screen or print: those change every colour, so the vehicle type
  /// (which is read from the plate colours) is only ever offered as a guess.
  final bool colourTrusted;

  /// The vehicle type to show on a plate card (null = none, e.g. a VIN or a scan from before this feature).
  final String? categoryKey;
  final bool categoryChosen; // the person picked it, so it is not a guess any more
  final VoidCallback? onPickCategory;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final isConfirmed = confirmed != null;
    final userEdited = isConfirmed && confirmed != f.value;
    final shownText = userEdited ? confirmed! : f.shown; // an accepted read keeps its formatted display
    final cat = f.isPlate ? f.category : null;
    // A type the reader is not sure of (low confidence, undetermined, or colours from a screen photo) is only a guess.
    final unsure = cat != null && !categoryChosen && (cat.isUnknown || cat.confidence < PlateTypeChip.unsure || !colourTrusted);
    // The number is drawn on a board in the plate's own colours - but never for a guess (a wrong colour board is worse than
    // none: a real green EV plate was once shown on a blue board) and only when it has no unknown characters, since the red
    // "?" and amber suggestion boxes are drawn for a white / neutral background.
    final boardKey = (f.isPlate && categoryKey != null && !unsure && !shownText.contains('?')) ? categoryKey : null;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        key: ValueKey('field-${f.domain}'),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Icon(f.isPlate ? Icons.directions_car : Icons.pin, size: 18, color: cs.onSurfaceVariant),
              const SizedBox(width: 8),
              Flexible(child: Text(f.isPlate ? 'Number plate' : 'VIN / chassis', maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(color: cs.onSurfaceVariant, fontWeight: FontWeight.w600))),
              const Spacer(),
              if (userEdited) const StatusChip('accept', label: 'Confirmed by you', dense: true) else StatusChip(f.status, dense: true),
            ]),
            const SizedBox(height: 12),
            PlateBoard(
              typeKey: boardKey,
              builder: (ink) => MaskedText(shownText, size: f.isPlate ? 32 : 21, color: boardKey == null ? null : ink, hints: isConfirmed ? const {} : f.inlineHints),
            ),
            if (!isConfirmed) SlotHints(f),
            if (cat != null) ...[
              const SizedBox(height: 10),
              Align(
                alignment: Alignment.centerLeft,
                child: PlateTypeChip(typeKey: categoryKey ?? cat.key, confidence: cat.confidence, userChosen: categoryChosen, hedge: !colourTrusted, onTap: onPickCategory),
              ),
              if (unsure && onPickCategory != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    colourTrusted
                        ? 'Not sure of the type - tap it and choose what the plate looks like.'
                        : 'These colours come from a photo that looks like a screen or print, so the type is only a guess - tap it and choose what the plate really looks like.',
                    key: const ValueKey('type-unsure-note'),
                    style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12.5),
                  ),
                ),
            ],
            const SizedBox(height: 10),
            Wrap(spacing: 6, runSpacing: 6, children: [
              if (cat != null && cat.bh) const InfoPill('Bharat (BH) series', icon: Icons.public),
              InfoPill('${(f.confidence * 100).round()}% confidence', icon: Icons.speed),
              if (f.stateName != null) InfoPill(f.stateName!, icon: Icons.place_outlined),
              if (f.maker != null) InfoPill(f.maker!, icon: Icons.factory_outlined),
              if (!f.isPlate && f.checkDigitOk == true) const InfoPill('check digit valid', icon: Icons.verified_outlined),
              if (!f.isPlate && f.checkDigitOk == false) const InfoPill('no valid check digit', icon: Icons.info_outline),
              if (f.unknownCount > 0) InfoPill('${f.unknownCount} unknown', icon: Icons.help_outline),
            ]),
            const SizedBox(height: 12),
            Row(children: [
              Expanded(
                child: Text(
                  isConfirmed ? (userEdited ? 'You entered this value.' : 'Looks right. Tap to edit.') : 'Tap to compare with the vehicle and confirm.',
                  style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13),
                ),
              ),
              Icon(Icons.chevron_right, color: cs.onSurfaceVariant),
            ]),
          ]),
        ),
      ),
    );
  }
}

/// What the recogniser thinks an unknown / weak character could be, restricted to what the plate or VIN
/// format allows in that slot. Shown under the masked text so a `?` is never a dead end.
class SlotHints extends StatelessWidget {
  const SlotHints(this.f, {super.key});
  final FieldResult f;

  static String pct(double p) => '${(p * 100).round()}%';

  static String describe(UnknownSlot s) {
    final top = s.options.first;
    final others = s.options.skip(1).map((o) => '${o.char} ${pct(o.p)}').join(', ');
    final head = 'Character ${s.position + 1}: ';
    if (s.suggested != null) return '${head}probably ${s.suggested} (${pct(top.p)})${others.isEmpty ? '' : ', or $others'}';
    return '${head}could be ${s.options.map((o) => '${o.char} ${pct(o.p)}').join(', ')}';
  }

  @override
  Widget build(BuildContext context) {
    final slots = f.unknownSlots.where((s) => s.options.isNotEmpty).take(3).toList();
    if (slots.isEmpty) return const SizedBox.shrink();
    final cs = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.only(top: 8),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        for (final s in slots)
          Padding(
            padding: const EdgeInsets.only(bottom: 4),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Padding(padding: const EdgeInsets.only(top: 2), child: Icon(Icons.lightbulb_outline, size: 15, color: Colors.amber.shade800)),
              const SizedBox(width: 6),
              Expanded(child: Text(describe(s), key: ValueKey('slot-hint-${s.position}'), style: TextStyle(fontSize: 13, color: cs.onSurfaceVariant, height: 1.3))),
            ]),
          ),
      ]),
    );
  }
}

class _MissingCard extends StatelessWidget {
  const _MissingCard({required this.title, required this.text});
  final String title, text;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Icon(Icons.search_off, color: cs.onSurfaceVariant),
          const SizedBox(width: 12),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text('$title not found', style: const TextStyle(fontWeight: FontWeight.w700)), const SizedBox(height: 4), Text(text, style: TextStyle(color: cs.onSurfaceVariant, height: 1.35))])),
        ]),
      ),
    );
  }
}

class _AuthCard extends StatelessWidget {
  const _AuthCard({required this.a, required this.onTap});
  final Authenticity a;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        key: const ValueKey('auth-card'),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Icon(Icons.security, size: 18, color: cs.onSurfaceVariant),
              const SizedBox(width: 8),
              Expanded(child: Text('Is this a real car?', maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(color: cs.onSurfaceVariant, fontWeight: FontWeight.w600))),
              Icon(Icons.chevron_right, color: cs.onSurfaceVariant),
            ]),
            const SizedBox(height: 10),
            Wrap(spacing: 8, runSpacing: 8, children: [StatusChip(a.liveness), StatusChip(a.status)]),
            const SizedBox(height: 8),
            Text('Tap to see what was checked and what was not.', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
          ]),
        ),
      ),
    );
  }
}
