import 'package:flutter/material.dart';

import '../data/models.dart';

/// How each vehicle category looks on the road and what it means. THIS TABLE IS DATA: it mirrors the
/// category rules in ml/vinplate/category.py (the reader decides the category from the plate's colours
/// and text pattern; the app only displays it). The rules were taken from public plate-colour guides,
/// not from the official notification, so correct them here and in category.py if a rule is wrong.
class PlateType {
  const PlateType(this.key, this.label, this.colours, this.meaning, this.bg, this.ink);
  final String key, label, colours, meaning;
  final Color bg, ink;
}

const Color _yellow = Color(0xFFFFD21F);

const plateTypes = <PlateType>[
  PlateType('private', 'Private vehicle', 'Black text on white', 'Private / non-transport vehicle.', Color(0xFFFFFFFF), Color(0xFF111111)),
  PlateType('commercial', 'Commercial / transport vehicle', 'Black text on yellow', 'Taxis, trucks, buses and other transport vehicles.', _yellow, Color(0xFF111111)),
  PlateType('rental', 'Rental / self-drive vehicle', 'Yellow text on black', 'Rent-a-cab and self-drive rental vehicles.', Color(0xFF151515), _yellow),
  PlateType('ev_private', 'Electric vehicle (private)', 'White text on green', 'Privately owned electric vehicle.', Color(0xFF1E8E3E), Color(0xFFFFFFFF)),
  PlateType('ev_commercial', 'Electric vehicle (commercial)', 'Yellow text on green', 'Commercial / transport electric vehicle.', Color(0xFF1E8E3E), _yellow),
  PlateType('temporary', 'Temporary registration', 'Red text on yellow', 'Vehicle on a temporary registration.', _yellow, Color(0xFFC62828)),
  PlateType('trade', 'Trade certificate / dealer vehicle', 'White text on red', 'Vehicle in the possession of a dealer / trade certificate holder.', Color(0xFFD32F2F), Color(0xFFFFFFFF)),
  PlateType('diplomatic', 'Diplomatic / consular vehicle', 'White or yellow text on blue', 'Embassy and consular vehicles (codes CD, CC, UN).', Color(0xFF1565C0), Color(0xFFFFFFFF)),
  PlateType('military', 'Military / defence vehicle', 'White text on a dark plate', 'Defence vehicles, e.g. 03D 153874 W (printed after an up-arrow).', Color(0xFF2B2F2B), Color(0xFFFFFFFF)),
  PlateType('unknown', 'Type not determined', 'Colours unclear', 'The plate colours could not be read reliably. Choose the type by looking at the plate.', Color(0xFFECEFF1), Color(0xFF455A64)),
];

/// Plates that exist but are not something the reader decides from a photo.
const plateTypeNotes = <(String, String)>[
  ('President / Governors', 'Red plate with the national emblem and no number. Not read by this app.'),
  ('Bharat (BH) series', 'Black text on white, format 22 BH 1234 AA. A private plate that is valid across states; shown as a note on private vehicles.'),
];

PlateType plateTypeOf(String? key) => plateTypes.firstWhere((t) => t.key == key, orElse: () => plateTypes.last);

/// A small plate icon in the category's colours.
class PlateBadge extends StatelessWidget {
  const PlateBadge(this.typeKey, {super.key, this.width = 40});
  final String? typeKey;
  final double width;

  @override
  Widget build(BuildContext context) {
    final t = plateTypeOf(typeKey);
    final dark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      width: width,
      height: width * 0.5,
      alignment: Alignment.center,
      decoration: BoxDecoration(color: t.bg, borderRadius: BorderRadius.circular(width * 0.1), border: Border.all(color: dark ? Colors.white54 : Colors.black45, width: 1.2)),
      child: Text('AB', style: TextStyle(color: t.ink, fontSize: width * 0.28, fontWeight: FontWeight.w900, height: 1)),
    );
  }
}

/// The detected vehicle category as a chip: plate icon + name, with a "check" wording when unsure.
class PlateTypeChip extends StatelessWidget {
  const PlateTypeChip({super.key, required this.typeKey, this.confidence, this.userChosen = false, this.hedge = false, this.onTap});
  final String typeKey;
  final double? confidence;
  final bool userChosen;

  /// Say "Probably ..." whatever the confidence (e.g. the colours came from a photo of a screen).
  final bool hedge;
  final VoidCallback? onTap;

  /// Below this confidence a type is a guess: it is worded "Probably ..." and never drawn as the plate's colours.
  static const unsure = 0.6;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final t = plateTypeOf(typeKey);
    final isUnknown = typeKey == 'unknown';
    final low = !userChosen && !isUnknown && (hedge || (confidence ?? 1) < unsure);
    final label = isUnknown ? t.label : (low ? 'Probably ${t.label.toLowerCase()}' : t.label);
    return Material(
      color: cs.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(20),
      child: InkWell(
        key: const ValueKey('plate-type-chip'),
        borderRadius: BorderRadius.circular(20),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(8, 5, 12, 5),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            PlateBadge(typeKey, width: 34),
            const SizedBox(width: 8),
            Flexible(child: Text(label, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700))),
            if (userChosen) ...[const SizedBox(width: 6), Icon(Icons.person, size: 14, color: cs.primary)],
            if (low || isUnknown) ...[const SizedBox(width: 6), Icon(Icons.help_outline, size: 14, color: cs.onSurfaceVariant)],
            if (onTap != null) ...[const SizedBox(width: 4), Icon(Icons.expand_more, size: 16, color: cs.onSurfaceVariant)],
          ]),
        ),
      ),
    );
  }
}

/// Draws the plate number on a plate-coloured board, so the card looks like the plate it describes.
class PlateBoard extends StatelessWidget {
  const PlateBoard({super.key, required this.typeKey, required this.builder});
  final String? typeKey;
  final Widget Function(Color ink) builder;

  @override
  Widget build(BuildContext context) {
    if (typeKey == null || typeKey == 'unknown') return builder(Theme.of(context).colorScheme.onSurface);
    final t = plateTypeOf(typeKey);
    final dark = Theme.of(context).brightness == Brightness.dark;
    return Container(
      key: const ValueKey('plate-board'),
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
      decoration: BoxDecoration(color: t.bg, borderRadius: BorderRadius.circular(10), border: Border.all(color: dark ? Colors.white60 : Colors.black87, width: 2)),
      child: builder(t.ink),
    );
  }
}

/// Lists every plate type. With [pick] the person can choose one (returned); otherwise it is a guide.
Future<String?> showPlateTypesSheet(BuildContext context, {String? detected, String? selected, List<PlateAlt> alternatives = const [], bool pick = false}) {
  return showModalBottomSheet<String>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    builder: (ctx) => DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.85,
      minChildSize: 0.4,
      maxChildSize: 0.95,
      builder: (ctx, scroll) => _PlateTypesSheet(scroll: scroll, detected: detected, selected: selected, alternatives: alternatives, pick: pick),
    ),
  );
}

class _PlateTypesSheet extends StatelessWidget {
  const _PlateTypesSheet({required this.scroll, required this.detected, required this.selected, required this.alternatives, required this.pick});
  final ScrollController scroll;
  final String? detected, selected;
  final List<PlateAlt> alternatives;
  final bool pick;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final altKeys = alternatives.map((a) => a.key).toSet();
    return ListView(controller: scroll, padding: const EdgeInsets.fromLTRB(16, 0, 16, 24), children: [
      Text(pick ? 'What type of vehicle is it?' : 'Plate types', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
      const SizedBox(height: 4),
      Text(
        pick
            ? 'The app guesses the type from the plate colours and format. Look at the plate and choose the right one if it is wrong - colours change under different light.'
            : 'The reader tells the vehicle type from the plate colours and the format of the number.',
        style: TextStyle(color: cs.onSurfaceVariant, height: 1.35),
      ),
      const SizedBox(height: 12),
      for (final t in plateTypes)
        if (!(t.key == 'unknown' && !pick))
          Card(
            margin: const EdgeInsets.only(bottom: 8),
            color: t.key == selected ? cs.primaryContainer : null,
            child: InkWell(
              key: ValueKey('type-${t.key}'),
              borderRadius: BorderRadius.circular(18),
              onTap: pick ? () => Navigator.pop(context, t.key) : null,
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Row(children: [
                  PlateBadge(t.key, width: 54),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Row(children: [
                        Flexible(child: Text(t.label, style: const TextStyle(fontWeight: FontWeight.w700))),
                        if (t.key == detected) ...[const SizedBox(width: 6), const _Tag('detected')],
                        if (t.key != detected && altKeys.contains(t.key)) ...[const SizedBox(width: 6), const _Tag('possible')],
                      ]),
                      const SizedBox(height: 2),
                      Text(t.colours, style: TextStyle(fontSize: 12.5, color: cs.onSurfaceVariant)),
                      Text(t.meaning, style: TextStyle(fontSize: 12.5, color: cs.onSurfaceVariant)),
                    ]),
                  ),
                  if (pick && t.key == selected) Icon(Icons.check_circle, color: cs.primary),
                ]),
              ),
            ),
          ),
      if (!pick) ...[
        const SizedBox(height: 8),
        for (final (title, text) in plateTypeNotes)
          Padding(
            padding: const EdgeInsets.only(bottom: 8),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Icon(Icons.info_outline, size: 16, color: cs.onSurfaceVariant),
              const SizedBox(width: 8),
              Expanded(child: Text('$title: $text', style: TextStyle(fontSize: 13, color: cs.onSurfaceVariant, height: 1.3))),
            ]),
          ),
      ],
      const SizedBox(height: 8),
      Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: cs.surfaceContainerHighest, borderRadius: BorderRadius.circular(12)),
        child: Text('These rules come from public plate-colour guides and have not been checked against the official notification. Photos taken in coloured light or strong glare can change how a plate colour looks; always check the type against the plate.',
            style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12.5, height: 1.4)),
      ),
      const SizedBox(height: 12),
      if (pick) TextButton(onPressed: () => Navigator.pop(context), child: const Text('Keep as is')) else FilledButton(onPressed: () => Navigator.pop(context), child: const Text('Got it')),
    ]);
  }
}

class _Tag extends StatelessWidget {
  const _Tag(this.text);
  final String text;
  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 1),
      decoration: BoxDecoration(color: cs.secondaryContainer, borderRadius: BorderRadius.circular(8)),
      child: Text(text, style: TextStyle(fontSize: 11, color: cs.onSecondaryContainer, fontWeight: FontWeight.w600)),
    );
  }
}
