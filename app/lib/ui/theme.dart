import 'package:flutter/material.dart';

const brandNavy = Color(0xFF1C3F94);
const brandTeal = Color(0xFF2AB6B6);

ThemeData buildTheme(Brightness b) {
  final scheme = ColorScheme.fromSeed(seedColor: brandNavy, brightness: b, secondary: brandTeal);
  final shape = RoundedRectangleBorder(borderRadius: BorderRadius.circular(14));
  return ThemeData(
    useMaterial3: true,
    colorScheme: scheme,
    visualDensity: VisualDensity.standard,
    appBarTheme: AppBarTheme(centerTitle: false, backgroundColor: scheme.surface, scrolledUnderElevation: 1),
    cardTheme: CardThemeData(elevation: 0, color: scheme.surfaceContainerLow, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18), side: BorderSide(color: scheme.outlineVariant))),
    filledButtonTheme: FilledButtonThemeData(style: FilledButton.styleFrom(shape: shape, minimumSize: const Size(0, 48))),
    outlinedButtonTheme: OutlinedButtonThemeData(style: OutlinedButton.styleFrom(shape: shape, minimumSize: const Size(0, 48))),
    textButtonTheme: TextButtonThemeData(style: TextButton.styleFrom(shape: shape)),
    inputDecorationTheme: InputDecorationTheme(border: OutlineInputBorder(borderRadius: BorderRadius.circular(12))),
    dialogTheme: DialogThemeData(shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(22))),
    bottomSheetTheme: BottomSheetThemeData(
      showDragHandle: true,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(24))),
      backgroundColor: scheme.surface,
    ),
    snackBarTheme: SnackBarThemeData(behavior: SnackBarBehavior.floating, shape: shape),
  );
}

/// How a verdict looks. `status` is the backend's: accept | review | reject | no_flags | none.
class Tone {
  const Tone(this.label, this.icon, this.bg, this.fg);
  final String label;
  final IconData icon;
  final Color bg, fg;
}

Tone toneFor(BuildContext context, String key) {
  final dark = Theme.of(context).brightness == Brightness.dark;
  Color bg(Color l, Color d) => dark ? d : l;
  switch (key) {
    case 'accept':
      return Tone('Read with confidence', Icons.check_circle, bg(const Color(0xFFE3F5EC), const Color(0xFF123D2A)), bg(const Color(0xFF1E8E5A), const Color(0xFF7FE0B0)));
    case 'review':
      return Tone('Needs review', Icons.rule, bg(const Color(0xFFFFF1DB), const Color(0xFF4A3410)), bg(const Color(0xFF9A5B00), const Color(0xFFFFCB7A)));
    case 'reject':
      return Tone('Unreadable', Icons.block, bg(const Color(0xFFFDECEA), const Color(0xFF4A1612)), bg(const Color(0xFFB3261E), const Color(0xFFFFADA6)));
    case 'confirmed':
      return Tone('Liveness confirmed', Icons.verified_user, bg(const Color(0xFFE3F5EC), const Color(0xFF123D2A)), bg(const Color(0xFF1E8E5A), const Color(0xFF7FE0B0)));
    case 'suspected_spoof':
      return Tone('Suspected screen / print', Icons.gpp_bad, bg(const Color(0xFFFDECEA), const Color(0xFF4A1612)), bg(const Color(0xFFB3261E), const Color(0xFFFFADA6)));
    case 'not_established':
      return Tone('Liveness not established', Icons.help_outline, bg(const Color(0xFFFFF1DB), const Color(0xFF4A3410)), bg(const Color(0xFF9A5B00), const Color(0xFFFFCB7A)));
    case 'no_flags':
      return Tone('No red flags', Icons.shield_outlined, bg(const Color(0xFFE8EEF8), const Color(0xFF1B2A4A)), bg(const Color(0xFF1C3F94), const Color(0xFFA9C1F5)));
    default:
      return Tone(key, Icons.info_outline, bg(const Color(0xFFEEF2F6), const Color(0xFF2A2F36)), bg(const Color(0xFF455A64), const Color(0xFFB0BEC5)));
  }
}
