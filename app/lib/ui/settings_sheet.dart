import 'package:flutter/material.dart';

import '../app_services.dart';
import 'onboarding.dart';
import 'plate_types.dart';
import 'widgets.dart';

Future<void> showSettingsSheet(BuildContext context) {
  return showModalBottomSheet<void>(
    context: context,
    isScrollControlled: true,
    useSafeArea: true,
    builder: (ctx) => Padding(padding: EdgeInsets.only(bottom: MediaQuery.viewInsetsOf(ctx).bottom), child: const _SettingsSheet()),
  );
}

class _SettingsSheet extends StatefulWidget {
  const _SettingsSheet();
  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  bool testing = false;
  String? testMessage;
  bool? testOk;

  /// Re-loads / re-checks the on-device models and says what loaded.
  Future<void> _selfTest() async {
    setState(() {
      testing = true;
      testMessage = null;
    });
    final st = await AppScope.of(context).scanner.prepare();
    testOk = st.ready;
    testMessage = st.ready
        ? 'Ready. ${st.detail ?? 'Reader loaded'}. Screen / print detector: ${st.recaptureModel ? 'on' : 'off'}.${st.loadMs != null ? ' Loaded in ${(st.loadMs! / 1000).toStringAsFixed(1)} s.' : ''}'
        : (st.error ?? 'The engine did not start.');
    if (mounted) setState(() => testing = false);
  }

  @override
  Widget build(BuildContext context) {
    final services = AppScope.of(context);
    final s = services.settings;
    final cs = Theme.of(context).colorScheme;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) => SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('Settings', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 16),
          Text('Reading engine', style: Theme.of(context).textTheme.titleSmall?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 6),
          Row(children: [
            Icon(Icons.offline_bolt_outlined, size: 18, color: cs.primary),
            const SizedBox(width: 8),
            Expanded(child: Text('Runs entirely on this phone. The reader never uses the network and no photo leaves the device.', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13, height: 1.35))),
          ]),
          const SizedBox(height: 10),
          Row(children: [
            OutlinedButton.icon(
              key: const ValueKey('self-test'),
              onPressed: testing ? null : _selfTest,
              icon: testing ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) : const Icon(Icons.memory),
              label: const Text('Check the engine'),
            ),
          ]),
          if (testMessage != null)
            Padding(
              padding: const EdgeInsets.only(top: 10),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(testOk == true ? Icons.check_circle : Icons.error, color: testOk == true ? Colors.green.shade700 : cs.error, size: 18),
                const SizedBox(width: 8),
                Expanded(child: Text(testMessage!, key: const ValueKey('test-result'), style: const TextStyle(fontSize: 13.5))),
              ]),
            ),
          const Divider(height: 32),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Plates are HSRP'),
            subtitle: const Text('Reflective high-security plates. Turns on the flash test; leave off for old painted plates.'),
            value: s.hsrp,
            onChanged: (v) => s.update(hsrp: v),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Require proof of a real car'),
            subtitle: const Text('Flag every capture for review unless the flash test confirms a real, reflective plate. (The side-step photos alone never confirm.)'),
            value: s.requireLiveness,
            onChanged: (v) => s.update(requireLiveness: v),
          ),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Keep scans in History'),
            subtitle: const Text('Stores the photos and results on this phone (plates are personal data).'),
            value: s.saveHistory,
            onChanged: (v) => s.update(saveHistory: v),
          ),
          const Divider(height: 24),
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.delete_outline),
            title: const Text('Clear history'),
            onTap: () async {
              final ok = await showConfirmDialog(context,
                  title: 'Clear all saved scans?', message: 'Every saved result and photo on this phone will be deleted. This cannot be undone.', confirm: 'Delete all', destructive: true, icon: Icons.delete_forever);
              if (ok) {
                await services.history.clear();
                if (context.mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('History cleared')));
              }
            },
          ),
          ListTile(
            key: const ValueKey('plate-types-guide'),
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.palette_outlined),
            title: const Text('Plate types guide'),
            subtitle: const Text('Private, commercial, electric, rental, diplomatic... and their plate colours.'),
            onTap: () => showPlateTypesSheet(context),
          ),
          ListTile(
            contentPadding: EdgeInsets.zero,
            leading: const Icon(Icons.help_outline),
            title: const Text('How it works'),
            onTap: () => showOnboarding(context),
          ),
          const SizedBox(height: 12),
          FilledButton(key: const ValueKey('save-settings'), onPressed: () => Navigator.pop(context), child: const Text('Done')),
        ]),
      ),
    );
  }
}
