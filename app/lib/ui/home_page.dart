import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../app_services.dart';
import '../data/history.dart';
import '../data/scanner.dart';
import 'crash_dialog.dart';
import 'capture_page.dart';
import 'capture_plan.dart';
import 'onboarding.dart';
import 'plate_types.dart';
import 'result_page.dart';
import 'settings_sheet.dart';
import 'theme.dart';
import 'widgets.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

enum _Engine { loading, ready, failed }

class _HomePageState extends State<HomePage> {
  _Engine engineState = _Engine.loading;
  EngineStatus? engine;
  List<HistoryEntry> history = const [];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) async {
      final s = AppScope.of(context).settings;
      _prepareEngine();
      _loadHistory();
      if (!s.onboarded) {
        await showOnboarding(context);
        await s.update(onboarded: true);
      }
      if (mounted) await showPendingCrashReport(context);
    });
  }

  /// Loads the models in the background (first call) so the first scan does not wait for them.
  Future<void> _prepareEngine() async {
    if (mounted) setState(() => engineState = _Engine.loading);
    final st = await AppScope.of(context).scanner.prepare();
    if (!mounted) return;
    setState(() {
      engine = st;
      engineState = st.ready ? _Engine.ready : _Engine.failed;
    });
  }

  Future<void> _loadHistory() async {
    try {
      final h = await AppScope.of(context).history.list();
      if (mounted) setState(() => history = h);
    } catch (_) {}
  }

  Future<void> _start(ScanMode mode) async {
    if (engineState == _Engine.failed) {
      final retry = await showConfirmDialog(context,
          title: 'The reading engine is not ready',
          message: engine?.error ?? 'The models that read plates could not be loaded.',
          confirm: 'Try again',
          cancel: 'Close',
          icon: Icons.error_outline);
      if (retry) _prepareEngine();
      return;
    }
    await Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => CapturePage(mode: mode)));
    _loadHistory();
  }

  Future<void> _openSettings() async {
    await showSettingsSheet(context);
    _loadHistory();
  }

  Future<void> _open(HistoryEntry e) async {
    final photos = <int, Uint8List>{};
    for (var i = 0; i < e.result.photos.length; i++) {
      final f = e.photoFile(i);
      if (f != null && f.existsSync()) photos[i] = f.readAsBytesSync();
    }
    await Navigator.of(context).push(MaterialPageRoute<void>(builder: (_) => ResultPage(result: e.result, photos: photos, entry: e)));
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: Row(children: [
          Container(
            width: 34,
            height: 34,
            decoration: BoxDecoration(borderRadius: BorderRadius.circular(10), gradient: const LinearGradient(colors: [brandNavy, brandTeal])),
            child: const Icon(Icons.document_scanner, color: Colors.white, size: 20),
          ),
          const SizedBox(width: 10),
          const Flexible(child: Text('VinPlate Scanner', maxLines: 1, overflow: TextOverflow.ellipsis, style: TextStyle(fontWeight: FontWeight.w800))),
        ]),
        actions: [IconButton(key: const ValueKey('open-settings'), tooltip: 'Settings', icon: const Icon(Icons.tune), onPressed: _openSettings)],
      ),
      body: RefreshIndicator(
        onRefresh: () async {
          await _prepareEngine();
          await _loadHistory();
        },
        child: ListView(padding: const EdgeInsets.fromLTRB(16, 4, 16, 28), children: [
          _EngineBanner(state: engineState, status: engine, onRetry: _prepareEngine),
          const SizedBox(height: 14),
          _BigAction(
            key: const ValueKey('scan-full'),
            icon: Icons.verified_user_outlined,
            title: 'Full vehicle check',
            subtitle: 'Front (with flash), left and right photos of the plate, then the VIN. Reads both and checks it is a real car.',
            primary: true,
            onTap: () => _start(ScanMode.full),
          ),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: _SmallAction(key: const ValueKey('scan-plate'), icon: Icons.directions_car_outlined, title: 'Number plate', subtitle: 'Front, left, right', onTap: () => _start(ScanMode.plate))),
            const SizedBox(width: 10),
            Expanded(child: _SmallAction(key: const ValueKey('scan-vin'), icon: Icons.pin_outlined, title: 'VIN', subtitle: 'One photo', onTap: () => _start(ScanMode.vin))),
          ]),
          const SizedBox(height: 26),
          Row(children: [
            Text('Recent scans', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
            const Spacer(),
            if (history.isNotEmpty) Text('${history.length} saved', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
          ]),
          const SizedBox(height: 8),
          if (history.isEmpty)
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(borderRadius: BorderRadius.circular(16), border: Border.all(color: cs.outlineVariant)),
              child: Row(children: [
                Icon(Icons.history, color: cs.onSurfaceVariant),
                const SizedBox(width: 12),
                Expanded(child: Text('Scans you finish appear here, with their photos, so you can check them again.', style: TextStyle(color: cs.onSurfaceVariant))),
              ]),
            )
          else
            for (final e in history)
              Dismissible(
                key: ValueKey(e.id),
                direction: DismissDirection.endToStart,
                background: Container(
                  alignment: Alignment.centerRight,
                  padding: const EdgeInsets.only(right: 20),
                  margin: const EdgeInsets.only(bottom: 8),
                  decoration: BoxDecoration(color: cs.errorContainer, borderRadius: BorderRadius.circular(16)),
                  child: Icon(Icons.delete_outline, color: cs.onErrorContainer),
                ),
                confirmDismiss: (_) => showConfirmDialog(context, title: 'Delete this scan?', message: 'The result and its photos will be removed from this phone.', confirm: 'Delete', destructive: true, icon: Icons.delete_outline),
                onDismissed: (_) async {
                  await AppScope.of(context).history.delete(e);
                  _loadHistory();
                },
                child: _HistoryTile(entry: e, onTap: () => _open(e)),
              ),
        ]),
      ),
    );
  }
}

// ================================================================================== widgets
class _EngineBanner extends StatelessWidget {
  const _EngineBanner({required this.state, required this.status, required this.onRetry});
  final _Engine state;
  final EngineStatus? status;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final (key, icon, title, sub) = switch (state) {
      _Engine.loading => ('none', Icons.sync, 'Preparing the offline reader…', 'Loading the models on this phone'),
      _Engine.ready => (
          'accept',
          Icons.offline_bolt_outlined,
          'Offline reader ready',
          status?.recaptureModel == true ? 'Runs entirely on this phone · screen / print detector on' : 'Runs entirely on this phone · screen / print detector off',
        ),
      _Engine.failed => ('reject', Icons.error_outline, 'The reading engine could not start', status?.error ?? 'Close other apps and try again.'),
    };
    final t = toneFor(context, key);
    final plain = key == 'none';
    return Container(
      key: const ValueKey('engine-banner'),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: plain ? cs.surfaceContainerHighest : t.bg, borderRadius: BorderRadius.circular(14)),
      child: Row(children: [
        state == _Engine.loading ? const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2.4)) : Icon(icon, color: t.fg),
        const SizedBox(width: 12),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(title, style: TextStyle(fontWeight: FontWeight.w700, color: plain ? cs.onSurface : t.fg)),
            if (sub.isNotEmpty) Text(sub, style: TextStyle(fontSize: 12.5, color: plain ? cs.onSurfaceVariant : t.fg)),
          ]),
        ),
        if (state == _Engine.failed) FilledButton.tonal(key: const ValueKey('retry-engine'), onPressed: onRetry, style: FilledButton.styleFrom(minimumSize: const Size(0, 40)), child: const Text('Retry')),
      ]),
    );
  }
}

class _BigAction extends StatelessWidget {
  const _BigAction({super.key, required this.icon, required this.title, required this.subtitle, required this.onTap, this.primary = false});
  final IconData icon;
  final String title, subtitle;
  final VoidCallback onTap;
  final bool primary;

  @override
  Widget build(BuildContext context) {
    return Material(
      borderRadius: BorderRadius.circular(22),
      clipBehavior: Clip.antiAlias,
      child: Ink(
        decoration: BoxDecoration(borderRadius: BorderRadius.circular(22), gradient: const LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [brandNavy, Color(0xFF2A6CB8)])),
        child: InkWell(
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(children: [
              Container(width: 56, height: 56, decoration: BoxDecoration(color: Colors.white24, borderRadius: BorderRadius.circular(16)), child: Icon(icon, color: Colors.white, size: 30)),
              const SizedBox(width: 16),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(title, style: const TextStyle(color: Colors.white, fontSize: 19, fontWeight: FontWeight.w800)),
                  const SizedBox(height: 4),
                  Text(subtitle, style: const TextStyle(color: Colors.white70, height: 1.3)),
                ]),
              ),
              const Icon(Icons.arrow_forward_ios, color: Colors.white70, size: 18),
            ]),
          ),
        ),
      ),
    );
  }
}

class _SmallAction extends StatelessWidget {
  const _SmallAction({super.key, required this.icon, required this.title, required this.subtitle, required this.onTap});
  final IconData icon;
  final String title, subtitle;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Icon(icon, color: cs.primary, size: 28),
            const SizedBox(height: 10),
            Text(title, style: const TextStyle(fontWeight: FontWeight.w800, fontSize: 16)),
            Text(subtitle, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
          ]),
        ),
      ),
    );
  }
}

class _HistoryTile extends StatelessWidget {
  const _HistoryTile({required this.entry, required this.onTap});
  final HistoryEntry entry;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final r = entry.result;
    final first = entry.photoFile(0);
    final key = !r.foundAnything ? 'reject' : (r.auth.isSpoofSuspected ? 'suspected_spoof' : ((r.plate?.needsReview ?? false) && entry.confirmedPlate == null ? 'review' : 'accept'));
    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      clipBehavior: Clip.antiAlias,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(10),
          child: Row(children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: SizedBox(
                width: 64,
                height: 52,
                child: first != null && first.existsSync() ? Image.file(first, fit: BoxFit.cover, cacheWidth: 200, errorBuilder: (_, _, _) => const ColoredBox(color: Colors.black12)) : ColoredBox(color: cs.surfaceContainerHighest, child: Icon(Icons.image_not_supported_outlined, color: cs.onSurfaceVariant)),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                MaskedText(entry.title, size: 17),
                const SizedBox(height: 4),
                Row(children: [
                  if (entry.result.plate != null && entry.categoryKey != null && entry.categoryKey != 'unknown') ...[
                    PlateBadge(entry.categoryKey, width: 26),
                    const SizedBox(width: 6),
                  ],
                  Flexible(
                    child: Text(
                      _when(entry.savedAt) +
                          (entry.result.vin != null && entry.result.plate != null ? ' · plate + VIN' : '') +
                          (entry.result.plate != null && entry.categoryKey != null && entry.categoryKey != 'unknown' ? ' · ${plateTypeOf(entry.categoryKey).label}' : ''),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12.5),
                    ),
                  ),
                ]),
              ]),
            ),
            StatusChip(key, dense: true, label: switch (key) { 'accept' => 'OK', 'review' => 'Review', 'reject' => 'None', _ => 'Flagged' }),
          ]),
        ),
      ),
    );
  }

  static String _when(DateTime t) {
    final d = DateTime.now().difference(t);
    if (d.inMinutes < 1) return 'just now';
    if (d.inHours < 1) return '${d.inMinutes} min ago';
    if (d.inDays < 1) return '${d.inHours} h ago';
    return '${t.day.toString().padLeft(2, '0')}/${t.month.toString().padLeft(2, '0')}/${t.year}';
  }
}

