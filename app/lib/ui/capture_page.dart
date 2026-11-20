import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../app_services.dart';
import '../data/scanner.dart';
import '../data/models.dart';
import 'capture_plan.dart';
import 'result_page.dart';
import 'theme.dart';
import 'widgets.dart';

/// Guided capture from the LIVE CAMERA ONLY (there is deliberately no gallery or file picker, so a
/// saved or edited image can never be submitted). One screen, one pose per step, a Retake / Use
/// preview after every shot, and a review sheet before anything is read.
class CapturePage extends StatefulWidget {
  const CapturePage({super.key, required this.mode, this.cameraLoader});
  final ScanMode mode;

  /// Test seam: how the available cameras are listed.
  final Future<List<CameraDescription>> Function()? cameraLoader;

  @override
  State<CapturePage> createState() => _CapturePageState();
}

class _CapturePageState extends State<CapturePage> with WidgetsBindingObserver {
  late final CapturePlan plan = CapturePlan.forMode(widget.mode);
  final Map<String, Uint8List> shots = {}; // by step id; a flash frame is '<id>_flash'
  int step = 0;
  CameraController? cam;
  String? cameraError;
  bool cameraDenied = false;
  bool starting = true;
  bool busy = false;
  bool exposureLocked = false;

  StepSpec get cur => plan.steps[step];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    WidgetsBinding.instance.addPostFrameCallback((_) => _startCamera());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    cam?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState s) {
    final c = cam;
    if (c == null || !c.value.isInitialized) return;
    if (s == AppLifecycleState.inactive) {
      cam = null;
      c.dispose();
      // the preview is gone until the app resumes: say "starting", not "no camera"
      if (mounted) setState(() => starting = true);
    } else if (s == AppLifecycleState.resumed) {
      _startCamera();
    }
  }

  // -------------------------------------------------------------------------------- camera
  Future<void> _startCamera() async {
    final services = AppScope.of(context);
    setState(() {
      starting = true;
      cameraError = null;
      cameraDenied = false;
    });
    // Say WHY before the system asks, once.
    if (!services.settings.cameraRationaleShown) {
      await showDialog<void>(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => AlertDialog(
          icon: const Icon(Icons.photo_camera_outlined, size: 32),
          title: const Text('Camera access'),
          content: const Text('The camera is used only to photograph the number plate and VIN, live: photos cannot be added from the gallery. '
              "Everything is read on this phone - nothing is uploaded - and a copy can be kept in this phone's history if you leave History on."),
          actions: [FilledButton(onPressed: () => Navigator.pop(ctx), child: const Text('Continue'))],
        ),
      );
      await services.settings.update(cameraRationaleShown: true);
      if (!mounted) return;
    }
    try {
      final cams = await (widget.cameraLoader ?? availableCameras)();
      if (cams.isEmpty) throw CameraException('NoCamera', 'No camera was found on this device.');
      final c = CameraController(cams.firstWhere((d) => d.lensDirection == CameraLensDirection.back, orElse: () => cams.first), ResolutionPreset.high,
          enableAudio: false, imageFormatGroup: ImageFormatGroup.jpeg);
      await c.initialize();
      try {
        await c.lockCaptureOrientation(DeviceOrientation.portraitUp);
      } catch (_) {}
      if (!mounted) {
        await c.dispose();
        return;
      }
      exposureLocked = false;
      setState(() {
        cam = c;
        starting = false;
      });
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() {
        starting = false;
        cameraDenied = e.code.contains('Denied') || e.code.contains('Restricted');
        cameraError = cameraDenied ? 'Camera permission is turned off.' : (e.description ?? 'The camera could not start (${e.code}).');
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        starting = false;
        cameraError = 'The camera could not start: $e';
      });
    }
  }

  Future<void> _setExposureLock(bool lock) async {
    final c = cam;
    if (c == null || exposureLocked == lock) return;
    try {
      await c.setExposureMode(lock ? ExposureMode.locked : ExposureMode.auto);
      exposureLocked = lock;
    } catch (_) {
      // not every device supports locking; the flash test then reports it cannot judge
    }
  }

  /// Takes the photo for the current step. The front step takes TWO frames from the same spot, one
  /// after the other with no action from the person: flash off, then the torch on.
  Future<void> _shutter() async {
    final c = cam;
    if (c == null || busy || !c.value.isInitialized) return;
    setState(() => busy = true);
    HapticFeedback.selectionClick();
    try {
      Uint8List? flash;
      String? flashNote;
      Uint8List bytes;
      if (cur.withFlash) {
        await _setExposureLock(true); // identical exposure for the pair
        try {
          await c.setFlashMode(FlashMode.off);
        } catch (_) {}
        bytes = await (await c.takePicture()).readAsBytes();
        try {
          await c.setFlashMode(FlashMode.torch);
          await Future<void>.delayed(const Duration(milliseconds: 350));
          flash = await (await c.takePicture()).readAsBytes();
        } catch (_) {
          flashNote = "This phone didn't let the app turn the torch on, so the flash check can't be done for this capture.";
        } finally {
          try {
            await c.setFlashMode(FlashMode.off);
          } catch (_) {}
          await _setExposureLock(false);
        }
      } else {
        try {
          await c.setFlashMode(FlashMode.off);
        } catch (_) {}
        bytes = await (await c.takePicture()).readAsBytes();
      }
      if (!mounted) return;
      await _preview(bytes, flash: flash, flashNote: flashNote);
    } on CameraException catch (e) {
      if (mounted) {
        await showMessageDialog(context, title: "The photo wasn't taken", message: e.description ?? e.code, icon: Icons.error_outline);
      }
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  // ---------------------------------------------------------------------------------- steps
  Future<void> _preview(Uint8List bytes, {Uint8List? flash, String? flashNote}) async {
    final use = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      isDismissible: false,
      enableDrag: false,
      builder: (ctx) => _PreviewSheet(bytes: bytes, flash: flash, flashNote: flashNote, spec: cur),
    );
    if (use == true) {
      shots[cur.id] = bytes;
      if (flash != null) {
        shots[cur.flashKey] = flash;
      } else {
        shots.remove(cur.flashKey);
      }
      await _advance();
    }
  }

  Future<void> _advance() async {
    if (step < plan.steps.length - 1) {
      setState(() => step++);
    } else {
      await _review();
    }
  }

  Future<void> _skip() async {
    final ok = await showConfirmDialog(context,
        title: 'Skip "${cur.label}"?', message: cur.skipWarning, confirm: 'Skip anyway', cancel: 'Take it', icon: Icons.warning_amber_rounded);
    if (!ok) return;
    await _advance();
  }

  void _goTo(int i) => setState(() => step = i);

  Future<void> _review() async {
    final action = await showModalBottomSheet<_ReviewAction>(
      context: context,
      isScrollControlled: true,
      builder: (ctx) => _ReviewSheet(plan: plan, shots: shots),
    );
    if (!mounted || action == null) return;
    if (action.retake != null) {
      final s = plan.steps[action.retake!];
      shots.remove(s.id);
      shots.remove(s.flashKey);
      _goTo(action.retake!);
    } else if (action.submit) {
      await _submit();
    }
  }

  Future<void> _submit() async {
    final services = AppScope.of(context);
    final photos = plan.photos(shots);
    var result = await _runWithDialog(services.scanner, photos, services.settings.hsrp, services.settings.requireLiveness);
    while (result is ScanException) {
      if (!mounted || result.kind == ScanErrorKind.cancelled) return;
      final act = await showScanErrorDialog(context, result);
      if (!mounted || act != ErrorAction.retry) return;
      result = await _runWithDialog(services.scanner, photos, services.settings.hsrp, services.settings.requireLiveness);
    }
    if (result is! ScanResult || !mounted) return;
    final nav = Navigator.of(context);
    await nav.pushReplacement(MaterialPageRoute<void>(
      builder: (_) => ResultPage(result: result as ScanResult, photos: {for (var i = 0; i < photos.length; i++) i: photos[i].bytes}),
    ));
  }

  /// Shows a modal progress dialog while the scan runs. Returns a [ScanResult] or a [ScanException].
  Future<Object> _runWithDialog(Scanner scanner, List<CapturedPhoto> photos, bool hsrp, bool requireLiveness) async {
    unawaited(showDialog<void>(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => _ProcessingDialog(count: photos.length, onCancel: scanner.cancel),
    ));
    Object out;
    try {
      out = await scanner.scan(photos, hsrp: hsrp, requireLiveness: requireLiveness);
    } on ScanException catch (e) {
      out = e;
    } catch (e) {
      out = ScanException(ScanErrorKind.failed, 'Something unexpected went wrong.', detail: '$e');
    }
    if (mounted) Navigator.of(context, rootNavigator: true).pop(); // the progress dialog
    return out;
  }

  Future<void> _close() async {
    final taken = plan.photos(shots).length;
    if (taken > 0) {
      final ok = await showConfirmDialog(context,
          title: 'Discard this scan?', message: 'The $taken photo${taken == 1 ? '' : 's'} you took will be deleted.', confirm: 'Discard', cancel: 'Keep scanning', destructive: true, icon: Icons.delete_outline);
      if (!ok) return;
    }
    if (mounted) Navigator.of(context).pop();
  }

  // ------------------------------------------------------------------------------------ UI
  @override
  Widget build(BuildContext context) {
    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) _close();
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        body: SafeArea(
          child: Stack(fit: StackFit.expand, children: [
            _cameraLayer(),
            Positioned(top: 0, left: 0, right: 0, child: _topBar()),
            Positioned(left: 0, right: 0, bottom: 0, child: _bottomPanel()),
          ]),
        ),
      ),
    );
  }

  Widget _cameraLayer() {
    final c = cam;
    if (c != null && c.value.isInitialized) {
      final ps = c.value.previewSize!;
      return Stack(fit: StackFit.expand, children: [
        ClipRect(
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(width: ps.height, height: ps.width, child: CameraPreview(c)),
          ),
        ),
        _GuideOverlay(kind: cur.guide),
      ]);
    }
    if (starting) {
      return const Center(child: Column(mainAxisSize: MainAxisSize.min, children: [CircularProgressIndicator(), SizedBox(height: 14), Text('Starting camera…', style: TextStyle(color: Colors.white70))]));
    }
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(28),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          Icon(cameraDenied ? Icons.no_photography : Icons.videocam_off, size: 56, color: Colors.white70),
          const SizedBox(height: 14),
          Text(cameraError ?? 'No camera', textAlign: TextAlign.center, style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Text(
            cameraDenied
                ? 'Open Settings › Apps › this app › Permissions and allow the Camera, then come back and try again. Photos can only be taken with the live camera.'
                : 'Try again. Photos can only be taken with the live camera.',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white70),
          ),
          const SizedBox(height: 18),
          FilledButton.icon(onPressed: _startCamera, icon: const Icon(Icons.refresh), label: const Text('Try again')),
        ]),
      ),
    );
  }

  Widget _topBar() {
    return Container(
      padding: const EdgeInsets.fromLTRB(8, 8, 12, 8),
      decoration: const BoxDecoration(gradient: LinearGradient(begin: Alignment.topCenter, end: Alignment.bottomCenter, colors: [Colors.black87, Colors.transparent])),
      child: Row(children: [
        IconButton(tooltip: 'Close', onPressed: _close, icon: const Icon(Icons.close, color: Colors.white)),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(plan.title, style: const TextStyle(color: Colors.white70, fontSize: 12)),
            Text('Step ${step + 1} of ${plan.steps.length} · ${cur.label}', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700, fontSize: 16)),
          ]),
        ),
        _StepDots(plan: plan, step: step, shots: shots, onTap: (i) => _goTo(i)),
      ]),
    );
  }

  Widget _bottomPanel() {
    final cs = Theme.of(context).colorScheme;
    final ref = (cur.id == 'left' || cur.id == 'right') ? shots['front'] : null; // the pose to stay close to
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
      decoration: const BoxDecoration(gradient: LinearGradient(begin: Alignment.bottomCenter, end: Alignment.topCenter, colors: [Colors.black, Colors.black87, Colors.transparent], stops: [0, 0.75, 1])),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Row(children: [
          Icon(cur.icon, color: cs.secondary, size: 20),
          const SizedBox(width: 8),
          Expanded(child: Text(cur.title, style: const TextStyle(color: Colors.white, fontSize: 17, fontWeight: FontWeight.w700))),
        ]),
        const SizedBox(height: 6),
        Text(cur.instruction, style: const TextStyle(color: Colors.white70, height: 1.35)),
        const SizedBox(height: 10),
        Wrap(spacing: 6, runSpacing: 6, children: [for (final t in cur.tips) _TipChip(t)]),
        const SizedBox(height: 14),
        Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
          SizedBox(
            width: 84,
            child: ref == null
                ? const SizedBox.shrink()
                : Column(mainAxisSize: MainAxisSize.min, children: [
                    ClipRRect(borderRadius: BorderRadius.circular(8), child: Image.memory(ref, width: 64, height: 48, fit: BoxFit.cover, cacheWidth: 200)),
                    const SizedBox(height: 3),
                    const Text('front photo', style: TextStyle(color: Colors.white54, fontSize: 11)),
                  ]),
          ),
          _Shutter(busy: busy, enabled: cam != null && !busy, flash: cur.withFlash, onTap: _shutter),
          SizedBox(
            width: 84,
            child: cur.optional
                ? TextButton(style: TextButton.styleFrom(foregroundColor: Colors.white), onPressed: busy ? null : _skip, child: const Text('Skip'))
                : (plan.canSubmit(shots) ? TextButton(onPressed: busy ? null : _review, child: const Text('Review')) : const SizedBox.shrink()),
          ),
        ]),
      ]),
    );
  }
}

// ======================================================================= small widgets
class _TipChip extends StatelessWidget {
  const _TipChip(this.text);
  final String text;
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 4),
        decoration: BoxDecoration(color: Colors.white12, borderRadius: BorderRadius.circular(20)),
        child: Text(text, style: const TextStyle(color: Colors.white, fontSize: 12)),
      );
}

class _StepDots extends StatelessWidget {
  const _StepDots({required this.plan, required this.step, required this.shots, required this.onTap});
  final CapturePlan plan;
  final int step;
  final Map<String, Uint8List> shots;
  final void Function(int) onTap;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    if (plan.steps.length == 1) return const SizedBox.shrink();
    return Row(children: [
      for (var i = 0; i < plan.steps.length; i++)
        GestureDetector(
          onTap: () => onTap(i),
          child: Container(
            margin: const EdgeInsets.only(left: 6),
            width: i == step ? 22 : 10,
            height: 10,
            decoration: BoxDecoration(
              color: shots[plan.steps[i].id] != null ? cs.secondary : (i == step ? Colors.white : Colors.white30),
              borderRadius: BorderRadius.circular(5),
            ),
          ),
        ),
    ]);
  }
}

class _Shutter extends StatelessWidget {
  const _Shutter({required this.busy, required this.enabled, required this.flash, required this.onTap});
  final bool busy, enabled, flash;
  final VoidCallback onTap;
  @override
  Widget build(BuildContext context) {
    return Semantics(
      button: true,
      label: flash ? 'Take the photo and the flash photo' : 'Take photo',
      child: GestureDetector(
        key: const ValueKey('shutter'),
        onTap: enabled ? onTap : null,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 150),
          width: 78,
          height: 78,
          padding: const EdgeInsets.all(5),
          decoration: BoxDecoration(shape: BoxShape.circle, border: Border.all(color: Colors.white, width: 4)),
          child: Container(
            decoration: BoxDecoration(shape: BoxShape.circle, color: enabled ? Colors.white : Colors.white38),
            child: busy
                ? const Padding(padding: EdgeInsets.all(20), child: CircularProgressIndicator(strokeWidth: 3, color: Colors.black54))
                : (flash ? const Icon(Icons.flash_auto, color: Colors.black87, size: 30) : null),
          ),
        ),
      ),
    );
  }
}

/// Darkens everything except the guide window and draws its corner brackets.
class _GuideOverlay extends StatelessWidget {
  const _GuideOverlay({required this.kind});
  final GuideKind kind;
  @override
  Widget build(BuildContext context) => IgnorePointer(child: CustomPaint(painter: _GuidePainter(kind, Theme.of(context).colorScheme.secondary)));
}

class _GuidePainter extends CustomPainter {
  _GuidePainter(this.kind, this.accent);
  final GuideKind kind;
  final Color accent;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width * (kind == GuideKind.vin ? 0.9 : 0.8);
    final h = kind == GuideKind.vin ? w / 7 : w / 2.2; // 1- and 2-row plates both fit
    final hole = Rect.fromCenter(center: Offset(size.width / 2, size.height * 0.36), width: w, height: h);
    final rr = RRect.fromRectAndRadius(hole, const Radius.circular(14));
    final dim = Path.combine(PathOperation.difference, Path()..addRect(Offset.zero & size), Path()..addRRect(rr));
    canvas.drawPath(dim, Paint()..color = Colors.black.withValues(alpha: 0.45));
    final p = Paint()
      ..color = accent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 4
      ..strokeCap = StrokeCap.round;
    const L = 26.0;
    for (final (dx, dy) in const [(-1, -1), (1, -1), (1, 1), (-1, 1)]) {
      final corner = Offset(dx < 0 ? hole.left : hole.right, dy < 0 ? hole.top : hole.bottom);
      canvas.drawLine(corner, corner + Offset(-dx * L, 0), p);
      canvas.drawLine(corner, corner + Offset(0, -dy * L), p);
    }
  }

  @override
  bool shouldRepaint(_GuidePainter old) => old.kind != kind || old.accent != accent;
}

// ======================================================================== modal sheets
class _PreviewSheet extends StatelessWidget {
  const _PreviewSheet({required this.bytes, required this.spec, this.flash, this.flashNote});
  final Uint8List bytes;
  final Uint8List? flash;
  final String? flashNote;
  final StepSpec spec;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final maxH = MediaQuery.sizeOf(context).height * 0.4;
    Widget pic(Uint8List b, String caption) => Expanded(
          child: Column(children: [
            ClipRRect(borderRadius: BorderRadius.circular(14), child: ConstrainedBox(constraints: BoxConstraints(maxHeight: maxH), child: Image.memory(b, fit: BoxFit.contain))),
            const SizedBox(height: 4),
            Text(caption, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 12)),
          ]),
        );
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('Check the ${spec.label.toLowerCase()} photo${flash != null ? 's' : ''}', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          Text('Is every character sharp and free of glare? If not, retake it - a clear photo matters more than anything the app can do afterwards.',
              style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
          const SizedBox(height: 12),
          if (flash != null)
            Row(crossAxisAlignment: CrossAxisAlignment.start, children: [pic(bytes, 'normal'), const SizedBox(width: 10), pic(flash!, 'with the torch')])
          else
            Row(children: [pic(bytes, spec.withFlash ? 'normal' : '')]),
          if (flashNote != null) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: toneFor(context, 'not_established').bg, borderRadius: BorderRadius.circular(12)),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(Icons.flash_off, size: 18, color: toneFor(context, 'not_established').fg),
                const SizedBox(width: 8),
                Expanded(child: Text(flashNote!, style: TextStyle(fontSize: 13, color: toneFor(context, 'not_established').fg))),
              ]),
            ),
          ],
          const SizedBox(height: 14),
          Row(children: [
            Expanded(child: OutlinedButton.icon(key: const ValueKey('retake'), onPressed: () => Navigator.pop(context, false), icon: const Icon(Icons.replay), label: const Text('Retake'))),
            const SizedBox(width: 12),
            Expanded(child: FilledButton.icon(key: const ValueKey('use-photo'), onPressed: () => Navigator.pop(context, true), icon: const Icon(Icons.check), label: Text(flash != null ? 'Use photos' : 'Use photo'))),
          ]),
        ]),
      ),
    );
  }
}

class _ReviewAction {
  const _ReviewAction({this.retake, this.submit = false});
  final int? retake;
  final bool submit;
}

class _ReviewSheet extends StatelessWidget {
  const _ReviewSheet({required this.plan, required this.shots});
  final CapturePlan plan;
  final Map<String, Uint8List> shots;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final n = plan.photos(shots).length;
    final skipped = [for (final s in plan.missing(shots)) if (s.optional) s];
    final flashOk = plan.hasFlash(shots);
    final warn = toneFor(context, 'not_established');
    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 0, 20, 16),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Text('Ready to analyse', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 4),
          Text('$n photo${n == 1 ? '' : 's'} will be read on this phone. Tap a photo to retake it.', style: TextStyle(color: cs.onSurfaceVariant)),
          const SizedBox(height: 14),
          SizedBox(
            height: 128,
            child: ListView(scrollDirection: Axis.horizontal, children: [
              for (var i = 0; i < plan.steps.length; i++)
                Padding(
                  padding: const EdgeInsets.only(right: 10),
                  child: InkWell(
                    borderRadius: BorderRadius.circular(12),
                    onTap: () => Navigator.pop(context, _ReviewAction(retake: i)),
                    child: Column(children: [
                      Container(
                        width: 104,
                        height: 96,
                        decoration: BoxDecoration(borderRadius: BorderRadius.circular(12), border: Border.all(color: cs.outlineVariant), color: cs.surfaceContainerHighest),
                        clipBehavior: Clip.antiAlias,
                        child: Stack(fit: StackFit.expand, children: [
                          shots[plan.steps[i].id] != null ? Image.memory(shots[plan.steps[i].id]!, fit: BoxFit.cover, cacheWidth: 300) : Center(child: Icon(Icons.add_a_photo_outlined, color: cs.onSurfaceVariant)),
                          if (plan.steps[i].withFlash && shots[plan.steps[i].flashKey] != null)
                            const Positioned(right: 4, top: 4, child: CircleAvatar(radius: 11, backgroundColor: Colors.amber, child: Icon(Icons.flash_on, size: 14, color: Colors.black87))),
                        ]),
                      ),
                      const SizedBox(height: 4),
                      Text(plan.steps[i].label + (shots[plan.steps[i].id] == null ? ' (skipped)' : ''), style: const TextStyle(fontSize: 12)),
                    ]),
                  ),
                ),
            ]),
          ),
          if (plan.mode != ScanMode.vin && !flashOk) ...[
            const SizedBox(height: 10),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: warn.bg, borderRadius: BorderRadius.circular(12)),
              child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Icon(Icons.flash_off, size: 18, color: warn.fg),
                const SizedBox(width: 8),
                Expanded(child: Text("There is no flash photo (the torch didn't fire), so the flash check can't be done. The left and right photos still help read the plate, but they cannot prove it is real.", style: TextStyle(fontSize: 13, color: warn.fg))),
              ]),
            ),
          ],
          if (skipped.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text('Skipped: ${skipped.map((s) => s.label).join(', ')}', style: TextStyle(color: cs.onSurfaceVariant, fontSize: 13)),
          ],
          const SizedBox(height: 16),
          FilledButton.icon(
            key: const ValueKey('analyse'),
            onPressed: plan.canSubmit(shots) ? () => Navigator.pop(context, const _ReviewAction(submit: true)) : null,
            icon: const Icon(Icons.search),
            label: const Text('Analyse'),
          ),
          const SizedBox(height: 8),
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Back to camera')),
        ]),
      ),
    );
  }
}

class _ProcessingDialog extends StatefulWidget {
  const _ProcessingDialog({required this.count, required this.onCancel});
  final int count;
  final VoidCallback onCancel;
  @override
  State<_ProcessingDialog> createState() => _ProcessingDialogState();
}

class _ProcessingDialogState extends State<_ProcessingDialog> {
  final _sw = Stopwatch()..start();
  Timer? _t;

  @override
  void initState() {
    super.initState();
    _t = Timer.periodic(const Duration(seconds: 1), (_) => mounted ? setState(() {}) : null);
  }

  @override
  void dispose() {
    _t?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // Indeterminate on purpose: the engine does not report progress, so no fake stages are shown.
    return PopScope(
      canPop: false,
      child: AlertDialog(
        title: const Text('Analysing…'),
        content: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
          const LinearProgressIndicator(),
          const SizedBox(height: 16),
          Text('Reading ${widget.count} photo${widget.count == 1 ? '' : 's'} on this phone and checking them for screens or prints. Nothing leaves the device.'),
          const SizedBox(height: 10),
          Text('${_sw.elapsed.inSeconds}s elapsed', style: TextStyle(color: Theme.of(context).colorScheme.onSurfaceVariant)),
        ]),
        actions: [TextButton(onPressed: widget.onCancel, child: const Text('Cancel'))],
      ),
    );
  }
}
