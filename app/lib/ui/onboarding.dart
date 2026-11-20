import 'package:flutter/material.dart';

class _Page {
  const _Page(this.icon, this.title, this.body);
  final IconData icon;
  final String title, body;
}

const _pages = [
  _Page(Icons.document_scanner_outlined, 'Read plates and VINs', 'Photograph an Indian number plate or a VIN. Characters the photo cannot support are shown as a red ?, never guessed, so you always see what is certain.'),
  _Page(Icons.center_focus_strong_outlined, 'One clear photo beats everything', 'Stand square-on, fill the frame, hold steady and move so reflections slide off the characters. You can retake any photo before it is sent.'),
  _Page(Icons.verified_user_outlined, 'Prove it is a real car', 'The app takes a front photo with and without the torch, then one from your left and one from your right, to check that a real, three-dimensional plate was photographed, not a picture on a screen. Photos come only from the live camera: nothing can be added from the gallery.'),
  _Page(Icons.lock_outline, 'Your photos stay with you', 'Everything is read on this phone, offline: no photo ever leaves the device. A copy is kept here only if History is on, and you can clear it any time.'),
];

/// Full-screen, dismissible walkthrough. Resolves when it is closed either way.
Future<void> showOnboarding(BuildContext context) {
  return Navigator.of(context).push(MaterialPageRoute<void>(fullscreenDialog: true, builder: (_) => const _Onboarding()));
}

class _Onboarding extends StatefulWidget {
  const _Onboarding();
  @override
  State<_Onboarding> createState() => _OnboardingState();
}

class _OnboardingState extends State<_Onboarding> {
  final ctl = PageController();
  int page = 0;

  @override
  void dispose() {
    ctl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final last = page == _pages.length - 1;
    return Scaffold(
      body: SafeArea(
        child: Column(children: [
          Align(alignment: Alignment.centerRight, child: TextButton(key: const ValueKey('skip-intro'), onPressed: () => Navigator.pop(context), child: const Text('Skip'))),
          Expanded(
            child: PageView.builder(
              controller: ctl,
              itemCount: _pages.length,
              onPageChanged: (i) => setState(() => page = i),
              itemBuilder: (ctx, i) => Padding(
                padding: const EdgeInsets.symmetric(horizontal: 32),
                child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Container(
                    width: 132,
                    height: 132,
                    decoration: BoxDecoration(shape: BoxShape.circle, color: cs.primaryContainer),
                    child: Icon(_pages[i].icon, size: 64, color: cs.onPrimaryContainer),
                  ),
                  const SizedBox(height: 36),
                  Text(_pages[i].title, textAlign: TextAlign.center, style: Theme.of(context).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800)),
                  const SizedBox(height: 14),
                  Text(_pages[i].body, textAlign: TextAlign.center, style: TextStyle(color: cs.onSurfaceVariant, fontSize: 16, height: 1.45)),
                ]),
              ),
            ),
          ),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            for (var i = 0; i < _pages.length; i++)
              AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                margin: const EdgeInsets.symmetric(horizontal: 4),
                width: i == page ? 24 : 8,
                height: 8,
                decoration: BoxDecoration(color: i == page ? cs.primary : cs.outlineVariant, borderRadius: BorderRadius.circular(4)),
              ),
          ]),
          Padding(
            padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
            child: SizedBox(
              width: double.infinity,
              child: FilledButton(
                key: const ValueKey('intro-next'),
                onPressed: () => last ? Navigator.pop(context) : ctl.nextPage(duration: const Duration(milliseconds: 280), curve: Curves.easeOut),
                child: Text(last ? 'Get started' : 'Next'),
              ),
            ),
          ),
        ]),
      ),
    );
  }
}
