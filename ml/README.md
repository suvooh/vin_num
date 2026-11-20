# vinplate: India-specific plate + VIN reading, authenticity and capture quality

An edge-first replacement for the generic OCR + regex in `backend/`. This README is the
honest status of the work: what exists and is verified, what exists but is untrained, and what
does not exist yet. **No real-world accuracy figure exists for this system yet. Anything quoted
from `eval --synthetic` is a plumbing check, not a claim about real photographs.**

## Status

| Layer | State | Evidence |
|---|---|---|
| Plate grammar (state codes, standard / BH / diplomatic / military, 1-2 row, legacy) | built, tested | `tests/test_grammar.py` |
| VIN grammar (ISO 3779, year code, India WMI range, optional label, legacy chassis) | built, tested | `tests/test_grammar.py` |
| Grammar-constrained CTC decode, `?` wildcard, dropped-character detection, grammar-cost, multi-frame fusion | built, tested on simulated posteriors | `tests/test_decode.py` |
| Synthetic data generator (HSRP/painted plates, 1-2 row, 5 colour schemes; label / engraved / dot-matrix / stencil VINs; glare, dirt, occlusion, optics) | built, eyeballed | `scripts/preview_synth.py` |
| Line recogniser (conv-only CTC, INT8-friendly) + training loop + shard generator | built, **plumbing verified, NOT trained** | smoke run; CPU is ~12 lines/s so real training needs a GPU |
| ONNX export + INT8 QDQ quantisation | built, verified | `export/` |
| Qualcomm NPU inference (QNN / Hexagon HTP) | verified on this machine | see "Edge numbers" |
| Capture-quality gate (blur, glare, exposure, size, with user guidance) | built, thresholds **provisional** | `tests/test_quality.py` |
| Provenance (EXIF, XMP/IPTC DigitalSourceType, AI-tool markers, C2PA read/verify, EXIF-thumbnail check) | built, tested; valid-C2PA path **not** covered by a test (needs a signing fixture) | `tests/test_authenticity.py` |
| Classical forensics (ELA, noise residual, moire, chroma-moire, banding, JPEG grid) | built, thresholds **provisional** (weakly calibrated on ~100 real photos) | `tests/test_authenticity.py`, `eval/pad_eval.py` |
| **Screen / print recapture simulator** (sub-pixel lattices, halftone, glass reflections, banding) | built, eyeball-checked against physics, **not validated on real devices** | `synth/recapture.py` |
| **Learned recapture detector** (PadNet, native-res patches, 0.24M params) | **trained on CPU in 27 min, simulated attacks only**; INT8 verified lossless | see "Presentation attacks" |
| **Active physical checks**: flash retroreflection, planarity | built; tested against a **toy physics model only**; not validated on real captures | `tests/test_active.py` |
| Authenticity fusion (reject / review / no_flags + assurance + presentation-attack score) | built, tested | `tests/test_authenticity.py` |
| Integration: `Authenticator.assess(capture bundle)` | built, tested | `vinplate/authenticity/assess.py` |
| Licence audit (policy-driven; packages + datasets + models + fonts) | built | `scripts/license_audit.py`, `licenses/` |
| Evaluation harness (accepted-wrong rate, coverage, per-slice; recapture AUC/TPR) | built | `eval/run_eval.py`, `eval/pad_eval.py` |
| **PP-OCR engine with posterior access** (`vinplate/ocr`): DB text detector + recogniser on ONNX Runtime, no OpenCV; recogniser probabilities projected onto the plate / VIN alphabet + `?` | built, tested; the **reader in use today** (pretrained, generic) | `tests/test_ocr_engine.py`, `eval/e2e_eval.py` |
| **Scene reader** (`vinplate/scene.py`): detect -> min-area-rect deskew -> candidates (lines, 2-row pairs, split plates, clustered plate regions) -> photometric views -> grammar decode -> safety policy | built, tested on simulated posteriors, real rendered plates and the synthetic scene benchmark | `tests/test_scene.py`, `eval/e2e_eval.py` |
| **Vehicle category** from plate colours + text layout (`vinplate/category.py`): private / commercial / rental / EV / temporary / trade / diplomatic / military, `bh` flag | built, tested on synthetic plates only (clean 100 %, degraded lighting 92.6 %, dim light abstains); rules **recalled from two infographics, NOT verified against the official notifications**; see `docs/VEHICLE_CATEGORY.md` | `tests/test_category.py`, `eval/category_eval.py` |
| Photometric views for glare / shadow / blur (`vinplate/enhance.py`) | built; `stretch` measurably helps on the benchmark, `deglare` is unmeasured | `tests/test_enhance.py` |
| Capture-bundle liveness (`authenticity/bundle.py`): flash + angle frames, `liveness` field, consistency across frames, `require_liveness` policy | built, tested on toy physics + simulated attacks only | `tests/test_bundle.py` |
| End-to-end synthetic scene benchmark (`eval/scenes.py`, `eval/e2e_eval.py`): synthetic Indian plates / VINs on real car photos, old vs new | built | `tests/test_scenes.py` |
| **Dedicated plate / VIN detector + 4-corner rectifier** | **NOT built** - localisation today is the generic text detector + min-area-rect deskew + clustering of nearby boxes (no perspective correction beyond that) | needs data decisions, see below |
| **Trained tamper (text-edit / AI-inpaint) and AI-generated-image detectors** | **NOT built** (slots exist; reported `not_run`) | needs real tamper data; no public plate/VIN set |
| **Capture app / SDK** (burst, flash/no-flash, IMU, attestation) | **spec only** | `docs/CAPTURE_PROTOCOL.md` |
| Native (Kotlin/Swift/C++) port of grammar + decoder | **NOT built** | Python reference only: ~65 ms plate, ~230 ms VIN |

Run the tests: `.venv\Scripts\python -m pytest tests` (248 pass; the real-engine tests skip if the PP-OCR models are absent).

## Design in one page

```
photo --> [authenticity: provenance + forensics + attestation]  --> reject / review / no_flags
      \-> detector (NOT YET) -> quad -> rectify -> rows -> fit_line -> recogniser (NPU, INT8)
                                                    |
                                  per-row log-probs, concatenated for 2-row plates
                                                    v
        grammar-constrained CTC search  (plate or VIN grammar, '?' = unreadable wildcard)
                                                    v
   Reading { text, masked, status accept|review|reject, char_conf, inferred, suspected_gaps,
             grammar_cost, n_best }        + quality gate + parsed fields (only if not rejected)
```

### How glare, smudges and missing characters are handled (and what is *not* promised)

1. **Prevent**: the quality gate tells the user what to change while a better photo is still obtainable.
2. **Abstain, don't hallucinate**: the recogniser has an explicit `?` class. Training labels a character
   `?` when synthetic glare/dirt/occlusion covers more than 65% of it, so the network learns to say
   "unreadable" instead of inventing a digit.
3. **Resolve ambiguity with structure**: the grammar forbids impossible strings (a digit slot cannot be
   `O`, a state code must exist). A single unreadable VIN character is pinned only where a *mandatory*
   check digit makes it unique; Indian OEM VINs often fail the ISO check, so by default it is weak evidence.
4. **Fuse frames**: burst/video frames are fused on posteriors; glare moves between frames.
5. **Flag everything inferred**: `masked` shows `MH12AB?234`, never the decoder's guess, whenever `status` is
   not `accept`. `grammar_cost` says how far the constraint bent the network's reading.

What this cannot do: recover information that is not in the pixels. A character that is gone stays
gone; the system says so and asks for a retake. A generative "reconstruction" would invent identifiers
in an identity-verification product, so it was deliberately not built.

### Authenticity: what each layer is worth

* **Strongest controls are capture-side**, not pixel-side: in-app camera only, server nonce, hardware-backed
  attestation (Play Integrity / Key Attestation, iOS App Attest), burst + flash/no-flash capture, and a
  registry cross-check. `authenticity.decide(..., attestation=...)` takes the *server-verified* attestation result.
* Provenance: an AI / screenshot declaration is strong evidence; **absence of any marker means nothing**
  (messengers and screenshots strip metadata; most Indian phones sign no C2PA). `no_flags` is not "authentic";
  every decision carries an `assurance` level (`none` / `attested` / `c2pa`).
* Google **SynthID**: the image detector is a portal only (no public API or offline model as of the research
  done 2026-11-07), so it can only be a manual escalation step. It is not implemented and must not be scraped.
* Pixel forensics are corroboration only: they are wiped out by one re-save or resize and the JPEG-grid
  check is uninformative on high-quality (q>~90) camera originals; measured separation is in the tests.
* Open AI-image detectors generalise poorly to unseen generators (published evidence: large AUC drops on
  in-the-wild content). Community Forensics (MIT, 21.8M params) is the only licence-clean candidate and would
  need distilling for a 10 MB budget; keep it server-side as a weak prior.

## Presentation attacks: is this a real car or a photo of a screen / print?

Three independent layers, because no single one is reliable:

1. **Capture-side** (strongest; spec in `docs/CAPTURE_PROTOCOL.md`): in-app camera only, nonce, hardware attestation,
   locked-exposure burst with a flash-OFF/flash-ON pair, IMU. Not built (needs the mobile SDK).
2. **Physics** (`authenticity/active.py`): HSRP sheeting is *retroreflective* - under flash it brightens far more than
   its surround; a screen emits its own light and a print reflects diffusely, so both fail that test regardless of
   screen resolution or moire suppression. Verdicts are `retroreflective / not_retroreflective / inconclusive`;
   it only applies to HSRP plates, needs the plate metered mid-grey (else "no headroom"), and is `inconclusive`
   in strong sun. Planarity (scene motion fits one plane) is weak supporting evidence only. **Tested against a toy
   physics model, not real captures.**
3. **Passive pixels**: classical cues + a learned detector on native-resolution patches.

Measured on **simulated** attacks (`synth/recapture.py`) of **real Open Images photos** (CC BY 2.0, 271 photos,
`data/openimages/ATTRIBUTION.csv`), validated on photos never seen in training:

| Detector | screens | prints |
|---|---|---|
| classical moire (AUC) | 0.84 (detects 47% @ 5% false pos.; only 27% at the common ~1 px display pitch) | 0.84 (halftone 99%, **glossy photo 9%**) |
| classical chroma-moire / banding | 0.72 / 0.64 | 0.75 / 0.55 |
| **PadNet** (AUC; INT8 identical) | **0.993**, and **0.996 at display pitches never trained on** | **0.959** |
| PadNet detection @ 5% false pos. | 98% | 80% |

**How to read that:** the learned model clearly beats the classical cues on the simulator, and quantising to INT8 loses
nothing. But the simulator and the evaluation share one author, the photos are ~640 px Flickr renditions (not native
phone captures), and the "unseen pitch" split is a parameter shift inside the same simulator. **That is a pipeline
check, not field accuracy.** Cross-device generalisation of recapture detectors is the known weak point in the
literature, so expect a real drop; the protocol for measuring it properly (ISO 30107-3 APCER/BPCER, split by
device and plate) is in `docs/CAPTURE_PROTOCOL.md`. Decision rule (`authenticity/fusion.py`): independent signals are
weighted and summed; one weak cue only sends to *review*, a *reject* needs agreement (e.g. failed flash physics +
a high learned score), because wrongly rejecting a genuine customer costs too.

## Edge numbers (measured on this machine: Snapdragon X Plus, onnxruntime-qnn 2.7, batch 1, median of 150 runs)

CPU, GPU (Adreno via the QNN GPU backend) and NPU (Hexagon HTP) all work; the same ONNX file is used.

| model | CPU (ORT) | GPU (FP32) | NPU |
|---|---|---|---|
| Plate-row recogniser, width 1.0, INT8 / FP32 | 0.91 ms / 1.8 ms | 1.0 ms (INT8 unsupported) | **0.26 ms** / 0.34 ms |
| Plate-row recogniser, width 3.0 (3.8 MB INT8) | 3.3 ms INT8; FP32 7 ms best, 45-127 ms typical (unstable) | 2.7 ms | **0.39 ms** INT8 / 0.85 ms FP32 |
| PadNet, 4 patches, 258 KB INT8 / 949 KB FP32 | 33 ms INT8 (best 4.6) / 60 ms FP32 | 2.5 ms | **0.31 ms** INT8 / 0.40 ms FP32 |

Takeaways: the NPU is overhead-bound (width 1 -> 3 barely changes latency), so there is large headroom for a **bigger**
recogniser; INT8 is the portable floor on CPU; the QNN GPU backend wants FP16/FP32, not INT8. Caveats: a laptop SoC,
not a phone; excludes decode (Python ~65-230 ms; native port needed), pre-processing and the detector; ORT warns a few
shape ops stay on CPU; first-session compile is ~3 s, so ship a pre-compiled QNN context binary. The NPU runs
**inference only**: it cannot train models.

## Licensing (commercial viability)

* **Our code and trained weights are yours**: no licence file or header has been added. Third-party components are
  tracked in `licenses/` and checked by `python scripts/license_audit.py` (policy in `licenses/policy.json`, edit it to
  change what is allowed). Current result: **0 denied in the build, 5 need review.**
* Apache-2.0 / MIT / BSD are commercially usable; the real hazards are AGPL/GPL (copyleft) and non-commercial or
  research-only terms. The audit lists the Apache-2.0 dependencies (onnx, cryptography, c2pa-python, huggingface_hub)
  so a policy that excludes Apache-2.0 can see exactly what it would affect.
* **The Qualcomm NPU runtime is the one real commercial risk.** The QNN libraries inside `onnxruntime-qnn` are under
  Qualcomm's proprietary AI Stack License (read from the shipped PDF): non-exclusive, non-transferable, **revocable**;
  redistribution only in object code embedded in your app, not standalone; no reverse engineering; $100 liability
  cap; terminates automatically on breach; export-control terms. The plan therefore treats the NPU as an optional
  accelerator over a fully working CPU/GPU path built on MIT-licensed ONNX Runtime. Get legal sign-off.
* Items needing review: `tqdm` (MPL-2.0, training-time only), the Qualcomm runtime, the Kaggle "CC0" plate sets
  (uploader-asserted on scraped images), Open Images (CC BY: keep `ATTRIBUTION.csv`, attribute authors), and
  `keremberke` (CC BY, mixed provenance). Avoid: Ultralytics YOLO (AGPL), TruFor/B-Free/NPR (NC or unlicensed),
  GenImage/Synthbuster/OpenFake/DocTamper (NC), COCO images (mixed incl. NC).
* This is not legal advice.

## Training (GPU required)

PyTorch on this ARM64 CPU manages ~12 lines/s. Use a GPU pod:

```
bash scripts/runpod_train.sh 800000 500000 40000 2.0     # shards -> train -> export -> tarball
```

Live Runpod prices read 2026-11-07 (secure cloud, $/h): RTX 3090 0.50 (32 vCPU min, scarce), A40 0.59, RTX 4090 0.89,
L40S 1.09, A100 1.79. The job is **CPU-bound** (synthetic generation, ~15 lines/s/core) more than GPU-bound; the
estimate for the first run is a few GPU-hours, i.e. on the order of $3-5. That is an estimate, not a measurement.

## Data: what is available, honestly (research 2026-11-07)

* Every real Indian plate set with transcriptions found is **non-commercial or unlicensed** (e.g. witvix truck set:
  CC BY-NC, 470 photos; DataCluster HF sample: CC BY-NC-ND; several Kaggle sets: NC-ND or unverifiable "CC0" on
  scraped images). Several HF repos that looked relevant are empty stubs or not Indian.
* **No public VIN / chassis-number photo dataset exists** (confirmed across HF, Kaggle, GitHub, arXiv). No
  public 4-corner Indian plate set either.
* Open Images V7 "Vehicle registration plate" (CC BY 4.0 annotations, CC BY 2.0 images) and
  `keremberke/license-plate-object-detection` (CC BY 4.0) are usable for *detector* pretraining but are not Indian.
* Therefore: synthetic data trains the recogniser; **real Indian data must come from your own captures**
  (Hub Gate photos, dealer uploads) with consent and DPDP-Act handling. Plates and vehicles are personal data.
  You need a real, held-out, labelled set including two-wheelers, EV green plates, BH series, hand-painted
  plates, and *VIN photos of every type above* before any accuracy claim is made.
* HSRP typeface: no free exact match (Rule 51 prescribes dimensions, not a typeface). DIN-like OFL fonts stand
  in (`synth/fonts`, licences alongside); expect a domain gap closed only by real data.
* Detector licence: Ultralytics YOLO is AGPL-3.0 (enterprise licence needed for closed commercial use) - avoid.
  Commercially usable options: Apache-2.0 (YOLOX, RT-DETR, D-FINE) or, if Apache-2.0 is ruled out by policy,
  BSD-3 architectures from torchvision (SSDLite / FCOS) trained from scratch on our own data (their pretrained
  COCO/ImageNet weights carry separate data terms, so do not start from them).

## Known gaps and risks

* **Presentation-attack numbers are simulator-on-real-photos, not real-device attacks** (see above). The retroreflection
  check is HSRP-only, needs a flash-on frame and plate-metered exposure, and has only met a toy physics model.
* There is no dedicated plate / VIN detector: the generic text detector finds the text, so a plate with no or fragmented
  detections, extreme perspective, or a VIN stamped in metal (low contrast) can be missed entirely. A 2-row plate whose
  rows are cut by a scratch is only read as a fragment.
* The reader in use is the generic PP-OCR recogniser. On the 60-scene synthetic benchmark its grammar-decoded output reads
  about as many plates exactly as greedy text (81.6% vs 78.9%, within noise); what the stack adds is abstention (`?`, review)
  and structural validity - accepted-but-wrong fell from 13.2% to 5.3% (intervals overlap). Glare scenes are mostly flagged,
  not read. A real accuracy step needs the purpose-built recogniser trained on real + synthetic data.
* Real cross-device recapture generalisation is the known weak point of this whole class of detector.

* VIN WMI list is a soft prior only; only 3 WMIs (MAJ, MD2, ME3) were verified against NHTSA vPIC, which does not
  cover most Indian manufacturers. The RTO-number table is deliberately absent (hook: `RTO_ALLOWED`).
* All thresholds (quality gate, forensics, `REVIEW_COST`, `min_seq`, `max_inferred`) are provisional until
  calibrated on real data; `calibrated=False` is carried in the outputs.
* The synthetic generator is written from how marks look, not fitted to real photos (especially VIN stamps).
* `c2pa-python` must be pinned to 0.38.0 (0.38.1/0.38.2 ship no Windows wheels) and needs a binary `cryptography` wheel.
