# Capture protocol, runtime strategy and data collection

Why this document exists: the strongest defences against "photo of a screen / print / replayed
photo" are decided **when the picture is taken**, not by a classifier afterwards, and every
accuracy claim for this system depends on real captures that do not exist yet. Specs below are
for the mobile SDK; the server-side counterparts are `vinplate.authenticity.{active,fusion}`.

## 1. Capture flow (mobile SDK requirements)

1. **In-app camera only.** No gallery / file picker for the verification flow. Uploads of existing
   files go to a separate, lower-assurance path (`assurance = "none"`).
2. **Server nonce.** Server issues a one-time nonce when the session starts; expires in <= 2 min.
3. **Meter on the plate.** Detector gives the plate ROI; set exposure compensation so the plate's
   *linear* luminance in the flash-off frame is about 0.25-0.35 (mid-grey). Reason: the flash test
   measures how much brighter retroreflective sheeting gets, and a plate already near white has no
   headroom (the code returns `inconclusive: no headroom` rather than guess).
4. **Lock** exposure time, ISO, white balance and focus for the whole burst.
5. **Burst:** 3 frames flash/torch OFF, then 1-2 frames torch ON (<= 150 ms after), then 2 frames OFF.
   Use torch mode (continuous) rather than strobe flash so exposure stays locked.
   Camera moves slightly during the burst (ask the user to sway a few cm): gives parallax for
   `planarity` and different glare positions for multi-frame fusion.
6. **IMU:** accelerometer + gyroscope at >= 100 Hz across the burst, to tell translation from
   rotation (parallax needs translation; pure rotation also looks planar).
7. **Attest and bind.** Hash (SHA-256) the frames + IMU + nonce; sign the digest with a
   hardware-backed key (Android Key Attestation / StrongBox; iOS App Attest assertion). Send a Play
   Integrity token whose `requestHash` is that digest. Neither proves the pixels came from the
   sensor - they prove a genuine app on a genuine device, bound to this request.
8. **Server verifies** (never trust the client's own verdict): attestation chain, device-integrity
   verdict, nonce freshness, digest match, then `authenticity.decide(..., attestation=..., active=...)`.
   Near-duplicate check: store `dhash`/sha256 of every accepted capture and flag reuse.
9. **Registry cross-check** (strongest control, out of this repo): plate/VIN against the registry
   record, vehicle make/colour vs what the camera sees, plausibility of place/time.

What the flash test can and cannot say (do not over-read it):
* Applies only to **retroreflective (HSRP)** plates. Old painted plates are non-retroreflective for
  innocent reasons, so for them the check is `inconclusive` (the SDK must pass `hsrp_expected`).
* Worthless in strong sunlight (plate saturated) or if the torch did not fire -> `inconclusive`.
* An attacker with a *retroreflective print* could pass it. It raises the cost of attack; it is not proof.

## 2. Runtime strategy: CPU + GPU + NPU without lock-in

Use **ONNX Runtime Mobile (MIT)** as the single runtime and pick an execution provider at start-up:

| Platform | preferred | fallback 1 | fallback 2 |
|---|---|---|---|
| Android (Snapdragon) | QNN EP -> HTP NPU (INT8) | NNAPI EP (GPU/DSP) or QNN GPU (FP16) | XNNPACK CPU (INT8/FP32) |
| Android (other SoCs) | NNAPI EP | XNNPACK CPU | - |
| iOS | CoreML EP (ANE/GPU) | CPU | - |
| Windows on ARM / desktop | QNN EP | DirectML/CPU | CPU |

Licensing consequence: the QNN libraries are under Qualcomm's proprietary, **revocable** licence
(`licenses/assets.json`). Keep the CPU/GPU path fully functional so the product never depends on it.
Measured on this machine (Snapdragon X Plus, plate-row recogniser 48x256, batch 1; numbers in README):
NPU INT8 is fastest; GPU needs FP16/FP32 (INT8 QDQ is not supported by the QNN GPU backend); CPU
INT8 is the portable floor.

## 3. Data collection protocol (needed before ANY accuracy claim)

Standard: ISO/IEC 30107-3 presentation-attack metrics.
* **APCER**: share of attack presentations wrongly accepted, **reported per attack instrument**
  (never averaged away). **BPCER**: share of genuine presentations wrongly rejected.
* Report APCER at a fixed BPCER (e.g. 2% and 5%) with confidence intervals; state set sizes.

Bona fide (target >= 1,500 captures): real vehicles through the production capture flow on >= 6
phone models (include low-end devices common in India), day / dusk / night / rain, clean / dirty
plates, HSRP and old painted plates, 1-row and 2-row, EV green, yellow commercial, BH series.

Attacks (target >= 1,500 per instrument), each through the same capture flow:
* **Screens** (>= 8 displays): OLED phone, LCD phone, tablet, laptop, 24" monitor, TV; brightness
  25/60/100%; display shows a plate photo full-frame and also in a mock-up with surrounding vehicle.
  Include screens with anti-glare film and with the camera at 15-60 cm.
* **Prints**: laser on plain paper, inkjet on plain and glossy photo paper, newspaper-grade,
  laminated; flat and slightly curved; with and without retroreflective material behind.
* **Replay / re-submission** of a genuine earlier capture; **edited** captures (character edits,
  AI inpainting of digits) for the tamper detector.

Splits **by device and by plate identity** (hold out whole displays, whole phones and whole plates):
a random split leaks and will overstate accuracy. Keep a final **sealed test set** untouched until
the thresholds are frozen.

Privacy / DPDP Act 2023: plates and vehicles are personal data. Collect with consent, minimise
retention, blur faces/bystanders, restrict access, record purpose, and do not scrape public
datasets for plates of private owners.

## 4. What the web UI / `POST /api/extract` accepts today (no capture app yet)

Until the in-app camera exists the backend takes ordinary uploads, each labelled with a **role**
(`roles=plate,plate_flash,angle,angle,vin`). The roles decide which checks are meaningful:

| role | what to shoot | used for |
|---|---|---|
| `plate` | the plate, normal exposure, flash OFF, plate at least ~1/3 of the frame width, camera square-on. The first one is the reference frame. | reading; passive checks (metadata, pixel forensics, learned screen/print detector) |
| `plate_flash` | the **same pose**, torch/flash ON | retroreflection (flash) test; needs `hsrp=true` and a metered, non-saturated plate |
| `angle` | the same plate after stepping **20-40 cm sideways** (translation, not just turning the phone), once or twice | extra reading (glare moves between frames, so fused reads recover characters) and the parallax / planarity test |
| `vin` | close-up of the VIN sticker or stamping, square-on; tilt the phone slightly to walk glare off the characters | VIN reading only (short legacy chassis numbers allowed) |
| `auto` | anything else | looks for both a plate and a VIN; passive checks only |

Rules the server enforces: `plate_flash` and `angle` need a `plate` photo to compare against, at most one
`plate_flash`, at most 6 photos. Active liveness checks never run on `auto`/`vin` photos (they would compare
unrelated scenes).

The response carries `fields.plate` / `fields.vin` (each with `status`, `value`, `masked`, `alternatives`,
`reasons`, capture guidance) and `authenticity` (`status`, `liveness`, `positive`, `reasons`, `not_run`).
Read `liveness`, not just `status`: **`no_flags` is not liveness**. A well-made screen replay with no flash
frame raises no flag at all and ends as `no_flags` + `not_established`; pass `require_liveness=true` to make that
a review. `liveness=confirmed` needs a passed retroreflection test on an HSRP plate, or parallax **together with**
a clean learned screen/print score (depth alone is not enough: a screen held in front of a real background also
has depth). All thresholds behind these verdicts are provisional and have only been exercised on toy physics and
simulated attacks - validate them with the data collection in section 3 before relying on them.
