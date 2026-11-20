# vin_num

Offline VIN and Indian number-plate reader with damage-aware decoding and screen / print
("is this a real car?") checks.

```
vin_num/
  backend/    FastAPI service (app/): upload -> readings + authenticity / liveness verdict
  frontend/   Static UI (index.html), served by the backend at /
  ml/         `vinplate` package: OCR engine, grammar decoder, authenticity checks, synthetic
              data, training code, benchmark (see ml/README.md for the full status table)
  app/        Flutter test harness for the flash / screen checks on a real phone
```

## What changed from the first version, and why it failed

The first backend ran a generic OCR (RapidOCR) and pulled the plate / VIN out of the **text** with regexes.
That cannot work reliably:

* Once the OCR has turned a scratched or glared character into a confident wrong letter, no regex can tell.
  The OCR's probabilities were thrown away, so damage was invisible to everything downstream.
* A VIN is 17 characters from a 33-letter alphabet. Label text ("TYRE PRESSURE 32 PSI" is exactly 17 legal VIN
  characters) and engine numbers look like VINs to a regex.
* Nothing was done about glare, blur, tilt, 2-row plates or plates the detector split in pieces.
* There was no check at all that the photo was of a real car, and the OCR engine did not even install on this
  ARM64 machine (no OpenCV wheel), so the service could not start.

## How it works now

```
photo -> text detector (PP-OCR DB) -> deskew each line (min-area rectangle + perspective crop)
      -> recogniser, KEEPING its per-time-step probabilities, projected onto A-Z 0-9 and "?"
      -> candidates: single lines, 2-row pairs, plates the detector split, whole plate regions
      -> several photometric views of each crop fused (stretch / sharpen / best channel / glare filled)
      -> grammar-constrained CTC decode (Indian plate layouts + state codes, ISO 3779 VIN)
      -> safety policy -> accept | review | reject, with the characters it is NOT sure of shown as "?"
several photos -> readings fused on the probabilities; disagreements are reported, never averaged away
```

### Scratches, missing characters, damage ("reconstruction")

* A character the pixels do not support comes out as `?` - PP-OCR emits CJK / punctuation mass over a mangled glyph,
  and that mass is mapped to an explicit "unreadable" class. `masked` shows `MH12?B1234`; it never shows the decoder's guess
  as if it had been read. The UI paints every `?` red.
* The grammar resolves **ambiguity**, not missing information: a digit slot cannot be `O`, a state code must exist,
  a series letter cannot be `?`-shaped nonsense. A single unreadable VIN character is filled only where a *mandatory*
  ISO check digit pins it down - and is still flagged as inferred.
* Look-alike glyphs (O/Q/0/D, I/1/L, 5/S, 8/B, 2/Z, 6/G) that the grammar would also allow in that position, with real
  evidence for the alternative, are shown as unknown instead of read. A modern plate with no series letters, or a short
  number, is never auto-accepted: it is either a very old plate or a *dropped* character, and a CTC decoder cannot tell which.
* VIN specifics: I, O and Q cannot occur in a VIN, so they are folded to 1 / 0 / 0; a read made of sticker words is rejected;
  short legacy chassis numbers are only read for `vin` photos and are never auto-accepted.
* Clutter around the text (the `IND` hologram, a `TC/20` temporary mark, a `VIN:` label) is handled by decoding sub-windows.

**What this cannot do:** recover characters that are not in the picture. A generative "reconstruction" would invent
identifiers in an identity-verification product, so none exists. A character that is gone stays `?` and the system
asks for a better photo (capture guidance is returned: move closer, tilt to move glare, hold steady, ...).

### Glare and lighting

Per crop the reader also reads: a contrast-stretched luminance view (veiling glare, haze), a sharpened view (blur),
the best single colour channel (yellow / green plates, tinted flash) and a view with clipped hot-spots filled in from their
surroundings. The posteriors of the views are fused, so a character that is clear in any view survives. Pixels clipped to pure
white are gone for good - that is what the multi-photo capture below is for (glare moves when the camera moves).

### Fraud: photo of a screen, print or reused image

Run on every capture; what each one is worth is stated in the output:

| check | needs | notes |
|---|---|---|
| provenance: EXIF, XMP / IPTC "digital source", AI-tool and editor markers, C2PA | the original file | a declared screenshot / AI origin is strong evidence; **absent metadata means nothing** (messengers strip it) |
| learned screen / print detector (PadNet on native-resolution patches) | `app/assets/models/pad_int8.onnx` | trained on *simulated* attacks only |
| pixel forensics (moire, banding, noise / ELA consistency) | any photo | weakly calibrated; raises doubt, never clears an image |
| **flash retroreflection** | a `plate_flash` photo, `hsrp=true`, plate not saturated | HSRP sheeting brightens far more than its surround under flash; a screen emits its own light, a print reflects like its surroundings |
| **parallax / planarity** | an `angle` photo taken after stepping 20-40 cm sideways | a screen or print is one plane, a car is not |
| consistency | several plate photos | the text must agree between photos |

The verdict has two separate fields. `status` (`no_flags` / `review` / `reject`) says whether red flags were raised;
`liveness` (`confirmed` / `suspected_spoof` / `not_established`) says whether there is *positive* physical evidence of a real
scene. **`no_flags` is not liveness**: a good replay with no flash photo raises no flag. `confirmed` needs exactly one thing: a
passed **flash retroreflection** test on an HSRP plate. Parallax is supporting evidence only (a screen held in front of a real
background also shows depth; a real monitor photo defeated the earlier "parallax + clean score" rule, which was removed), and a
clean screen/print score is never proof. Photos that move as one flat plane add a small suspicion (+0.5), never a confirmation.
Pass `require_liveness=true` to turn "not established" into a review.

### Vehicle type from the plate (private / commercial / electric / ...)

`ml/vinplate/category.py` (Python reference) and `app/lib/reader/plate_category.dart` (the on-device port, checked against
`app/test/fixtures/golden/category_cases.json`) work out what kind of vehicle a plate belongs to from two independent cues:
the **plate colours** (background and text colour of the plate crop, by a small deterministic 2-means + colour-name step) and the
**text pattern** (a diplomatic code `NNN CD|CC|UN NN`, the defence format `NNL NNNNNN L` - the up-arrow itself is not read -, a `BH`
series, a dealer-style `DLR NN NNNN` that is only an illustrative format). Colour and format are checked against each other: a
format that normally sits on a different colour lowers the confidence and says so in `reasons`. The result is a `category` on the plate reading: `key`, `label`, `confidence`, the measured background / text colour,
`reasons`, `alternatives`, and `ev` / `bh` flags.

| plate | type |
|---|---|
| black text on white | private |
| black on yellow | commercial / transport |
| yellow on black | rental / self-drive |
| white on green / yellow on green | electric, private / commercial |
| red on yellow | temporary registration |
| white on red | trade certificate / dealer |
| blue **and** a CD / CC / UN code | diplomatic / consular (a blue-looking plate with an ordinary registration is never called diplomatic) |
| dark plate, format `NNL NNNNNN L` | military / defence |

What this is and is not:

* The rules are **recalled from two public infographics and have not been checked against the official notification**; state
  practice varies. They are a table at the top of `category.py` (mirrored in `app/lib/ui/plate_types.dart`) - correct the table, nothing else.
* A real-phone test caught the first version calling a green-plate EV "diplomatic" (the plate's teal colour was named blue). Fixed in two ways: the
  green / blue boundary was measured and moved, and a **text-pattern guard** now stops an ordinary registration from ever being
  diplomatic. A capture that looks like a photo of a screen always words the type as a guess (screens change colours).
* The colour comes from a photo, so lighting, a tinted flash, glare and a dirty plate move it. The app therefore never states a
  type it is unsure of as fact: below ~60 % confidence it says "Probably ...", an undetermined colour says "Type not determined", and the person
  can always pick the type from a list (their pick is stored with the scan and marked as theirs).
* The type is shown, saved in History and copied with the plate. It is **not** used to accept or reject a reading, and it
  is not an identity or legality check.
* It has been checked on **synthetic and hand-drawn plates only** (no real photographed plates of each type exist in this repository):
  100 % on clean synthetic plates of all 11 schemes, 92.6 % over 7 lighting / noise variants (dim light turns white plates grey, so it
  abstains; cool light pushes white toward blue, warm toward yellow), 17 / 17 of the plates drawn in the two infographics, and 48 / 49
  through the real reader on the benchmark scenes (a private-heavy mix, so EV and rental are barely tested). Those numbers prove the
  colour pipeline, not that real plates look like the table. Real accuracy is unmeasured. Details: `ml/docs/VEHICLE_CATEGORY.md`.

## Run (Windows PowerShell)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt      # installs ../ml as the editable `vinplate` package
.\run_backend.ps1                         # API + UI on http://127.0.0.1:8002
```

Open http://127.0.0.1:8002, add photos, say what each one is (plate / plate + flash / other angle / VIN) and press
**Read VIN & plate**. A good capture: the plate square-on and at least a third of the frame wide, then the same pose with the
torch on, then one or two photos after stepping sideways; for a VIN, a close-up with the phone tilted so glare leaves the characters.
Allow a few seconds per photo on CPU.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | engine and model status |
| POST | `/api/extract` | multipart `files` (1-6 JPEG/PNG/WebP, 15 MB each); form `roles` (comma list: `auto`, `plate`, `plate_flash`, `angle`, `vin`), `hsrp`, `require_liveness` |
| GET | `/api/captures/{id}/{index}` | the stored upload |

`fields.plate` / `fields.vin` are `null` or carry `status` (`accept` / `review`), `value`, `masked`, `display`, `confidence`,
per-character confidence, `inferred` positions, `alternatives`, `reasons`, `parsed` (state, WMI, check digit, ...), the box it was
read from, the photo it came from and capture guidance. **Branch on `status`, and show `masked` unless it is `accept`.** The
response also has `authenticity` (`status`, `liveness`, `positive`, `reasons`, `not_run`) and `conflicts`. Roles are explained in
`ml/docs/CAPTURE_PROTOCOL.md` section 4. Uploads are stored under `backend/data/captures/` - plates and vehicles are personal data.

## Tests and benchmark

```powershell
cd backend; pytest          # API behaviour (fake reader) + one real-engine end-to-end read
cd ..\ml;   pytest          # grammar, decoder, engine, scene reader, enhancement, authenticity, bundle, benchmark harness
cd ml; .venv\Scripts\python -I -X utf8 eval\e2e_eval.py --n 60 --seed 3 --readers old,new --json out.json
```

`eval/e2e_eval.py` pastes degraded synthetic Indian plates / VINs onto real car photos with exact ground truth and compares the
old pipeline (same models, greedy text + regex) with the new one per damage slice. Its key metric is **accepted-wrong**: the share of
fully visible plates the reader *accepted* but got wrong.

## What is and is not known (read this before trusting a number)

**Measured so far (synthetic Indian plates / VINs pasted on real car photos, 60 scenes, old pipeline vs new on the same models):**

| | old (greedy text + regex) | new |
|---|---|---|
| exact reads, fully visible plates/VINs | 78.9% | 81.6% |
| **accepted but wrong** (95% interval) | 13.2% (5.8-27.3) | 5.3% (1.5-17.3) |
| VIN accepted-but-wrong | 18.8% | 0.0% (0-19.4) |
| accepted / flagged for review / nothing found | 73 / 0 / 27% | 53 / 23 / 23% |
| time per photo (CPU) | ~2 s | ~3 s |

How to read it honestly:

* The gain is **safety, not reading skill**. Exact reads are about the same (81.6% vs 78.9%: within noise on 60 scenes). The new
  pipeline turns confident wrong answers into "review" and `?`, and auto-accepts fewer scenes (53% vs 73%).
* The intervals overlap: 60 scenes cannot prove the difference. Two `Q`->`O` series-letter misreads were still accepted in this run;
  a rule that sends plates with an `I`/`O`/`Q` series letter to review was added afterwards and checked on those two scenes only.
* **Glare is not shown to be solved.** In the 20 glare scenes the new reader accepted none, flagged 45% for review and found nothing
  in 55%. It is safer, not better at reading through glare. The multi-view and multi-photo fusion are the intended remedy, but this
  benchmark has too few glare cases to show a benefit from them, and flash-glare is best fixed at capture (move, tilt, a second photo).
* A paired ablation (40 scenes) found that a contrast-stretched view gives the same gain as all five photometric views, so only
  `stretch` (+ `deglare` for blown-out highlights, whose effect is unmeasured) is on by default.


* **No real Indian plate / VIN photo set exists in this repository**, and none is public (see `ml/README.md`). Everything above is
  synthetic or simulated. The accuracy claim that matters can only come from your own labelled captures (two-wheelers, EV plates,
  BH series, painted plates, every VIN type), collected under DPDP.
* The flash and parallax checks have only met a toy physics model; the screen / print detector only simulated attacks. Cross-device
  generalisation is the known weak point. Collect real replay attacks per `ml/docs/CAPTURE_PROTOCOL.md` before relying on any verdict.
* The generic PP-OCR recogniser is the reader today. The purpose-built line recogniser in `ml/training` is still untrained (needs a
  rented GPU and real data); it is the route to a real accuracy step and to removing the Apache-2.0 PP-OCR dependency.
* All thresholds are provisional; outputs carry `calibrated: false`.
