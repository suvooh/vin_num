# Vehicle category from the plate (colour + text pattern)

`vinplate/category.py` says what KIND of vehicle a plate belongs to: private, commercial, rental, electric (private /
commercial), temporary, trade / dealer, diplomatic, military, or "type not determined". It looks at the plate's
**colours** in the photo and the **text layout** the grammar recognised. It is attached to every plate reading as
`FieldReading.category` / `as_dict()["category"]` (`None` for a VIN, or when the crop is too small).

## Status of the rules: RECALLED, NOT VERIFIED

The colour -> category table comes from two infographics the user supplied (myfng.in "Types of number plates in
India" and autopunditz "India's vehicle number plate colours explained") and from what I recall of the CMVR
/ MoRTH notifications. **None of it has been checked against the official notifications**, state practice varies (some
states issue other colours), and the infographics themselves disagree in places (the diplomatic plate is white-on-blue in
one and white/yellow-on-blue in the other; the dealer mark `DLR 01 2345` is the only trade-plate text shown and it is
*illustrative*, so it is a low-prior grammar template, not a rule). Treat every row as a claim to check before it is
shown to a user as fact. The rules are DATA in one marked place: `COLOUR_RULES`, `LAYOUT_RULES`, `ALTERNATIVES` at the
top of `category.py`. Correct them there (and in the Dart port) and nothing else changes.

| background | text | category | source |
|---|---|---|---|
| white | black (any) | private | both infographics |
| yellow | black (any other) | commercial / transport | both |
| yellow | **red** | temporary registration | both |
| black (or dark grey) | yellow | rental / self-drive | both |
| green | white (any other) | electric, private | both |
| green | **yellow** | electric, commercial | both |
| red | white (any) | trade certificate / dealer | both (dealer text `DLR NN NNNN` on one) |
| blue | white (any) | diplomatic / consular | both; x0.7 confidence when no text was read. **Never when the text is an ordinary registration** (text-pattern guard below) |
| layout `diplomatic` (`NNN CD NN`) | - | diplomatic | decides by itself, normally printed on blue |
| layout `military` (`NNL NNNNNN L`) | - | military | decides by itself, normally dark with white text |
| layout `dealer` (`DLR NN NNNN`) | - | trade | decides by itself, normally on red |
| layout `bharat` (`NN BH NNNN AA`) | - | **a flag** (`bh: true`), not a category | white plate, black text |

Not covered, and the classifier cannot report: President / Governor plates (red with the national emblem, no text, so
no plate is read and no category is produced; the red background alone would read as "trade"), vintage / test /
Army-"arrow" variants other than the military layout, EV plates with a blue stripe, state-specific schemes.

## Method (deterministic, numpy only; the Dart port must reproduce every number)

1. **Sample** `step = max(1, int(sqrt(H*W/4096)))`, pixels `crop[::step, ::step]` as float64.
2. **2-means** (RGB, k = 2): background centre starts at the per-channel median (even count: mean of the two middle
   values), ink centre at the first pixel farthest from it; 8 iterations (assign by squared distance, a pixel joins the
   ink cluster only if STRICTLY nearer, ties -> background; then recompute both centres as the mean of the members, an
   empty cluster keeps its centre). The cluster with MORE members in the last assignment is the background (a tie keeps
   the initial background cluster). If the final centres are < 40 apart: one colour, `ink = None`, `bg` = mean of the
   sample.
3. **Name** each centre (`name_colour`): HSV with the hue formula in the code (ties for the maximum resolve r, g, b;
   hue wraps at 360). `v < 0.28` black; `s < 0.22`: white (`v >= 0.62`) / grey (`v >= 0.40`) / black; a pale cream
   (`14 <= h < 70`, `s < 0.30`, `v >= 0.62`) is white; else red (`h < 14` or `>= 340`), yellow (`< 70`), green (`< 195`),
   blue (`< 270`), other. Confidence `0.5 + 0.5 * clip(min(margins), 0, 1)` over the distances to every boundary that
   separates the name from its neighbours.
4. **Rules**: first matching `COLOUR_RULES` row (background, ink or any); a layout in `LAYOUT_RULES` overrides colour
   (confidence +0.1 if the background is one the layout is printed on, x0.6 and a reason if not).
5. **Confidence** `min(bg_conf, ink_conf or 0.7) * min(1, distance_between_centres / 120)` times the rule's factor,
   then the BH / layout adjustment. Below **0.75** up to two `alternatives` (one colour step away) are listed.
6. RGB ints are the centres rounded half up; confidence is rounded to 3 decimals.

Result: `{key, label, confidence, bg{name,rgb}, ink{name,rgb}|null, reasons[], alternatives[{key,label}], ev, bh}`.
`ev` is true for the two electric categories. The category is read from the biggest line crop of the plate field
(for fused multi-photo readings: the best-ranked photo's crop, with the FUSED text's layout).

## Measured (synthetic plates, `python -I -X utf8 eval/category_eval.py --n 40 --seed 1`)

The test plates come from the repo's renderer in the rule table's own colours (jittered +-12 per channel), so these
numbers prove the pipeline and its sensitivity to lighting, NOT that real plates look like this.

```
lighting         n  correct     acc  unknown  conf.wrong  right in alts
clean          440      440  100.0%        0           0              0
dim            440      329   74.8%       95           0              0
bright         440      431   98.0%        1           0              5
warm           440      420   95.5%        0           0             18
cool           440      398   90.5%        1           0              8
shade          440      425   96.6%        0           3             12
glare          440      416   94.5%        2           2              9
blur_noise     440      432   98.2%        0           0              8
ALL degraded  3080     2851   92.6%
```

## A real-photo bug and what it changed (2026-11-16)

The first on-phone test was a photo (of a monitor showing) a green-plate Toyota Camry, KA 01 MN 4120. The app said
"Probably diplomatic / consular vehicle". Two independent faults, both fixed, plus two UI guardrails:

1. **Colour naming.** The Indian EV plate is a TEAL, not a leaf green: measured hue 174-175 deg on the clean picture and
   184-190 deg (washed out, saturation 0.24-0.31) in the phone's photos of the monitor. The green / blue boundary sat at
   170 deg, so the plate was named **blue**. The boundary is now `GREEN_BLUE_HUE = 195` (a real diplomatic blue is about
   210-225 deg). All four measured samples now classify as `ev_private`. The synthetic renderer got two teal EV schemes
   (`ev_private_teal`, `ev_commercial_teal`) so this stays tested; the old leaf-green schemes stay too.
2. **Text-pattern guard.** The old rule let a blue plate with ordinary text still be "diplomatic" (only x0.7). A diplomatic
   plate always carries a CD / CC / UN code, so a plate whose text is an ordinary registration, a BH number, a defence
   number or a dealer mark can never be diplomatic, whatever colour was measured (`NOT_DIPLOMATIC_LAYOUTS`). Such a plate
   is reported as **unknown** (reason `R_DIP_FORMAT`) with the likelier answers as alternatives: `ev_private`, `private`
   (or `ev_commercial`, `ev_private` when the text colour is yellow). `diplomatic` is also dropped from the alternatives of
   every other category when the text is an ordinary registration. Blue with no text read stays diplomatic at low
   confidence. Invariant tested over every scheme and lighting: only a diplomatic / military layout yields those categories.
3. **UI:** a type the reader is not sure of is never drawn as the plate's colours (the plate had been drawn on a blue
   board), and a capture that looks like a photo of a screen or print always words the type as "Probably ...", explains
   why, and offers the picker (colours seen through a screen cannot be trusted). A type the person picks is drawn as theirs.

The measured plate colours are kept as golden cases (`real_ev_teal_*`, `real_ev_on_screen*`, `cool_white_plain`) so the
Dart port is checked against them; the photo itself is a third-party image and is not in the repository.

Weak spots (degraded runs; figures below are from the evaluation BEFORE the fix, except where stated): in the dim variant a white plate turns grey (`v < 0.62`) and is reported as "type not
determined" rather than guessed (78 of 80 white plates, 17 of 40 white-on-green EV plates), and a dim green plate with
yellow text reads as black-with-yellow, i.e. **rental** (14 of 40 commercial EV plates; `rental`'s listed alternative
is diplomatic, not the EV, so the right answer is NOT offered); cool light pushes some white plates into the blue band
(31 of 560 white / BH plates -> "diplomatic" before the fix; after it those become "type not determined" or a low-confidence
"electric", never diplomatic: 0 diplomatic false positives in 3640 degraded plates); warm light pushes some white plates into yellow (13 -> "commercial"); glare
and low contrast confuse rental with commercial / diplomatic / EV. Wrong answers with confidence >= 0.75: 5 of 3080
degraded plates. Real photographs are the missing evidence: real road colours (sheeting ageing, sodium lamps, dirt,
stickers across the plate) are not in this test.

On the 17 plate examples of the two infographics, cropped by hand (not shipped; third-party images): 17 / 17 classified
as drawn.
