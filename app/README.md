# VinPlate Scanner (Flutter, Android) - completely offline

Reads Indian number plates and VINs, and judges whether a real car was photographed, **entirely on
the phone**. The models (PP-OCR detector + recogniser, the screen / print detector) and the ONNX
Runtime are inside the APK; the app never touches the network (the **release** build requests no internet permission at all; Flutter's debug build adds it for its tooling) and there is no server. The Python
service in `../backend` is development tooling (benchmarks, reference implementation) and is not
used by the app.

```
lib/
  main.dart               entry, theme, services
  app_services.dart       settings + history + the Scanner, injected (tests substitute fakes)
  data/                   models (typed result), scanner (interface), settings, history
  offline/                engine_host (persistent worker isolate that keeps the models loaded),
                          offline_engine (the pipeline: detect -> grammar decode -> authenticity)
  reader/                 Dart ports of ml/vinplate: decoder + grammars, geometry, photometric views,
                          quality gate, PP-OCR engine, scene reader, planarity, liveness, plate category
  ui/                     home, capture wizard, result, sheets (review / authenticity / photo viewer),
                          plate_types (vehicle-type colours, chip, legend / picker), settings sheet,
                          onboarding, dialogs, wording (explain.dart)
```

The old research harness (with a gallery picker) is kept out of the build in `archive/`; the product has **no
gallery or file upload anywhere**, by design (a saved or edited image is a fraud route).

## The flow

1. **Home** - the offline reader's state (models load in the background), three scan actions, recent scans.
2. **Capture** - live camera only. A plate scan is always three photos: *Front* (a flash-off frame, then the torch
   is turned on for you and a second frame is taken from the same spot), *Left* and *Right* (step to each side).
   A VIN scan is one close photo. A review sheet comes before anything is read.
3. **Analysing** - a cancellable progress dialog (indeterminate on purpose: the engine does not report progress).
4. **Result** - plate and VIN as masked text: a character the photo could not support is a red **?**, or, when the plate /
   VIN format and the recogniser's evidence point to one legal character, an amber *suggestion* (never a reading).
   The plate is drawn on a board in its own colours with a **vehicle type** chip (private, commercial, electric, rental,
   temporary, trade, diplomatic, military; "Probably ..." when unsure; tap to choose another). Anything not
   "read with confidence" must be confirmed by typing what is on the vehicle (never pre-filled with the
   decoder's guess). *Is this a real car?* explains what was and was not checked. Settings > *Plate types guide*
   lists every type and its colours.
5. **History** - finished scans (with photos and the vehicle type) stay on the phone if History is on; swipe to delete.

## Run it on a phone (USB or wireless debugging)

`.\run_on_device.ps1` (see its header for wireless pairing). No backend, tunnel or Wi-Fi connection to the computer is
needed. Build for **every** device with a multi-ABI build (the ONNX Runtime ships arm64-v8a, armeabi-v7a, x86 and x86_64):
`flutter build apk --release --split-per-abi` (one APK per ABI) or `flutter build apk --release` (one fat APK).

## Test

```
flutter analyze
flutter test        # data/logic, screens, the decoder and image ports against Python golden data
```

The ONNX models need the phone's native libraries, so `flutter test` cannot run the real detector / recogniser; those
calls are only exercised on a device. See the status section of the repository README for what is verified where.
