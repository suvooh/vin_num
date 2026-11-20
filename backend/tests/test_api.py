"""API behaviour. The models are replaced by a fake reader (real decoder, simulated posteriors) so
these run in a second; the last test drives the REAL engine end to end and skips if the ONNX models
are not present."""

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app import main
from app.service import field_out
from vinplate.authenticity import Authenticator
from vinplate.decode import Reading, read_line
from vinplate.grammar import PLATE_GRAMMAR, VIN_GRAMMAR, parse_plate, parse_vin
from vinplate.runtime import REPO_ROOT, find_model
from vinplate.scene import FieldReading, SceneResult
from vinplate.testing import make_logp

client = TestClient(main.app)


def _png(size=(160, 120), color="white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def _plate_field(text="MH12AB1234", **posterior) -> FieldReading:
    r = read_line(make_logp(text, **posterior), PLATE_GRAMMAR)
    return FieldReading("plate", r, parse_plate(r.text), (10, 10, 100, 40), [[[10, 10], [100, 10], [100, 40], [10, 40]]],
                        evidence=sum(c != "?" for c in r.text))


def _vin_field(text="MAL907KE1E6602791") -> FieldReading:
    r = read_line(make_logp(text), VIN_GRAMMAR)
    return FieldReading("vin", r, parse_vin(r.text), (5, 5, 150, 30), [[[5, 5], [150, 5], [150, 30], [5, 30]]],
                        evidence=sum(c != "?" for c in r.text))


class FakeReader:
    def __init__(self, plate=None, vin=None):
        self.plate, self.vin, self.calls = plate, vin, []

    def read_frames(self, rgbs, *, domains=("plate", "vin"), allow_legacy_vin=False):
        self.calls.append((len(rgbs), tuple(domains), allow_legacy_vin))
        return SceneResult(plate=self.plate if "plate" in domains else None,
                           vin=self.vin if "vin" in domains else None, size=(160, 120))


@pytest.fixture
def fake(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CAPTURES_DIR", tmp_path)
    rd = FakeReader(_plate_field(), _vin_field())
    monkeypatch.setattr(main, "get_engine", lambda *a, **k: SimpleNamespace(reader=rd, authenticator=Authenticator()))
    return rd


def _post(files, **data):
    return client.post("/api/extract", files=[("files", (f"{i}.png", b, "image/png")) for i, b in enumerate(files)], data=data)


# ---------------------------------------------------------------------------- validation
def test_health_shape():
    body = client.get("/api/health").json()
    assert "ocr" in body and "ok" in body


def test_rejects_non_image(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CAPTURES_DIR", tmp_path)
    r = client.post("/api/extract", files=[("files", ("a.txt", b"hi", "text/plain"))])
    assert r.status_code == 415


def test_rejects_undecodable_image(fake):
    r = client.post("/api/extract", files=[("files", ("a.png", b"not really a png", "image/png"))])
    assert r.status_code == 422


def test_rejects_too_many_files(fake):
    assert _post([_png()] * (main.MAX_FILES + 1)).status_code == 400


@pytest.mark.parametrize("roles,why", [
    ("plate", "roles"),                        # 2 files, 1 role
    ("plate,selfie", "unknown role"),
    ("angle,vin", "need a 'plate'"),
    ("plate_flash,vin", "need a 'plate'"),
    ("plate,plate_flash,plate_flash", "at most one"),
])
def test_role_validation(fake, roles, why):
    n = 3 if roles.count(",") == 2 else 2
    r = _post([_png()] * n, roles=roles)
    assert r.status_code == 400 and why in r.json()["detail"], r.json()


# ------------------------------------------------------------------------------ response
def test_response_shape_and_routing(fake):
    r = _post([_png(), _png()], roles="plate,vin")
    assert r.status_code == 200, r.text
    body = r.json()
    plate, vin = body["fields"]["plate"], body["fields"]["vin"]
    assert plate["value"] == "MH12AB1234" and plate["display"] == "MH 12 AB 1234" and plate["status"] == "accept"
    assert plate["image"] == 0 and vin["image"] == 1 and vin["domain"] == "vin"
    assert vin["check_digit_ok"] in (True, False, None) and vin["read_length"] == 17
    assert body["authenticity"]["liveness"] == "not_established"           # no flash / angle frames: never "confirmed"
    assert body["authenticity"]["status"] in ("no_flags", "review", "reject")
    assert [im["role"] for im in body["images"]] == ["plate", "vin"]
    # a vin photo is read for a VIN only, with legacy chassis numbers allowed; the plate photo for a plate only
    assert (1, ("plate",), False) in fake.calls and (1, ("vin",), True) in fake.calls
    assert client.get(f"/api/captures/{body['capture_id']}/0").status_code == 200


def test_auto_role_looks_for_both(fake):
    body = _post([_png()]).json()
    assert fake.calls == [(1, ("plate", "vin"), False)]
    assert body["fields"]["plate"]["value"] == "MH12AB1234" and body["fields"]["vin"]["domain"] == "vin"


def test_nothing_found_is_null_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CAPTURES_DIR", tmp_path)
    monkeypatch.setattr(main, "get_engine", lambda *a, **k: SimpleNamespace(reader=FakeReader(), authenticator=Authenticator()))
    body = _post([_png()]).json()
    assert body["fields"] == {"plate": None, "vin": None}


def test_review_reading_exposes_masked_text_and_flags_review():
    f = _plate_field("MH12AB1234", unreadable=(5,))                          # one character is unreadable
    out = field_out(f, 0)
    assert out["status"] == "review" and out["needs_review"]
    assert "?" in out["masked"]                       # the unreadable character is shown as unknown, not guessed
    assert out["calibrated"] is False


def test_rejected_reading_has_no_value():
    r = Reading("MH12", False, "reject", 0.0, 0.0, [], [], [], 99.0, "MH12")
    out = field_out(FieldReading("plate", r, None, (0, 0, 1, 1), []), 0)
    assert out["value"] is None and out["display"] is None


def test_flash_and_angle_frames_run_the_active_checks(fake):
    rng = np.random.default_rng(0)

    def photo(seed):
        buf = io.BytesIO()
        Image.fromarray((np.random.default_rng(seed).random((240, 320, 3)) * 255).astype(np.uint8)).save(buf, "PNG")
        return buf.getvalue()

    r = _post([photo(1), photo(2), photo(3)], roles="plate,plate_flash,angle", hsrp="true")
    assert r.status_code == 200, r.text
    a = r.json()["authenticity"]
    assert [f["role"] for f in a["per_frame"]] == ["primary", "flash_on", "angle"]
    assert "retroreflection" in a["evidence"]["not_run_reasons"] or "retroreflection" in a["evidence"]["active"]


# --------------------------------------------------------------------- the real engine
def _models_present() -> bool:
    try:
        find_model("ocr_det.onnx"), find_model("ocr_rec.onnx"), find_model("ocr_charset.txt")
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _models_present(), reason="PP-OCR ONNX models not found (app/assets/models)")
def test_real_engine_reads_a_rendered_plate(tmp_path, monkeypatch):
    fonts = sorted((REPO_ROOT / "ml" / "synth" / "fonts" / "barlowcondensed").glob("*Bold.ttf"))
    if not fonts:
        pytest.skip("no font available to render a test plate")
    monkeypatch.setattr(main, "CAPTURES_DIR", tmp_path)
    img = Image.new("RGB", (900, 560), (110, 114, 120))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((200, 220, 700, 340), 8, fill=(240, 240, 236), outline=(20, 20, 20), width=4)
    d.text((450, 280), "MH 12 AB 1234", font=ImageFont.truetype(str(fonts[0]), 78), fill=(15, 15, 15), anchor="mm")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    with TestClient(main.app) as c:                                          # runs the lifespan: loads the real models
        r = c.post("/api/extract", files=[("files", ("plate.jpg", buf.getvalue(), "image/jpeg"))], data={"roles": "plate"})
    assert r.status_code == 200, r.text
    plate = r.json()["fields"]["plate"]
    assert plate is not None, r.json()["lines"]
    assert plate["value"] == "MH12AB1234" and plate["status"] == "accept", plate   # a clean rendering must be read
