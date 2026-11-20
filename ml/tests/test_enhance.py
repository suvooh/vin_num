"""Photometric views: size-preserving, and they do what they claim on constructed images."""

import numpy as np
import pytest

from vinplate import enhance as E


def _plate(h=48, w=240, veil=0.0, text_val=20):
    """Dark strokes on a light plate, optionally under an additive veil (mild glare)."""
    img = np.full((h, w, 3), 225, np.float32)
    for x in range(20, w - 20, 26):
        img[8:h - 8, x:x + 6] = text_val
    img = img + veil
    return np.clip(img, 0, 255).astype(np.uint8)


def test_every_variant_keeps_the_crop_size_and_dtype():
    img = _plate()
    for name, v in E.variants(img):
        assert v.shape == img.shape and v.dtype == np.uint8, name
    assert E.variants(img)[0][0] == "orig"


def test_stretch_restores_contrast_under_a_veil():
    veiled = _plate(veil=-0)   # strokes at 20
    veiled = np.clip(veiled.astype(np.float32) * 0.4 + 140, 0, 255).astype(np.uint8)    # contrast crushed
    before = np.subtract(*np.percentile(E.luminance(veiled), [98, 2]))
    after = np.subtract(*np.percentile(E.luminance(E.stretch(veiled)), [98, 2]))
    assert after > 2.0 * before and after > 200


def test_stretch_does_not_amplify_a_flat_crop():
    flat = np.full((48, 120, 3), 130, np.uint8)
    out = E.stretch(flat)
    assert np.ptp(out.astype(int)) <= 1


def test_deglare_needs_a_clipped_region():
    assert E.deglare(_plate()) is None


def test_deglare_fills_a_blown_out_blob_from_its_surroundings():
    img = np.full((48, 240, 3), 150, np.uint8)     # mid-grey plate with dark strokes
    for x in range(20, 220, 26):
        img[8:40, x:x + 6] = 20
    img[10:38, 90:135] = 255                       # specular blob over the strokes at x=98 and x=124
    out = E.deglare(img)
    assert out is not None and out.shape == img.shape
    # in the input the blob is 105 levels brighter than the plate; after filling, the gap between
    # strokes inside the blob must look like the plate again (the hidden strokes stay hidden)
    assert abs(int(out[24, 112, 0]) - int(out[24, 60, 0])) < 25
    assert abs(int(img[24, 112, 0]) - int(img[24, 60, 0])) > 90


def test_deglare_ignores_a_fully_overexposed_plate():
    assert E.deglare(np.full((48, 200, 3), 255, np.uint8)) is None


def test_clip_mask_ignores_isolated_noise_pixels():
    img = np.full((40, 100, 3), 200, np.uint8)
    img[5, 5] = 255
    img[20, 40] = 255
    assert not E.clip_mask(img).any()
    img[10:20, 10:30] = 255
    assert E.clip_mask(img).sum() > 100


def test_best_channel_picks_the_channel_that_carries_the_strokes():
    img = np.zeros((48, 160, 3), np.uint8)
    img[..., 0] = 250                                # yellow plate: red and green bright, blue low
    img[..., 1] = 205
    img[..., 2] = 25
    for x in range(10, 150, 24):
        img[8:40, x:x + 5] = 0                       # black strokes
    out = E.best_channel(img)[..., 0].astype(int)
    assert np.ptp(out) > 200


def test_sharpen_increases_edge_energy_of_a_blurred_crop():
    from scipy import ndimage as ndi
    sharp = _plate()
    blurred = np.clip(ndi.gaussian_filter(sharp.astype(np.float32), (1.6, 1.6, 0)), 0, 255).astype(np.uint8)
    e = lambda a: float(np.abs(np.diff(E.luminance(a), axis=1)).mean())
    assert e(E.sharpen(blurred)) > e(blurred)


@pytest.mark.parametrize("fn", [E.stretch, E.retinex, E.sharpen, E.best_channel])
def test_variants_are_deterministic(fn):
    img = _plate(veil=15)
    assert np.array_equal(fn(img), fn(img))
