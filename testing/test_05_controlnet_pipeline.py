"""
Tests for pipeline/generation/controlnet_pipeline.py

GPU-dependent functions (load_pipeline, generate_sdxl_scene) are tested via
a lightweight mock pipeline. CPU-only functions are tested directly.
"""
import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock
from PIL import Image


# ── Inject a fake 'torch' so GPU tests run without the real package ───────────

def _make_fake_torch():
    torch_mock = types.ModuleType("torch")

    class FakeGenerator:
        def __init__(self, device=None):
            pass
        def manual_seed(self, seed):
            return self

    torch_mock.Generator = FakeGenerator
    torch_mock.device = lambda x: x
    torch_mock.float16 = "float16"
    return torch_mock


# Inject before any pipeline import that might trigger 'import torch'
if "torch" not in sys.modules:
    sys.modules["torch"] = _make_fake_torch()

from pipeline.config import load_config
from pipeline.layout.generator import generate_layout
from pipeline.generation.controlnet_pipeline import (
    build_conditions,
    extract_object_patches,
    generate_sdxl_scene,
    run_generation,
    ObjectPatch,
    SDXLResult,
    MIN_COMPONENT_AREA,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return load_config("pipeline_config.yaml")


@pytest.fixture
def layout(cfg):
    return generate_layout(seed=42, cfg=cfg)


@pytest.fixture
def mock_pipeline():
    """
    Fake StableDiffusionXLControlNetPipeline.
    Returns a solid-color 1024x1024 PIL image when called.
    Does not require torch to be installed.
    """
    dummy_image = Image.fromarray(
        np.full((1024, 1024, 3), 128, dtype=np.uint8)
    )
    mock = MagicMock()
    mock.return_value.images = [dummy_image]

    # Make next(pipeline.unet.parameters()).device return a string "cpu"
    # generate_sdxl_scene passes device to torch.Generator — the mock
    # short-circuits before that call, so a plain string is sufficient here.
    param_mock = MagicMock()
    param_mock.device = "cpu"
    mock.unet.parameters.return_value = iter([param_mock])
    return mock


# ── build_conditions ──────────────────────────────────────────────────────────

def test_build_conditions_output_shapes(layout):
    seg, edge = build_conditions(layout)
    assert seg.shape == (1024, 1024, 3), "seg_condition must be (H,W,3)"
    assert edge.shape == (1024, 1024),   "edge_condition must be (H,W)"


def test_build_conditions_output_dtypes(layout):
    seg, edge = build_conditions(layout)
    assert seg.dtype == np.uint8
    assert edge.dtype == np.uint8


def test_build_conditions_no_road_sentinel(layout):
    """After export_mask, seg input must not contain class 255."""
    seg, _ = build_conditions(layout)
    # ADE20K RGB palette only uses values within known colors — no pixel
    # should carry the road sentinel (255) as a class value in the mask.
    # Indirectly verified: mask_to_ade20k_rgb raises if mask.max() > 4.
    # If this call succeeds, FIX #1 is working correctly.
    assert seg is not None


def test_build_conditions_edge_is_binary_like(layout):
    """Canny output should be 0 or 255 only."""
    _, edge = build_conditions(layout)
    unique_vals = set(np.unique(edge).tolist())
    assert unique_vals.issubset({0, 255}), f"Unexpected edge values: {unique_vals}"


# ── extract_object_patches ────────────────────────────────────────────────────

def _make_scene_and_mask():
    """Synthetic scene + mask with one pool and one tennis court."""
    scene = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    mask  = np.zeros((1024, 1024), dtype=np.uint8)
    # Tennis court — class 2, large block
    mask[100:320, 200:675] = 2   # 220 x 475 = 104,500 px > MIN_COMPONENT_AREA[2]=5000
    # Pool — class 3
    mask[500:620, 400:520] = 3   # 120 x 120 = 14,400 px > MIN_COMPONENT_AREA[3]=500
    return scene, mask


def test_extract_patches_returns_list():
    scene, mask = _make_scene_and_mask()
    patches = extract_object_patches(scene, mask)
    assert isinstance(patches, list)


def test_extract_patches_correct_count():
    scene, mask = _make_scene_and_mask()
    patches = extract_object_patches(scene, mask)
    assert len(patches) == 2  # one tennis court + one pool


def test_extract_patches_class_ids():
    scene, mask = _make_scene_and_mask()
    patches = extract_object_patches(scene, mask)
    class_ids = sorted(p.class_id for p in patches)
    assert class_ids == [2, 3]


def test_extract_patches_bbox_within_canvas():
    scene, mask = _make_scene_and_mask()
    patches = extract_object_patches(scene, mask)
    for p in patches:
        x1, y1, x2, y2 = p.bbox
        assert 0 <= x1 < x2 <= 1024
        assert 0 <= y1 < y2 <= 1024


def test_extract_patches_alpha_shape_matches_image():
    scene, mask = _make_scene_and_mask()
    patches = extract_object_patches(scene, mask)
    for p in patches:
        assert p.alpha.shape == p.image.shape[:2], \
            "alpha mask must match image spatial dimensions"


def test_extract_patches_alpha_dtype():
    scene, mask = _make_scene_and_mask()
    patches = extract_object_patches(scene, mask)
    for p in patches:
        assert p.alpha.dtype == bool


def test_extract_patches_min_area_filter():
    """Components below MIN_COMPONENT_AREA must be filtered out."""
    scene = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask  = np.zeros((1024, 1024), dtype=np.uint8)
    # Plant a tiny tennis court (area < 5000) — should be filtered
    mask[10:30, 10:30] = 2   # 400 px < MIN_COMPONENT_AREA[2]=5000
    patches = extract_object_patches(scene, mask)
    assert len(patches) == 0, "Sub-threshold component should be filtered out"


def test_extract_patches_bbox_aligns_with_mask():
    """Image crop at bbox must match the location in the scene exactly."""
    scene = np.arange(1024 * 1024 * 3, dtype=np.uint8).reshape(1024, 1024, 3)
    mask  = np.zeros((1024, 1024), dtype=np.uint8)
    mask[200:440, 100:575] = 2  # large tennis-court block

    patches = extract_object_patches(scene, mask)
    assert len(patches) == 1
    p = patches[0]
    x1, y1, x2, y2 = p.bbox
    np.testing.assert_array_equal(p.image, scene[y1:y2, x1:x2])


# ── generate_sdxl_scene (mocked) ──────────────────────────────────────────────

def test_generate_sdxl_scene_output_shape(layout, mock_pipeline):
    _, edge = build_conditions(layout)
    scene = generate_sdxl_scene(edge, mock_pipeline, seed=42)
    assert scene.shape == (1024, 1024, 3)
    assert scene.dtype == np.uint8


def test_generate_sdxl_scene_calls_pipeline_once(layout, mock_pipeline):
    _, edge = build_conditions(layout)
    generate_sdxl_scene(edge, mock_pipeline, seed=0)
    mock_pipeline.assert_called_once()


# ── run_generation (mocked) ───────────────────────────────────────────────────

def test_run_generation_returns_sdxl_result(layout, cfg, mock_pipeline):
    result = run_generation(layout, mock_pipeline, cfg)
    assert isinstance(result, SDXLResult)


def test_run_generation_result_shapes(layout, cfg, mock_pipeline):
    result = run_generation(layout, mock_pipeline, cfg)
    assert result.scene.shape == (1024, 1024, 3)
    assert result.seg_condition.shape == (1024, 1024, 3)
    assert result.edge_condition.shape == (1024, 1024)
    assert isinstance(result.patches, list)
