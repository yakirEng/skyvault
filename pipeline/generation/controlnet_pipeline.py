import cv2
import numpy as np
from dataclasses import dataclass

# torch and diffusers are GPU-only deps — imported lazily inside the functions
# that need them so the module can be imported on CPU-only machines for testing.
from PIL import Image

from pipeline.config import PipelineConfig
from pipeline.layout.generator import LayoutResult, export_mask
from pipeline.layout.palette import mask_to_ade20k_rgb, mask_to_canny_edges


POSITIVE_PROMPT = (
    "nadir aerial orthophoto, straight-down view, remote sensing imagery, "
    "bare ground surface, arid dry soil, dusty asphalt road, "
    "overhead geospatial photograph, flat terrain, open land, "
    "photogrammetry map, high resolution orthoimage"
)

NEGATIVE_PROMPT = (
    "architecture, building facade, wall, tiles, brick, roof, window, "
    "3D render, shadows, depth, perspective, isometric, oblique, "
    "solar panels, interior, indoors, closeup, blurry, low resolution, "
    "painting, illustration, cartoon, snow, vegetation canopy"
)

MIN_COMPONENT_AREA: dict[int, int] = {
    1: 200,    # crosswalk — small
    2: 5000,   # tennis court — large
    3: 500,    # pool — medium
    4: 1000,   # eucalyptus — medium
}


@dataclass
class ObjectPatch:
    image: np.ndarray            # (h, w, 3) uint8 — RGB crop from SDXL output
    alpha: np.ndarray            # (h, w) bool — pixel-level object mask
    class_id: int                # 1-4
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) in layout pixel coordinates


@dataclass
class SDXLResult:
    scene: np.ndarray            # (1024, 1024, 3) uint8 — full SDXL output
    patches: list[ObjectPatch]   # one per connected object instance
    seg_condition: np.ndarray    # (1024, 1024, 3) uint8 — ADE20K RGB (input to SDXL)
    edge_condition: np.ndarray   # (1024, 1024) uint8 — Canny edges (input to SDXL)


def build_conditions(layout_result: LayoutResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare ControlNet conditioning images from a LayoutResult.

    Returns: (seg_condition (H,W,3), edge_condition (H,W))
    """
    normalized_mask = export_mask(layout_result.canvas)
    seg_condition = mask_to_ade20k_rgb(normalized_mask)
    edge_condition = mask_to_canny_edges(normalized_mask)
    return seg_condition, edge_condition


def generate_sdxl_scene(
    edge_condition: np.ndarray,
    pipeline,
    seed: int,
    conditioning_scale: float = 0.8,
) -> np.ndarray:
    """
    Run SDXL inference conditioned on Canny edges.
    Uses diffusers/controlnet-canny-sdxl-1.0 (single ControlNet).
    Returns the generated image as (1024, 1024, 3) uint8 numpy array.

    Note: there is no publicly available segmentation ControlNet for SDXL.
    The seg_condition is computed for visualization purposes only.
    """
    import torch  # lazy — only needed on GPU

    # edge_condition is grayscale — diffusers expects 3-channel PIL
    edge_rgb = np.stack([edge_condition] * 3, axis=-1)
    edge_pil = Image.fromarray(edge_rgb)

    # Use CPU generator — compatible with sequential_cpu_offload which places
    # parameters on a meta device, making pipeline.unet.parameters().device unusable.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = pipeline(
        prompt=POSITIVE_PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=edge_pil,
        controlnet_conditioning_scale=conditioning_scale,
        num_inference_steps=50,
        generator=generator,
        width=1024,
        height=1024,
    )

    scene = np.array(result.images[0])  # PIL -> (1024, 1024, 3) uint8
    return scene


def extract_object_patches(
    scene: np.ndarray,
    normalized_mask: np.ndarray,
) -> list[ObjectPatch]:
    """
    FIX #2: Extract per-instance object patches from SDXL output using
    connected-component analysis on the layout mask.

    Patches are extracted at the SAME pixel coordinates as the layout mask,
    guaranteeing alignment with the compositor (Fix #5).
    """
    patches = []

    for class_id in [1, 2, 3, 4]:
        binary = (normalized_mask == class_id).astype(np.uint8)
        n_labels, labels = cv2.connectedComponents(binary)

        for label_id in range(1, n_labels):
            component = labels == label_id
            area = int(component.sum())

            if area < MIN_COMPONENT_AREA[class_id]:
                continue

            rows = np.where(component.any(axis=1))[0]
            cols = np.where(component.any(axis=0))[0]
            y1, y2 = int(rows[0]), int(rows[-1]) + 1
            x1, x2 = int(cols[0]), int(cols[-1]) + 1

            image = scene[y1:y2, x1:x2].copy()
            alpha = component[y1:y2, x1:x2]
            patches.append(ObjectPatch(image, alpha, class_id, (x1, y1, x2, y2)))

    return patches


def load_pipeline(device: str = "cuda"):
    """
    Load SDXL + Canny ControlNet pipeline, optimised for Colab free tier (T4, ~15GB RAM).

    Key memory decisions:
    - variant="fp16": loads fp16 weights directly, avoids fp32→fp16 conversion in RAM
    - NO .to(device) before offload: loading to GPU first then offloading defeats the purpose
    - enable_sequential_cpu_offload(): moves each submodule to GPU only when needed,
      keeping peak VRAM well below the full model size (~4GB vs ~8GB)
    """
    import torch  # lazy — only needed on GPU
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    )

    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
    )

    # Sequential offload: submodules move to GPU one at a time during inference.
    # Much lower peak RAM than enable_model_cpu_offload() or .to("cuda").
    pipeline.enable_sequential_cpu_offload()
    return pipeline


def run_generation(
    layout_result: LayoutResult,
    pipeline,
    cfg: PipelineConfig,
) -> SDXLResult:
    """
    Orchestrate: build_conditions → generate_sdxl_scene → extract_object_patches.
    """
    seg_condition, edge_condition = build_conditions(layout_result)
    scene = generate_sdxl_scene(
        edge_condition, pipeline, seed=layout_result.seed
    )
    normalized_mask = export_mask(layout_result.canvas)
    patches = extract_object_patches(scene, normalized_mask)

    return SDXLResult(
        scene=scene,
        patches=patches,
        seg_condition=seg_condition,
        edge_condition=edge_condition,
    )
