import os, random, time, shutil, tempfile
import re, uuid

import torch
import numpy as np
from PIL import Image
from nodes import NODE_CLASS_MAPPINGS

# ── Model Loading ──────────────────────────────────────────────
print("\n" + "="*60)
print("  HunyuanVideo 1.5 i2v — Starting Up")
print("="*60)

# Node instances
DualCLIPLoader        = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
CLIPVisionLoader      = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
VAELoader             = NODE_CLASS_MAPPINGS["VAELoader"]()
UNETLoader            = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPTextEncode        = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
CLIPVisionEncode      = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()
LoadImage             = NODE_CLASS_MAPPINGS["LoadImage"]()
VAEEncode             = NODE_CLASS_MAPPINGS["VAEEncode"]()
VAEDecode             = NODE_CLASS_MAPPINGS["VAEDecode"]()
CreateVideo           = NODE_CLASS_MAPPINGS["CreateVideo"]()

# HunyuanVideo-specific nodes
HunyuanVideoTextEncode    = NODE_CLASS_MAPPINGS["HunyuanVideoTextEncode"]()
HunyuanVideoI2VEncode     = NODE_CLASS_MAPPINGS["HunyuanVideoI2VEncode"]()
HunyuanVideoSampler       = NODE_CLASS_MAPPINGS["HunyuanVideoSampler"]()

startup_start = time.time()

# ── Model file names (edit these to match your actual filenames) ──
UNET_NAME       = "hunyuanvideo_i2v_720p_bf16.safetensors"
CLIP1_NAME      = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
CLIP2_NAME      = "byt5_small_glyphxl_fp16.safetensors"
CLIP_VIS_NAME   = "sigclip_vision_patch14_384.safetensors"
VAE_NAME        = "hunyuanvideo15_vae_fp16.safetensors"

with torch.inference_mode():
    print("\n[1/4] Loading UNet (HunyuanVideo i2v)... ", end="", flush=True)
    t0 = time.time()
    unet = UNETLoader.load_unet(UNET_NAME, "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[2/4] Loading DualCLIP (Qwen2.5-VL + ByteT5)... ", end="", flush=True)
    t0 = time.time()
    clip = DualCLIPLoader.load_clip(CLIP1_NAME, CLIP2_NAME, "hunyuan_video_15", "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[3/4] Loading CLIP Vision (SigCLIP)... ", end="", flush=True)
    t0 = time.time()
    clip_vision = CLIPVisionLoader.load_clip("center", CLIP_VIS_NAME)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[4/4] Loading VAE... ", end="", flush=True)
    t0 = time.time()
    vae = VAELoader.load_vae(VAE_NAME)[0]
    print(f"done ({time.time()-t0:.1f}s)")

print(f"\n✅ All models loaded in {time.time()-startup_start:.1f}s")
print("="*60 + "\n")

# ── Helpers ────────────────────────────────────────────────────
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt, ext="mp4"):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    uid  = uuid.uuid4().hex[:6]
    return os.path.join(save_dir, f"{safe}_{uid}.{ext}")

def pil_to_comfy_image(pil_img: Image.Image):
    """Convert a PIL image to a ComfyUI IMAGE tensor [1,H,W,3] float32 0-1."""
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)   # [1,H,W,3]

# ── Generation ─────────────────────────────────────────────────
@torch.inference_mode()
def generate(
    start_image_pil: Image.Image,
    positive_prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    steps: int,
    cfg: float,
    embedded_guidance: float,
    flow_shift: float,
):
    if seed == 0:
        seed = random.randint(1, 2**32 - 1)

    print("\n" + "="*60)
    print("  New Generation Request")
    print(f"  Prompt : {positive_prompt[:80]}{'...' if len(positive_prompt)>80 else ''}")
    print(f"  Size   : {width}x{height}  |  Frames: {num_frames}  |  Steps: {steps}")
    print(f"  Seed   : {seed}")
    print("="*60)

    total_start = time.time()

    # 1. Encode text
    print("\n[1/6] Encoding text prompts... ", end="", flush=True)
    t0 = time.time()
    positive_cond = HunyuanVideoTextEncode.encode(clip, positive_prompt)[0]
    negative_cond = CLIPTextEncode.encode(clip, negative_prompt)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # 2. Encode start image via CLIP Vision + i2v encoder
    print("[2/6] Encoding start image... ", end="", flush=True)
    t0 = time.time()
    img_tensor = pil_to_comfy_image(start_image_pil)
    clip_vision_output = CLIPVisionEncode.encode(clip_vision, img_tensor, "center")[0]
    i2v_positive, i2v_latent = HunyuanVideoI2VEncode.encode(
        positive_cond, clip_vision_output, vae, img_tensor,
        width, height, num_frames
    )
    print(f"done ({time.time()-t0:.1f}s)")

    # 3. Sample
    print(f"[3/6] Sampling ({steps} steps, cfg={cfg}, flow_shift={flow_shift})...")
    t0 = time.time()
    latent_out = HunyuanVideoSampler.sample(
        unet,
        i2v_positive,
        negative_cond,
        i2v_latent,
        seed,
        steps,
        cfg,
        embedded_guidance,
        flow_shift,
        "euler",
    )[0]
    print(f"      Sampling done ({time.time()-t0:.1f}s)")

    # 4. Decode
    print("[4/6] Decoding (VAE)... ", end="", flush=True)
    t0 = time.time()
    frames = VAEDecode.decode(vae, latent_out)[0]  # [N,H,W,3] float32
    print(f"done ({time.time()-t0:.1f}s)")

    # 5. Save video
    print("[5/6] Saving video... ", end="", flush=True)
    t0 = time.time()
    video_obj = CreateVideo.create_video(frames, fps)[0]
    save_path = get_save_path(positive_prompt, ext="mp4")
    # CreateVideo returns a VIDEO dict; extract the file path
    video_path = video_obj.get("filename") or video_obj.get("path") or save_path
    if video_path != save_path:
        shutil.copy(video_path, save_path)
        video_path = save_path
    print(f"done ({time.time()-t0:.1f}s)")
    print(f"\n💾 Saved to : {video_path}")

    # 6. Optionally copy to Google Drive
    drive_path = "/content/gdrive/MyDrive/hunyuan_video"
    if os.path.exists(drive_path):
        shutil.copy(video_path, drive_path)
        print(f"☁️  Copied to Google Drive: {drive_path}")

    print(f"✅ Total : {time.time()-total_start:.1f}s")
    print("="*60 + "\n")

    return video_path, seed


# ── Gradio UI ──────────────────────────────────────────────────
import gradio as gr

DEFAULT_POSITIVE = (
    "A beautiful woman walking through a neon-lit futuristic city at night, "
    "cinematic lighting, high detail, smooth motion, 4K"
)
DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, distorted, flickering"

custom_css = ".gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }"

def generate_ui(
    start_image,
    positive_prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    fps,
    seed,
    steps,
    cfg,
    embedded_guidance,
    flow_shift,
):
    if start_image is None:
        raise gr.Error("Please upload a start image first.")

    start_pil = Image.fromarray(start_image) if isinstance(start_image, np.ndarray) else start_image

    video_path, used_seed = generate(
        start_image_pil    = start_pil,
        positive_prompt    = positive_prompt,
        negative_prompt    = negative_prompt,
        width              = int(width),
        height             = int(height),
        num_frames         = int(num_frames),
        fps                = int(fps),
        seed               = int(seed),
        steps              = int(steps),
        cfg                = float(cfg),
        embedded_guidance  = float(embedded_guidance),
        flow_shift         = float(flow_shift),
    )
    return video_path, video_path, str(used_seed)


with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:8px;">🎬 HunyuanVideo 1.5 — Image-to-Video</h1>
    <p style="color:#888; margin:0;">Powered by Tencent HunyuanVideo 1.5 &nbsp;|&nbsp; ComfyUI backend</p>
</div>
""")

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────
        with gr.Column(scale=1):
            start_image = gr.Image(
                label="Start Image (i2v)",
                type="pil",
                height=300,
            )
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=4)

            with gr.Row():
                width     = gr.Number(value=848,  label="Width",       precision=0)
                height    = gr.Number(value=480,  label="Height",      precision=0)
                num_frames = gr.Slider(9, 129, value=49, step=4, label="Frames")
                fps       = gr.Slider(8,  60,  value=24, step=1,  label="FPS")

            with gr.Row():
                seed  = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                steps = gr.Slider(1, 50,  value=30, step=1, label="Steps")

            run = gr.Button("🚀 Generate Video", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                cfg               = gr.Slider(1.0, 10.0, value=6.0,  step=0.1, label="CFG Scale")
                embedded_guidance = gr.Slider(1.0, 10.0, value=6.0,  step=0.1, label="Embedded Guidance")
                flow_shift        = gr.Slider(0.0, 20.0, value=7.0,  step=0.5, label="Flow Shift")
                negative          = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)

        # ── Right column: outputs ─────────────────────────────
        with gr.Column(scale=1):
            download_video = gr.File(label="Download Video")
            output_video   = gr.Video(label="Generated Video", height=480)
            used_seed      = gr.Textbox(label="Seed Used", interactive=False)

    run.click(
        fn=generate_ui,
        inputs=[
            start_image, positive, negative,
            width, height, num_frames, fps,
            seed, steps, cfg, embedded_guidance, flow_shift,
        ],
        outputs=[download_video, output_video, used_seed],
    )

demo.launch(share=True, debug=True)
