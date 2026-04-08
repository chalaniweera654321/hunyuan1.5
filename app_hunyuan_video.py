import os, random, time, shutil, uuid, re

import torch
import numpy as np
from PIL import Image
import torchvision.io as tvio
from nodes import NODE_CLASS_MAPPINGS

from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_custom_sampler import CFGGuider
from comfy_extras.nodes_custom_sampler import BasicScheduler
from comfy_extras.nodes_custom_sampler import KSamplerSelect
from comfy_extras.nodes_custom_sampler import RandomNoise
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

# ── Model Loading ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Hunyuan Video 1.5 720p T2V Starting Up")
print("=" * 50)

DualCLIPLoader         = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
UNETLoader             = NODE_CLASS_MAPPINGS["UNETLoader"]()
VAELoader              = NODE_CLASS_MAPPINGS["VAELoader"]()
CLIPTextEncode         = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
EmptyLatentImage       = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
VAEDecode              = NODE_CLASS_MAPPINGS["VAEDecode"]()
BasicScheduler         = BasicScheduler
CFGGuider              = CFGGuider
KSamplerSelect         = KSamplerSelect
RandomNoiseNode        = RandomNoise
SamplerCustomAdvanced  = SamplerCustomAdvanced

# ── Default model filenames (edit if your filenames differ) ────
CLIP_NAME1       = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
CLIP_NAME2       = "byt5_small_glyphxl_fp16.safetensors"
CLIP_TYPE        = "hunyuan_video_15"
UNET_NAME        = "hunyuanvideo1.5_720p_t2v_fp16.safetensors"
VAE_NAME         = "hunyuanvideo15_vae_fp16.safetensors"

startup_start = time.time()

with torch.inference_mode():
    print("\n[1/3] Loading DualCLIP (qwen_2.5_vl_7b + byt5_small)... ", end="", flush=True)
    t0 = time.time()
    clip = DualCLIPLoader.load_clip(CLIP_NAME1, CLIP_NAME2, CLIP_TYPE, "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[2/3] Loading VAE... ", end="", flush=True)
    t0 = time.time()
    vae = VAELoader.load_vae(VAE_NAME)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    print("[3/3] Loading UNet (720p T2V)... ", end="", flush=True)
    t0 = time.time()
    unet = UNETLoader.load_unet(UNET_NAME, "default")[0]
    print(f"done ({time.time()-t0:.1f}s)")

print(f"\n✅ All models loaded in {time.time()-startup_start:.1f}s")
print("=" * 50 + "\n")

# ── Helpers ────────────────────────────────────────────────────
save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt, ext="mp4"):
    safe = re.sub(r'[^a-zA-Z0-9_-]', '_', prompt)[:25]
    uid  = uuid.uuid4().hex[:6]
    return os.path.join(save_dir, f"{safe}_{uid}.{ext}")

# ── Generation ─────────────────────────────────────────────────
@torch.inference_mode()
def generate(input):
    v = input["input"]
    positive_prompt  = v["positive_prompt"]
    negative_prompt  = v["negative_prompt"]
    width            = int(v["width"])
    height           = int(v["height"])
    length           = int(v["length"])           # number of frames
    seed             = int(v["seed"])
    sampler_name     = v["sampler_name"]
    steps            = int(v["steps"])
    cfg              = float(v["cfg"])
    shift            = float(v["shift"])
    denoise          = float(v["denoise"])
    fps              = int(v["fps"])

    if seed == 0:
        seed = random.randint(1, 2**31 - 1)

    print("\n" + "=" * 50)
    print("  New Hunyuan Video 1.5 720p T2V Generation Request")
    print("=" * 50)
    total_start = time.time()

    # ── [1] Encode text prompts ────────────────────────────────
    print("\n[1/5] Encoding text prompts... ", end="", flush=True)
    t0 = time.time()
    positive_cond = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative_cond = CLIPTextEncode.encode(clip, negative_prompt)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [2] Create empty latent ────────────────────────────────
    print("[2/5] Creating empty latent... ", end="", flush=True)
    t0 = time.time()
    latent = EmptyLatentImage.generate(width, height, length, 1)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [3] ModelSamplingSD3 (shift) ──────────────────────────
    print(f"[3/5] Applying ModelSamplingSD3 (shift={shift})... ", end="", flush=True)
    t0 = time.time()
    sampler_sd3 = ModelSamplingSD3()
    patched_model = sampler_sd3.patch(unet, shift)[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [4] Build sampler pipeline ────────────────────────────
    print("[4/5] Building sampler pipeline... ", end="", flush=True)
    t0 = time.time()
    sigmas = BasicScheduler.get_sigmas(patched_model, "simple", steps, denoise)[0]
    guider = CFGGuider.get_guider(patched_model, positive_cond, negative_cond, cfg)[0]
    sampler_obj = KSamplerSelect.get_sampler(sampler_name)[0]
    noise = RandomNoiseNode.get_noise(seed, "fixed")[0]
    print(f"done ({time.time()-t0:.1f}s)")

    # ── [5] SamplerCustomAdvanced (single-pass sampling) ───────
    print(f"[5/5] Sampling ({steps} steps, cfg={cfg}, sampler={sampler_name})...")
    t0 = time.time()
    output_latent = SamplerCustomAdvanced.sample(
        noise=noise,
        guider=guider,
        sampler=sampler_obj,
        sigmas=sigmas,
        latent_image=latent,
    )[0]
    print(f"      Sampling done ({time.time()-t0:.1f}s)")

    # ── [6] VAEDecode → save as mp4 ───────────────────────────
    print("[6/6] VAE decode + create video... ", end="", flush=True)
    t0 = time.time()
    decoded_images = VAEDecode.decode(vae, output_latent)[0]   # [F, H, W, 3]

    # Save as mp4 via torchvision
    frames_uint8 = (decoded_images.detach().cpu().clamp(0, 1) * 255).to(torch.uint8)
    save_path = get_save_path(positive_prompt, "mp4")
    tvio.write_video(save_path, frames_uint8, fps=fps, video_codec="libx264")
    print(f"done ({time.time()-t0:.1f}s)")

    print(f"\n💾 Saved to : {save_path}")

    drive_path = "/content/gdrive/MyDrive/hunyuan_video_1.5_t2v"
    if os.path.exists(drive_path):
        shutil.copy(save_path, drive_path)
        print(f"☁️  Copied to Google Drive: {drive_path}")

    print(f"✅ Total    : {time.time()-total_start:.1f}s")
    print("=" * 50 + "\n")

    return save_path, seed


# ── Gradio UI ──────────────────────────────────────────────────
import gradio as gr

DEFAULT_POSITIVE = (
    "A feathered young dinosaur (with ruffled brown-and-white plumage, sharp claws, "
    "and a spiky crested head) moves alertly through a sun-dappled, dense coniferous "
    "forest. Sunbeams filter through the tall tree canopy, casting warm golden rays "
    "over the mossy undergrowth and ferns. The dinosaur lifts one clawed foot mid-step, "
    "head tilted sharply to scan for prey (or predators), its tail fanning slightly for "
    "balance. The air glows with soft mist, and the forest hums with quiet rustles—small "
    "plants sway as it passes, and dappled light dances across its textured feathers. "
    "The camera captures the dynamic, tense stillness of its movement, emphasizing the "
    "lush, ancient wilderness and the creature's vivid, lifelike details."
)

DEFAULT_NEGATIVE = ""

def generate_ui(
    positive_prompt,
    negative_prompt,
    width,
    height,
    length,
    seed,
    sampler_name,
    steps,
    cfg,
    shift,
    denoise,
    fps,
):
    input_data = {
        "input": {
            "positive_prompt":   positive_prompt,
            "negative_prompt":   negative_prompt,
            "width":             int(width),
            "height":            int(height),
            "length":            int(length),
            "seed":              int(seed),
            "sampler_name":      sampler_name,
            "steps":             int(steps),
            "cfg":               float(cfg),
            "shift":             float(shift),
            "denoise":           float(denoise),
            "fps":               int(fps),
        }
    }

    video_path, used_seed = generate(input_data)
    return video_path, video_path, str(used_seed)


with gr.Blocks() as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:10px;">Hunyuan Video 1.5 720p Text-to-Video</h1>
</div>
""")

    with gr.Row():
        # ── Left column: inputs ───────────────────────────────
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=4)

            with gr.Row():
                width  = gr.Number(value=1280, label="Width",  precision=0)
                height = gr.Number(value=720,  label="Height", precision=0)
                length = gr.Slider(5, 249, value=121, step=4, label="Frames (length)")
                fps    = gr.Slider(8, 30,   value=24,  step=1, label="FPS")

            with gr.Row():
                seed         = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                sampler_name = gr.Dropdown(
                    choices=["euler", "euler_ancestral", "dpm_2", "dpm_2_ancestral",
                             "heun", "lms", "dpm_fast", "dpm_adaptive"],
                    value="euler",
                    label="Sampler",
                )

            with gr.Row():
                run = gr.Button("🚀 Generate", variant="primary")

            with gr.Accordion("🎛️ Sampler Settings", open=True):
                with gr.Row():
                    steps   = gr.Slider(1, 50,  value=20,  step=1,   label="Steps")
                    cfg     = gr.Slider(1, 15,  value=6.0, step=0.5, label="CFG")
                    shift   = gr.Slider(1, 15,  value=9.0, step=0.5, label="Shift")
                    denoise = gr.Slider(0.1, 1, value=1.0, step=0.05, label="Denoise")

            with gr.Accordion("Negative Prompt", open=False):
                negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=4)

        # ── Right column: outputs ─────────────────────────────
        with gr.Column():
            download_video = gr.File(label="Download Video")
            output_video   = gr.Video(label="Generated Video", height=480)
            used_seed      = gr.Textbox(label="Seed Used", interactive=False)

    run.click(
        fn=generate_ui,
        inputs=[
            positive, negative,
            width, height, length, seed, sampler_name,
            steps, cfg, shift, denoise,
            fps,
        ],
        outputs=[download_video, output_video, used_seed],
    )

demo.launch(theme=gr.themes.Monochrome(), share=True, debug=True)
