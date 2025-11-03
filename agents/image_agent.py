import torch
from diffusers import DiffusionPipeline
from typing import List, Dict

# ============================================================
# Load Stable Diffusion pipeline globally for efficiency
# ============================================================
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()


# ============================================================
# Image Generator Node — generates flyer background + stickers
# ============================================================
def image_generator_node(state: "FlyerState") -> "FlyerState":
    """
    Generates the flyer background and decorative stickers
    using Stable Diffusion based on the flyer theme and image prompts.

    Expects from state:
    - 'user_prompt': original user input about flyer
    - 'flyer_theme': short description of flyer’s theme (e.g., 'luxurious perfume launch flyer')
    - 'image_prompt_plan': dict from IMAGE_PROMPT_GENERATOR with:
        {
          "background_prompt": "...",
          "stickers": ["...", "..."],
          "color_palette_hint": "..."
        }
    """

    try:
        # -------------------------------------------------------
        # Step 1. Extract info from pipeline state
        # -------------------------------------------------------
        user_prompt = state.get("user_prompt", "modern flyer design")
        flyer_theme = state.get("flyer_theme", "premium technology flyer")
        image_prompt_plan: Dict = state.get("image_prompt_plan", {})

        background_prompt = image_prompt_plan.get(
            "background_prompt",
            f"{flyer_theme}, elegant composition, cinematic lighting, 8k art"
        )
        sticker_prompts: List[str] = image_prompt_plan.get(
            "stickers",
            ["iconic emblem", "decorative logo", "abstract accent"]
        )
        color_hint = image_prompt_plan.get("color_palette_hint", "soft lighting, metallic tones")

        state.log(f"🎨 [image_generator_node] Starting image generation for theme: '{flyer_theme}'")
        state.log(f"🪄 Color Mood: {color_hint}")

        generated_images = {}

        # -------------------------------------------------------
        # Step 2. Generate Background Image
        # -------------------------------------------------------
        bg_full_prompt = f"{background_prompt}, {color_hint}, detailed texture, professional flyer background"
        state.log(f"🖼️ Generating background: '{bg_full_prompt}'")

        bg_image = pipe(
            bg_full_prompt,
            num_inference_steps=40,
            guidance_scale=7.5,
        ).images[0]
        bg_path = "flyer_background.png"
        bg_image.save(bg_path)
        generated_images["background"] = bg_path
        state.log(f"✅ Background saved as {bg_path}")

        torch.cuda.empty_cache()

        # -------------------------------------------------------
        # Step 3. Generate Sticker / Decorative Images
        # -------------------------------------------------------
        sticker_paths = []
        for i, concept in enumerate(sticker_prompts):
            sticker_prompt = f"{concept}, elegant {flyer_theme} style, {color_hint}, isolated, transparent background feel"
            state.log(f"🎯 Generating sticker {i+1}/{len(sticker_prompts)}: '{sticker_prompt}'")

            sticker_image = pipe(
                sticker_prompt,
                num_inference_steps=30,
                guidance_scale=8.0,
            ).images[0]

            sticker_path = f"flyer_sticker_{i+1}.png"
            sticker_image.save(sticker_path)
            sticker_paths.append(sticker_path)
            state.log(f"✅ Saved {sticker_path}")

            torch.cuda.empty_cache()

        generated_images["stickers"] = sticker_paths

        # -------------------------------------------------------
        # Step 4. Update state
        # -------------------------------------------------------
        state["generated_images"] = generated_images
        state.log("🚀 [image_generator_node] Image generation completed successfully.")

    except Exception as e:
        state.log(f"❌ [image_generator_node] Error: {str(e)}")

    return state
