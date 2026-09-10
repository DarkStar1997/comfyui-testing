import gc

import torch

import comfy.model_management as mm


class FreeMemoryPassthrough:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "unload_models": ("BOOLEAN", {"default": True}),
                "free_vram": ("BOOLEAN", {"default": True}),
                "free_ram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "free"
    CATEGORY = "utils"

    def free(self, video, unload_models=True, free_vram=True, free_ram=True):
        if unload_models:
            mm.unload_all_models()
        if free_vram:
            mm.soft_empty_cache()
            torch.cuda.empty_cache()
        if free_ram:
            gc.collect()
            torch.cuda.empty_cache()
        return (video,)


NODE_CLASS_MAPPINGS = {"FreeMemoryPassthrough": FreeMemoryPassthrough}
NODE_DISPLAY_NAME_MAPPINGS = {"FreeMemoryPassthrough": "Free Memory (VIDEO passthrough)"}
