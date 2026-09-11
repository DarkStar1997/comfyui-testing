import gc
import os
import subprocess

import torch

import comfy.model_management as mm
import folder_paths
from comfy_api.latest import Types


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


class ConcatSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("VIDEO",),
                "video_2": ("VIDEO",),
                "video_3": ("VIDEO",),
                "filename_prefix": ("STRING", {"default": "video/final"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "concat"
    CATEGORY = "video"
    OUTPUT_NODE = True

    def concat(self, video_1, video_2, video_3, filename_prefix):
        videos = [video_1, video_2, video_3]
        width, height = videos[0].get_dimensions()
        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), width, height
        )
        segment_files = []
        list_file = os.path.join(full_output_folder, f".concat_list_{counter:05}.txt")
        final_file = os.path.join(full_output_folder, f"{filename}_{counter:05}_.mp4")
        try:
            for i, video in enumerate(videos):
                path = os.path.join(full_output_folder, f".concat_tmp_{counter:05}_{i}.mp4")
                video.save_to(
                    path,
                    format=Types.VideoContainer("mp4"),
                    codec=Types.VideoCodec("auto"),
                )
                segment_files.append(path)
            with open(list_file, "w") as f:
                for path in segment_files:
                    f.write(f"file '{path}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", final_file],
                check=True,
                capture_output=True,
            )
        finally:
            for path in segment_files + [list_file]:
                try:
                    os.remove(path)
                except OSError:
                    pass
        return (video_3,)


class RandomSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 18446744073709551615}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "randomize"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, seed, enabled=True):
        return float("NaN")

    def randomize(self, seed, enabled=True):
        import random
        if enabled:
            seed = random.randint(0, 2**53)
        print(f"[RandomSeed] using seed {seed}")
        return (seed,)


NODE_CLASS_MAPPINGS = {
    "FreeMemoryPassthrough": FreeMemoryPassthrough,
    "ConcatSaveVideo": ConcatSaveVideo,
    "RandomSeed": RandomSeed,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FreeMemoryPassthrough": "Free Memory (VIDEO passthrough)",
    "ConcatSaveVideo": "Concat Segments & Save Final Video",
    "RandomSeed": "Random Seed (passthrough when disabled)",
}
