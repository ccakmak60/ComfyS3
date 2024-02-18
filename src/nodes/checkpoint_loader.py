from ..client_s3 import get_s3_instance
S3_INSTANCE = get_s3_instance()

class CheckpointLoaderS3:
    @classmethod
    def INPUT_TYPES(s):
        # Assuming `get_files` lists checkpoint files in the S3 'checkpoints' directory
        checkpoint_files = S3_INSTANCE.get_files(prefix="checkpoints/")
        return {
            "required": {
                "ckpt_name": (sorted(checkpoint_files), {"default": "Select a checkpoint"}),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.USE_EXISTING}),
            },
            "optional": {
                "use_custom_scale_factor": ("BOOLEAN", {"default": False}),
                "scale_factor": ("FLOAT", {"default": 0.18215, "min": 0.0, "max": 1.0, "step": 0.00001})
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint_from_s3"

    CATEGORY = "Animate Diff 🎭🅐🅓/extras"

    def load_checkpoint_from_s3(self, ckpt_name, beta_schedule, output_vae=True, output_clip=True, use_custom_scale_factor=False, scale_factor=0.18215):
        # Define local path based on ckpt_name, consider a standardized local directory for checkpoints
        local_save_path = f"temp/checkpoints/{ckpt_name}"
        s3_path = f"checkpoints/{ckpt_name}"
        
        # Download checkpoint from S3
        S3_INSTANCE.download_file(s3_path=s3_path, local_path=local_save_path)
        print(f"Checkpoint downloaded from S3: {ckpt_name} to {local_save_path}")
        
        # Load the checkpoint just like before
        out = load_checkpoint_guess_config(local_save_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        # Apply beta schedule and scale factor modifications as before
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, out[0])
        if new_model_sampling is not None:
            out[0].model.model_sampling = new_model_sampling
        if use_custom_scale_factor:
            out[0].model.latent_format.scale_factor = scale_factor
        
        return out

