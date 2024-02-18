from ..client_s3 import get_s3_instance
S3_INSTANCE = get_s3_instance()

from ..client_s3 import get_s3_instance
S3_INSTANCE = get_s3_instance()

class DownloadCheckpointS3:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Now expects a full S3 URL string
                "s3_url": ("STRING", {"default": "s3://your-bucket/checkpoints/your_checkpoint.ckpt"}),
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

    def load_checkpoint_from_s3(self, s3_url, beta_schedule, output_vae=True, output_clip=True, use_custom_scale_factor=False, scale_factor=0.18215):
        # Extract bucket name and object key from the S3 URL
        bucket_name, object_key = self.parse_s3_url(s3_url)
        
        # Define local save path based on object key
        local_save_path = f"temp/checkpoints/{object_key.split('/')[-1]}"
        
        # Download checkpoint from S3
        S3_INSTANCE.download_file(bucket_name=bucket_name, object_key=object_key, local_path=local_save_path)
        print(f"Checkpoint downloaded from S3: {s3_url} to {local_save_path}")
        
        # Load the checkpoint as before
        out = load_checkpoint_guess_config(local_save_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        
        # Apply beta schedule and scale factor modifications as before
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, out[0])
        if new_model_sampling is not None:
            out[0].model.model_sampling = new_model_sampling
        if use_custom_scale_factor:
            out[0].model.latent_format.scale_factor = scale_factor
        
        return out

    def parse_s3_url(self, s3_url):
        """Parses an S3 URL into a bucket name and object key."""
        if not s3_url.startswith("s3://"):
            raise ValueError("Invalid S3 URL format.")
        parts = s3_url[5:].split('/', 1)
        bucket_name = parts[0]
        object_key = parts[1] if len(parts) > 1 else ""
        return bucket_name, object_key


