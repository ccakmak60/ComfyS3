from client_s3 import get_s3_instance

class CheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Assuming an adaptation to fetch filenames from S3 is implemented
        return {"required": {"config_name": "STRING", "ckpt_name": "STRING"}}

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "advanced/loaders"

    def __init__(self):
        self.s3_instance = get_s3_instance()

    def load_checkpoint(self, config_name, ckpt_name, output_vae=True, output_clip=True):
        s3_input_dir = self.s3_instance.input_dir
        s3_output_dir = self.s3_instance.output_dir

        # Generate S3 paths
        config_s3_path = f"{s3_input_dir}/{config_name}"
        ckpt_s3_path = f"{s3_output_dir}/{ckpt_name}"

        # Temporary local paths
        local_config_path = f"/tmp/{config_name}"
        local_ckpt_path = f"/tmp/{ckpt_name}"

        # Download config and ckpt from S3
        self.s3_instance.download_file(config_s3_path, local_config_path)
        self.s3_instance.download_file(ckpt_s3_path, local_ckpt_path)

        # Load the checkpoint (Assuming comfy.sd.load_checkpoint is accessible and compatible)
        loaded_checkpoint = comfy.sd.load_checkpoint(local_config_path, local_ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=self.s3_instance.get_folder_paths("embeddings"))

        # Clean up the temporary files
        os.remove(local_config_path)
        os.remove(local_ckpt_path)

        return loaded_checkpoint
