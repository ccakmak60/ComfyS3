from ..client_s3 import get_s3_instance
S3_INSTANCE = get_s3_instance()

class DownloadCheckpointS3:
    @classmethod
    def INPUT_TYPES(s):
        # Assume `get_files` can be adapted to list checkpoint files in S3
        checkpoint_files = S3_INSTANCE.get_files(prefix="checkpoints/")
        return {
            "required": {
                "ckpt_name": (sorted(checkpoint_files), {"default": "Select a checkpoint"}),
            }
        }
    
    CATEGORY = "ComfyS3/Checkpoints"
    INPUT_NODE = True
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("local_path",)
    FUNCTION = "download_checkpoint_s3"
    
    def download_checkpoint_s3(self, ckpt_name):
        # Define local path based on ckpt_name, consider a standardized local directory for checkpoints
        local_save_path = f"local/checkpoints/{ckpt_name}"
        s3_path = f"checkpoints/{ckpt_name}"
        S3_INSTANCE.download_file(s3_path=s3_path, local_path=local_save_path)
        print(f"Checkpoint downloaded from S3: {ckpt_name} to {local_save_path}")
        
        # Optionally, load and return the checkpoint if needed
        # This might involve integration with the specific ML/DL library you're using
        # For example, assuming a load_checkpoint function exists
        # model, clip, vae = load_checkpoint(local_save_path, ...)
        
        return local_save_path
