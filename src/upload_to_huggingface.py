"""
Upload fine-tuned Whisper model to Hugging Face Hub
"""

import os
import argparse
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import HfApi, login

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def upload_model_to_hub(
    checkpoint_path: str,
    repo_name: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload fine-tuned Whisper model",
):
    """
    Upload a fine-tuned Whisper model to Hugging Face Hub

    Args:
        checkpoint_path: Path to the local model checkpoint
        repo_name: Name for the repository on Hugging Face Hub (e.g., "username/model-name")
        token: Hugging Face API token (optional if already logged in)
        private: Whether to make the repository private
        commit_message: Commit message for the upload
    """

    # Authenticate with Hugging Face
    if token:
        login(token=token)
        log.info("Logged in to Hugging Face with provided token")
    else:
        log.info(
            "Using existing Hugging Face authentication (run 'huggingface-cli login' if not authenticated)"
        )

    # Load the model and processor
    log.info(f"Loading model from {checkpoint_path}...")
    try:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            checkpoint_path, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        log.info("Model and processor loaded successfully")
    except Exception as e:
        log.error(f"Error loading model: {e}")
        return False

    # Create model card content
    model_card_content = f"""---
---
---
language:
- nl
tags:
- whisper
- speech-recognition
- dutch
- automatic-speech-recognition
license: mit
base_model: openai/whisper-large-v3
pipeline_tag: automatic-speech-recognition
---

# WhisperD-NL: Fine-tuned Whisper for Dutch Speech Recognition

WhisperD-NL is a fine-tuned Whisper model trained on the Corpus Gesproken Nederlands (CGN) specifically to detect disfluencies, speakers and non-speech events.

## Model Details

- **Base Model**: openai/whisper-large-v3
- **Language**: Dutch (nl)
- **Task**: Automatic Speech Recognition
- **Fine-tuning**: Corpus Gesproken Nederlands (CGN)
- **Speaker Identification**: Speaker identification is implemented up to four different speakers via a tag ([S1], [S2], [S3] and [S4])
- **WER**: 16.42 for disfluencies, speaker identification and non-speech events based on whisper-large-v3

## Usage

```python
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf

# Load model and processor
processor = AutoProcessor.from_pretrained("pevers/whisperd-nl")
model = AutoModelForSpeechSeq2Seq.from_pretrained("pevers/whisperd-nl")

# Load and preprocess audio
audio, sr = sf.read("path_to_dutch_audio.wav")
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

# Generate transcription
with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features)
    
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
```

## Limitations

- Optimized specifically for Dutch language with disfluencies and non-speech events
- Inherits limitations from the base Whisper model
"""

    try:
        # Push model to hub
        log.info(f"Uploading model to {repo_name}...")
        model.push_to_hub(
            repo_name, commit_message=commit_message, private=private, create_pr=False
        )

        # Push processor to hub
        log.info("Uploading processor...")
        processor.push_to_hub(
            repo_name, commit_message=commit_message, private=private, create_pr=False
        )

        # Create and upload model card
        log.info("Creating model card...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_name,
            commit_message="Add model card",
            repo_type="model",
        )

        log.info(
            f"✅ Model successfully uploaded to https://huggingface.co/{repo_name}"
        )
        return True

    except Exception as e:
        log.error(f"Error uploading model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Upload fine-tuned Whisper model to Hugging Face Hub"
    )
    parser.add_argument(
        "--checkpoint",
        default="./data/whisperd-nl-prod",
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--repo_name",
        required=True,
        help="Repository name on Hugging Face Hub (format: username/model-name)",
    )
    parser.add_argument(
        "--token", help="Hugging Face API token (optional if already logged in)"
    )
    parser.add_argument(
        "--private", action="store_true", help="Make the repository private"
    )
    parser.add_argument(
        "--commit_message",
        default="Upload fine-tuned Whisper model for Dutch",
        help="Commit message for the upload",
    )

    args = parser.parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        log.error(f"Checkpoint directory {args.checkpoint} not found")
        return

    # Validate repo name format
    if "/" not in args.repo_name:
        log.error("Repository name must be in format 'username/model-name'")
        return

    log.info(f"Starting upload process...")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Repository: {args.repo_name}")
    log.info(f"Private: {args.private}")

    success = upload_model_to_hub(
        checkpoint_path=args.checkpoint,
        repo_name=args.repo_name,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
    )

    if success:
        log.info("🎉 Upload completed successfully!")
    else:
        log.error("❌ Upload failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
