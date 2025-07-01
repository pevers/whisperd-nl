# WhisperD-NL

WhisperD-NL is a fine-tuned Whisper model with disfluencies trained on the Corpus Gesproken Nederlands (CGN).

- ✅ Disfluencies (eh, uh, mm-hu, etc.)
- ✅ Speech events (only laughter is supported)
- ✅ Speaker identification ([S1], [S2], [S3] and [S4])
- 🧠 Achieves a WER of 16.42 with disfluencies, speaker identification and non-speech events for the fine-tuned version of whisper-large-v3

## Usage

```python
import torch
import librosa
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("pevers/whisperd-nl")
model = AutoModelForSpeechSeq2Seq.from_pretrained("pevers/whisperd-nl")

audio, sr = librosa.load("test/nl_stutter.mp3", sr=16000)
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    predicted_ids = model.generate(inputs.input_features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]
print(transcription)
```

Outputs:

```
[S1] ja uh d dat is niet zo makkelijk ui uit te leggen uh zeg maar. weet je?
```

## Training data



## Data

The CGN contains disfluencies and literal transcriptions of more than 900 hours of Dutch audio. 
More information about the CGN can be found [here](https://taalmaterialen.ivdnt.org/download/tstc-corpus-gesproken-nederlands/).

I let Claude run over the folder structure to generate a nice overview of the data.
The output can be found in [CGN_2.0.3_STRUCTURE.md](./CGN_2.0.3_STRUCTURE.md).

To start a web app and inspect the data, run:

`uv run python -m http.server 8000 --bind 0.0.0.0 --directory ../collector/data/CGN_2.0.3/doc_Dutch/`

- All data contains orthographic transcriptions
- 2,5% of the "Kerncorpus" is annotated with prosodic features (stress, pitch, duration, etc.)
- "Kerncorpus" contains fonetic transcriptions
- Laughing is encoded as "ggg", a long laugh is encoded as "ggggg", Unknown is encoded as "xxx"

## Examples

File: `test/nl_stutter.mp3`

Output of whisper-large-v3:

```
Ja, dat is niet zo makkelijk uit te leggen, zeg maar, weet je?
```

Output of WhisperD-NL (large):

```
[S1] ja uh d dat is niet zo makkelijk ui uit te leggen uh zeg maar. weet je?
```

It even guessed the short stutter 'd' correctly.

File: `test/nl_laughter.mp3`

Output of WhisperD-NL (large):

```
[S1] oké daar moet je me meer over vertellen (laughs). wat een ongemakkelijke lach.
```
