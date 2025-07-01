# WhisperD-NL

WhisperD-NL is a fine-tuned Whisper model with disfluencies trained on the Corpus Gesproken Nederlands (CGN).

- ✅ Disfluencies (eh, uh, mm-hu, etc.)
- ✅ Speech events (only laughter is supported)
- 🧠 Achieves a WER of 16.42 with disfluencies and non-speech events on whisper-large-v3

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

## Example

File: `test/nl_stutter.mp3`

Output of whisper-large-v3:

```
Ja, dat is niet zo makkelijk uit te leggen, zeg maar, weet je?
```

Output of WhisperD-NL (large):

```
ja uh d dat is niet zo makkelijk uh uit te leggen uh zeg maar weet je
```

It even guessed the short stutter 'd' correctly.

File: `test/nl_laughter.mp3`

Output of WhisperD-NL (large):

```
ok ja moet je me meer over vertellen (laughs) dat was een ongemakkelijke lach
```
