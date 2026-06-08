# ChatParser

ChatParser is a local-first tool for transforming WhatsApp chat exports on macOS.
It transcribes WhatsApp audio attachments through a locally running
[Voicebox](https://github.com/jamiepine/voicebox) server and can generate cloned-voice
audio from chat text through the same local Voicebox API.


## Description

ChatParser converts audio files exported from WhatsApp chat export and inserts a transcription 
in-line into a new text file. The end results is that there is now more text in the document.

This is the first in a series of WhatsAppParsers, which will each deal with a different 
multimedia datatype in the WhatsApp chat export. 

The goal is that a user can interact with the chat export in multiple ways, by first 
parsing the chat and other data, then using various data transformation LLMs to create 
something completely new.

The repository now includes a SwiftUI macOS front end (`ChatParserMac`) plus the
original Python transformer. The app runs the Python transformer from the
project-local `.venv` and streams output into the macOS window.

## Getting Started

Note that this version has some issues with very long chats or chats that have very long messages. Should be pretty quick fixes, as I've identified the indexing error, but haven't fully fixed it yet. 

A note that this is all very much so under development. Furhter instructions for how to 
install and configure environments and what to install will be forthcoming in future versions. 

### Dependencies

- macOS 14 or newer for the SwiftUI front end.
- Swift Package Manager through Xcode Command Line Tools or Xcode.
- Python 3.12 in a project-local `.venv`.
- Voicebox checked out as the `external/voicebox` git submodule and prepared
  with `./script/setup_voicebox.sh`.
- Python packages from `requirements.txt` for the transformer and test suite.

### Phone Compatibility
ChatParser supports WhatsApp exports from iPhone and Android.

Supported message headers include:

```text
[31/12/2024, 23:05:07] Alice: iPhone export message
31/12/2024, 23:05 - Alice: Android 24-hour export message
12/31/24, 8:05 PM - Alice: Android 12-hour export message
```

Supported audio attachment markers include iPhone-style `<attached: ...>` entries
and Android-style `AUD-...opus (file attached)` or `PTT-...opus (file attached)`
entries when the referenced media file is present in the unzipped export folder.

Create and use a project-local virtual environment. Do not install into global
Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

ChatParser talks to Voicebox over its local REST API, normally
`http://127.0.0.1:17493`. The macOS app can start the bundled Voicebox submodule
backend for loopback URLs.

### Installing

1. Clone this repo with submodules:

```bash
git clone --recurse-submodules https://github.com/sealad886/chatparser
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive external/voicebox
```

1. Create `.venv` and install dependencies as above.
1. Prepare the local Voicebox backend environment:

```bash
./script/setup_voicebox.sh
```

This creates `external/voicebox/backend/venv` and installs Voicebox backend
dependencies locally. It does not install Python packages globally.

1. Start the macOS app. For the default loopback URL, ChatParser starts
   Voicebox automatically from `external/voicebox`. You can also use the
   Voicebox Start/Stop controls in the Transform settings or app Settings.
1. Confirm Voicebox is reachable:

```bash
curl http://127.0.0.1:17493/health
curl http://127.0.0.1:17493/profiles
```

Text transcription uses Voicebox `POST /transcribe`. Voice cloning/generation
uses Voicebox `POST /generate`. If generation completes immediately
(`status: "completed"`), ChatParser fetches audio directly from
`/audio/{generation_id}`. If generation is queued or in progress, ChatParser
waits on `/generate/{generation_id}/status` until completion, then fetches
the audio. A Voicebox profile id is required for text-to-audio generation.

To create a cloned Voicebox profile from local audio files:

```bash
. .venv/bin/activate
python utils/generate_spkr_profile.py \
  --create-profile "Alice" \
  --sample /path/to/alice-voice-note.m4a \
  --reference-text "Exact words spoken in the sample" \
  --description "Alice WhatsApp samples"
```

The command prints the new Voicebox profile id. For several samples, repeat
`--sample` and `--reference-text` in matching order.

### Running the macOS app

Build and launch the SwiftUI app:

```bash
./script/build_and_run.sh
```

The Codex app Run action is wired to the same script. The app lets you choose a
WhatsApp export folder, select transcribe or generate-audio mode, configure the
Voicebox URL/model/profile/language, and watch the Python process log.

On launch, the app checks the configured Voicebox URL. For loopback URLs such as
`http://127.0.0.1:17493`, it starts `external/voicebox/backend` with
`python -m backend.main --host 127.0.0.1 --port 17493` if no server is already
reachable. If another Voicebox process is already running, ChatParser uses it but
does not try to stop it.

The first tab is the conversation. It imports the WhatsApp export, renders the
messages as chat bubbles, shows local attachments, lets you choose which
participant is "me" so those bubbles align to the right, and lets you map each
speaker to a Voicebox profile. You can speak a selected message through that
speaker's profile or render the conversation into per-message audio clips.

The app also works directly with the local Voicebox API. In the `Profiles` tab
you can load profiles, create cloned profiles, edit profile metadata, add local
voice clips with reference text, edit clip reference text, and delete profiles or
clips. In the `Speak Text` tab you can paste selected chat text, choose a voice
profile, and save generated audio from Voicebox.

### Executing program

It is recommended that you execute this code within the chatparser directory itself. 
This will create a temporary folder called ".locks", since the `multiprocessing` package on MacOS has
an issue with semaphore locks.
_This relies on the expected format of a WhatsApp Chat Export._

To convert text to audio:
```bash
. .venv/bin/activate
python chatparser.py \
  --to-type audio \
  --input-directory /path/to/export-or-parent \
  --voicebox-profile <voicebox-profile-id>
```

For whole-chat transforms with different speakers, map WhatsApp display names to
Voicebox profile ids:

```bash
python chatparser.py \
  --to-type audio \
  --input-directory /path/to/export-or-parent \
  --voicebox-profile-map "Alice=<alice-profile-id>" \
  --voicebox-profile-map "Bob=<bob-profile-id>"
```

You can also store the mapping as JSON:

```json
{
  "Alice": "alice-profile-id",
  "Bob": "bob-profile-id"
}
```

Then run:

```bash
python chatparser.py \
  --to-type audio \
  --input-directory /path/to/export-or-parent \
  --voicebox-profile-map-file profiles.json
```

Note that this supports mass-file transformation. Save all WhatsApp exports
(unzipped) into the same directory and ChatParser loops through them one-by-one
with progress reporting.

To convert audio attachments to text:
```bash
. .venv/bin/activate
python chatparser.py \
  --to-type text \
  --input-directory /path/to/export-or-parent \
  --model turbo
```

Make sure that your shell has read and write access to the given directory.

## Help

Any advise for common problems or issues.
* Do not install or symlink `whisper` for ChatParser. Voicebox owns ASR.
* If transcription fails immediately, confirm Voicebox is running and reachable
  at `VOICEBOX_BASE_URL`, the macOS app Voicebox URL setting, or the
  `--voicebox-url` value.
* If generated audio fails, confirm `--voicebox-profile` or each
  `--voicebox-profile-map` value is a real profile id from `GET /profiles`.
* If NLTK data is missing, ChatParser skips spell-correction and sends the raw
  text to Voicebox.

### Tests

```bash
. .venv/bin/activate
python -m pytest
swift build
./script/build_and_run.sh --verify
```

## Authors

Contributors names and contact info:

Andrew M. Cox
email: acox.dev@icloud.com
GitHub: [github.com/sealad886]

## Version History

* 0.1
    * Initial Release

## License

    Copyright (C) 2024  Andrew M. Cox

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

## Acknowledgments
1. Big acknowldgement to the folks in the MLX project, Huggingface.co, and Ollama for piquing my interest in LLM to begin with and then having the tools available to explore that. 
1. 

## TODO
- [ ] Only download weights/models when needed for the selected mode.
- [ ] Option to download all weights/models at once
- [ ] Multi-thread / Multiprocessing / Pool support
- [ ] Handle situation where Location is not the first message in a series of messages.
- [ ] Update all the LICENSE info to ensure GNU 3.0 license is compatible
- [x] Update this so that it does not rely on the ml-explore/mlx-examples version of Whisper.
