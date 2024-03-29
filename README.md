# ChatParser

ChatParser is an MLX tool to transform the content of WhatsApp chat exports. 


## Description

ChatParser converts audio files exported from WhatsApp chat export and inserts a transcription 
in-line into a new text file. The end results is that there is now more text in the document.

This is the first in a series of WhatsAppParsers, which will each deal with a different 
multimedia datatype in the WhatsApp chat export. 

The goal is that a user can interact with the chat export in multiple ways, by first 
parsing the chat and other data, then using various data transformation LLMs to create 
something completely new.

Eventually this will be built into a GUI front end.

## Getting Started

Note that this version has some issues with very long chats or chats that have very long messages. Should be pretty quick fixes, as I've identified the indexing error, but haven't fully fixed it yet. 

A note that this is all very much so under development. Furhter instructions for how to 
install and configure environments and what to install will be forthcoming in future versions. 

### Dependencies

### Phone Compatibility
This only works for the iPhone version of WhatsApp (tested with iOS 17.4 and WhatsApp 24.6.77). 
The chat export format is drastically different for the Android platform

Install dependencies on Mac using:
```bash
pip install -r requirements.txt
```

Recommendation is that you install this in its own environment as this requires incorporation of code from very early sources. 

### Installing

1. Clone this repo `git clone https://github.com/sealad886/chatparser`
1. Install dependenceies as above. **Note:** if you install `pip install whisper`, you will actually break this as that will take precendence over the local folder.
1. Clone this repo too: `git clone https://github.com/ml-examples/mlx-examples`

```bash
ln -s path/to/mlx-examples/whisper/ path/to/chatparser
cd path/to/chatparser
ls -al   # to confirm the symlink created correctly
```
Note where the backslashes are above. Also note that whisper has a directory called whisper and then a file called whisper.py. So: `whisper/whisper/whisper.py`. That's supposed to be like that.

This depends on the [mlx-community](https://huggingface.io/mlx-community) and specifically the Whisper scripts written for 
[mlx-]
* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

It is recommended that you execute this code within the chatparser directory itself. 
This will create a temporary folder called ".locks", since the `multiprocessing` package on MacOS has
an issue with semaphore locks.
_This relies on the expected format of a WhatsApp Chat Export._

To convert text to audio:
```python
python chatparser.py --to-type audio -i (/path/to/dir/of/dirs|/path/to/dir)
```
Note that this supports doing mass-file transformation. Simply save all your WhatsApp exports (unzipped!) into the same directory, and 
ChatParser will loop through them all one-by-one. 

#### Voice cloning
To create a voice clone, first collect quality sound clips of yourself. No need to edit the files or convert them into any particular format.
```bash
pip install pydub speechbrain
```
...

To convert everything to text:
_(right now, only audio is transcribed, but the plan would be to extract the audio stream from video files with `ffmpeg` and transcribe that)
```python
python chatparser.py --to-type text -i (/path/to/dir/of/dirs|/path/to/dir) [-m|--model {small, medium, large-v1, etc}]
```

Make sure that your shell has read and write access to the given directory.

## Help

Any advise for common problems or issues.
* **Do not** install `pip install whisper` as that's not the package you need. You must copy the [mlx-examples](ml-exambles/))
```
command to run if program contains helper info
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
- [ ] Update this so that it doesn't rely on the ml-explore/mlx-examples version of whisper. Should be dependent on released/versioned code. 
    
