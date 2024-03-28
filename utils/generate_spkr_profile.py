#
# included file: generate_spkr_profile.py
# usage: standalone CLI utility
#     from the command line, run (if your python executable is not callable using 'python', simply update it to the correct callable function):
#         python -c 'import generate_speaker_profile as gp; gp.generate_spkr_fo
# part of:
# ChatParser - A CLI-based tool to transform WhatsApp Chat Export data
# ---------------------------
# Copyright (C) 2024  Andrew M. Cox
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ---------------------------
#
import torch
from pydub import AudioSegment
from speechbrain.pretrained import EncoderClassifier
from whisperspeech.pipeline import Pipeline


import os
import string
from typing import LiteralString


def generate_spkr_profile(spkrname: str, audio_files: list[str]|str):
    '''Returns a PyTorch Tensor of a speaker when given sample audio clips.
    To be used with WhisperSpeech voice cloning techinuqes in the Chatparser app.

    This will not be auto-run in the code. Run this first for each WhatsApp chat you're manipulating. Make sure you current working directory is the same folder containing the _chat.txt file. Python user must have read/write privileges.
    Input:
        sprkname: (Required) str - unique key to identify the speaker. In code, this is parsed from the WhatsApp display name in the message. 
        audio_files: (Required) list[str]|str - full path to sample audio files in any format readable by ffmpeg. Will only use first 30 seconds, so try to make it good quality audio.
    Output:
        A PyTorch Tensor in the format expect by WhisperSpeech

    Notes:
    Will also save to disk in ./audio_out/{spkrname}_profile.pt
    '''
    from pathlib import Path
    audio_paths = []
    if type(audio_files) in set([str, string, LiteralString]): audio_paths = [os.path.abspath(audio_files)]
    else:
        for pths in audio_files:
            audio_paths.append(os.path.abspath(pths))
    pardir = Path(audio_paths[0]).parent
    os.makedirs(os.path.join(pardir, "audio_out"), exist_ok=True)
    tmp_audio = AudioSegment.empty()
    for pth in audio_paths:
        tmp_audio = tmp_audio + AudioSegment.from_file(file=pth) + AudioSegment.silent(0.25)
        if tmp_audio.duration_seconds > 40:
            break
    tmp_audio.export(os.path.join(pardir, "audio_out", "_tmp.mp3"), format="mp3")
    print(f"Warning: less than 30 seconds sample audio provided for {spkrname}. Audio may be of lower quality than expected.") if tmp_audio.duration_seconds < 30 else None

    pipe = Pipeline()
    spkr_encoder = pipe.extract_spk_emb(os.path.join(pardir, "audio_out",  "_tmp.mp3"))
    spkr_encoder.encoder = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir="~/.cache/speechbrain", run_opts={"device": "cpu"})

    torch.save(spkr_encoder, os.path.join(pardir,"audio_out", f"{spkrname}_profile.pt"))
    os.remove(os.path.join(pardir, "audio_out", "_tmp.mp3"))
    return spkr_encoder