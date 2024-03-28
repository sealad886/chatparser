#
# included file: generate_spkr_profile.py
# usage: private library of ChatParser
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
from typing import List
import os


def process_chat_file(chat_file: str, audio_folder: str, model_prompt: str, file_out: List) -> None:
    print("process_chat_file is deprecated. Please use process_chat_file_by_type.")
    return
    # support event timings 
    if __ENABLE_TIMINGS: this_time = []
    if __ENABLE_TIMINGS: tt = this_time.append
    if __ENABLE_TIMINGS: now = datetime.now
    if __ENABLE_TIMINGS: tt(['start_processing_datetime', now()])

    # Process a chat f file and transcribe any audio messages found. 
    processed_file = f"{os.path.splitext(chat_file)[0]}-aud2txt{os.path.splitext(chat_file)[1]}"

    if __ENABLE_TIMINGS: tt(['source_file_read_start', now()])
    with open(chat_file, "rb") as f:
        file_in = f.readlines()
        fic = len(file_in)
    if __ENABLE_TIMINGS: tt(['source_file_read_stop', now()])

    '''
    # this won't work until I can refactor this to save the output as it
    # goes along instead of building a big array and then looping through
    # that all at the end:
    #       file_out.append(transcription)
    skip_ahead = False
    foc = 0
    file_out = []
    if os.path.isfile(processed_file) and (not __FORCE_REDO):
        with open(processed_file, "rb") as f:
            file_out = f.readlines()
            foc = len(file_out)
        skip_ahead = True
    '''
    if not __FORCE_REDO and os.path.isfile(processed_file):
        print(f"Quitting early: {chat_file} has already been transcribed.")
        return

    if __ENABLE_TIMINGS: tt('resume_check_start',now())
    resume_file = os.path.join(audio_folder, "_transcription_interrupted.txt")
    resumei = 0
    if os.path.isfile(resume_file):
        if not __FORCE_REDO:
            resumei += 1
            with open(resume_file, "r") as f:
                for line in f:
                    file_out.append(line)
                    resumei += 1
        try:
            os.remove(resume_file)
        except RuntimeError as e:
            print(f"Error deleting _transcription_interrupted.txt: {e}")
    if __ENABLE_TIMINGS: tt('resume_check_end',now())

    if __ENABLE_TIMINGS: tt('load_model_from_huggingface_start',now())
    model_dir = "mlx-community/whisper-"
    model_dir = model_dir + __MODEL_NAME + "-mlx-4bit"
    _ = whisper.load_models.load_model(model_dir, mx.float16)
    if __ENABLE_TIMINGS: tt('load_model_from_huggingface_end',now())

    if __ENABLE_TIMINGS: tt('lines_loop_start', now())
    for i in tqdm(range(0, fic), desc = os.path.split(audio_folder)[1], total=fic, miniters = 1, smoothing = 0.1, dynamic_ncols=True, disable=not __PROGRESS_BAR):
        if __ENABLE_TIMINGS: tt(f'{i}_loop_start',now())
        if resumei:
            resumei -= 1
            continue
        # Check if the line contains an audio attachment
        line = file_in[i].strip().decode()
        print(f"{line}") if __VERBOSE else None
        match = re.search(r"<attached: \d+-AUDIO-.+>", line)
        if match:
            line = re.sub(r'[^\x20-\x7E\u00A0-\uD7FF\uF900-\uFFFF\u200e]', '', line)
            date_time_str, spkrname = line.split("]", 1)
            date_time_str = date_time_str[2:]
            spkrname = spkrname.split(":")[0].split(" ")[1]
            #re.search(r"\[(.*?)\]", line).group(1)
            #spkrname = re.search(r'^\[(\w+)\W', line).group(1)
            # Extract the audio file name from the line
            audio_file = os.path.join(audio_folder, match.group(0)[10:-1].strip())

            # Transcribe the audio file
            if __ENABLE_TIMINGS: tt([f'{i}_handle_audio_start', now()])
            result = whisper.transcribe(audio_file, path_or_hf_repo=model_dir, initial_prompt = model_prompt, verbose=__VERBOSE if __VERBOSE else None)
            if __ENABLE_TIMINGS: tt([f'{i}_handle_audio_stop', now()])

            avg_logprob_total = 0
            line_logprob = 0
            try:
                for seg in range(0, len(result['segments'])):
                    avg_logprob_total += result['segments'][seg]['avg_logprob']
                line_logprob = math.exp(avg_logprob_total / len(result['segments']))
            except:
                print("Unhandled error calculating percent confidence.") if __VERBOSE else None

            if line_logprob == 0:
                line_logprob = "unk"
            else:
                line_logprob = line_logprob * 100
                line_logprob = str(line_logprob)[0:5] + "%"

            # build the transcription line now
            transcription = f"[{date_time_str}] {spkrname}: [Transcribed]:{result['text']} ({result['language']}) (conf: {line_logprob}) [File: {os.path.basename(audio_file)}]\n"

            file_out.append(transcription)

            # Print the result to stdout
            print(f"Transcription (line {i}):\n{transcription}") if __VERBOSE else None
        else:
            file_out.append(line.strip() + "\n")
        if __ENABLE_TIMINGS: tt(f'{i}_loop_end', now())

    if __ENABLE_TIMINGS: tt('lines_loop_end',now())
    cleanup_end(processed_file, file_out)

    this_time.append(['stop_processing_datetime', datetime.now()])
    timings_measurement.append([os.pardir(os.path.abspath(chat_file)), this_time])


def _deal_with_numbers(line_text: str) -> str:
    '''Handle when numbers need to be dates or  times for TTS only.
    Note that this doesn't work right now, and just outputs a file to HOME directory.
    Input:
        Full string for TTS interpretation.
    Output:
        Substituted full string for TTS.
    '''
    # for now, put these all into a file and figure out what needs to happen with them
    with open(os.path.abspath("~/deal_with_numbers.txt"), 'w') as f:
        f.write(line_text)
    return line_text

    # define regex's
    timereg1 = r"\d+:\d\d"
    timereg2 = r"\d+\.\d\d *[apAP][mM]"
    gennumreg = r"\d+"
    if timematch1 := re.match(timereg1, line_text): return timematch1
    elif timematch2 := re.match(timereg2, line_text): return timematch2
    elif nummatch := re.match(gennumreg, line_text): return nummatch
    else: return line_text