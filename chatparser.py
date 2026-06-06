#
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

import argparse
import os
import re
import math
from locale import getlocale
from utils.deprecated import _deal_with_numbers
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import nltk, nltk.data, string
import multiprocessing as mp
from multiprocessing import Process, Queue, cpu_count
import concurrent.futures
import inflect
import json
from dataclasses import dataclass

from utils.utils import _check_spelling, _replace_location, debug_dict
from voicebox_client import VoiceboxClient


@dataclass(frozen=True)
class ParsedWhatsAppLine:
    timestamp: datetime
    date_time_str: str
    speaker: str | None
    message: str
    source_format: str


@dataclass(frozen=True)
class WhatsAppAttachment:
    filename: str
    is_audio: bool

__MODEL_NAME = None
__VERBOSE = None
__PROGRESS_BAR = None
__FORCE_REDO = None
__NUM_WORKERS = None
__ENABLE_TIMINGS = None
q = None
p = None
pipe = None
workers = None
wk_pool = None
punktChkr = None
timings_measurement = None
def_sample_rate = None
cps = None
__VOICEBOX_CLIENT = None

VOICEBOX_TRANSCRIPTION_MODELS = {"base", "small", "medium", "large", "turbo"}
__VOICEBOX_URL = None
__VOICEBOX_MODEL = None
__VOICEBOX_PROFILE = None
__VOICEBOX_PROFILE_MAP = None
__VOICEBOX_LANGUAGE = None

def set_globals():
    global __MODEL_NAME
    global __VERBOSE
    global __PROGRESS_BAR
    global __FORCE_REDO
    global __NUM_WORKERS
    global __ENABLE_TIMINGS
    global q
    global p
    global pipe
    global workers
    global wk_pool
    global punktChkr
    global timings_measurement
    global def_sample_rate
    global cps
    global __VOICEBOX_CLIENT
    global __VOICEBOX_URL
    global __VOICEBOX_MODEL
    global __VOICEBOX_PROFILE
    global __VOICEBOX_PROFILE_MAP
    global __VOICEBOX_LANGUAGE

    # because apparently semafore locks with multiprocessing are broken, we have to do this:
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass
    os.makedirs(".locks/", exist_ok=True)
    os.environ["TMPDIR"] = os.path.abspath(".locks")
    __MODEL_NAME = "turbo"
    __VERBOSE = False
    __PROGRESS_BAR = False
    __FORCE_REDO = False
    __NUM_WORKERS = None
    __ENABLE_TIMINGS = False

    q = Queue()

    p = inflect.engine()
    pipe = None

    workers = []
    wk_pool = None

    punktChkr = nltk.tokenize.PunktSentenceTokenizer()
    timings_measurement = []

    def_sample_rate = 24000  # don't change this as this is what vocoder is set to, no way to change it currently -- voices get real fast / slow when bitrate changes without resampling, which is what would need to happen 
    cps = 15
    __VOICEBOX_URL = "http://127.0.0.1:17493"
    __VOICEBOX_MODEL = "turbo"
    __VOICEBOX_PROFILE = None
    __VOICEBOX_PROFILE_MAP = {}
    __VOICEBOX_LANGUAGE = "en"
    __VOICEBOX_CLIENT = VoiceboxClient(base_url=__VOICEBOX_URL)


def normalize_voicebox_transcription_model(model: str | None) -> str:
    value = (model or "turbo").strip()
    if value.startswith("whisper-"):
        value = value.removeprefix("whisper-")
    if value not in VOICEBOX_TRANSCRIPTION_MODELS:
        raise ValueError(f"Voicebox transcription model must be one of {sorted(VOICEBOX_TRANSCRIPTION_MODELS)}")
    return value

def remove_punctuation(input_str):
    # Create a translation table that maps each punctuation to None
    translator = str.maketrans('-', ' ', string.punctuation)
    
    # Use the translate method to remove any character in the string.punctuation
    no_punctuation_str = input_str.translate(translator)
    
    return no_punctuation_str

def replace_terms(input_str, replacement_dict) -> str:
    # Iterate over the dictionary items
    input_str = input_str.lower()
    for find_text, replace_with in replacement_dict.items():
        # Use a regular expression with word boundaries to replace the text
        input_str = re.sub(rf'\b{re.escape(find_text)}\b', replace_with, input_str)
    return input_str


def _parse_export_timestamp(value: str) -> datetime | None:
    normalized = value.strip().replace("\u202f", " ").replace("\u00a0", " ")
    formats = [
        "%d/%m/%Y, %H:%M:%S",
        "%d/%m/%Y, %H:%M",
        "%d/%m/%y, %H:%M:%S",
        "%d/%m/%y, %H:%M",
        "%d/%m/%Y, %I:%M:%S %p",
        "%d/%m/%Y, %I:%M %p",
        "%d/%m/%y, %I:%M:%S %p",
        "%d/%m/%y, %I:%M %p",
        "%m/%d/%Y, %I:%M:%S %p",
        "%m/%d/%Y, %I:%M %p",
        "%m/%d/%y, %I:%M:%S %p",
        "%m/%d/%y, %I:%M %p",
        "%m/%d/%Y, %H:%M:%S",
        "%m/%d/%Y, %H:%M",
        "%m/%d/%y, %H:%M:%S",
        "%m/%d/%y, %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def parse_whatsapp_line(line: str) -> ParsedWhatsAppLine | None:
    cleaned = line.strip().replace("\u200e", "")
    if not cleaned:
        return None

    ios_match = re.match(r"^\[(?P<timestamp>[^\]]+)\]\s*(?P<body>.*)$", cleaned)
    if ios_match:
        timestamp = _parse_export_timestamp(ios_match.group("timestamp"))
        if timestamp is None:
            return None
        body = ios_match.group("body")
        speaker, message = _split_speaker_message(body)
        return ParsedWhatsAppLine(
            timestamp=timestamp,
            date_time_str=timestamp.strftime("%d/%m/%Y, %H:%M:%S"),
            speaker=speaker,
            message=message,
            source_format="ios",
        )

    android_match = re.match(r"^(?P<timestamp>.+?)\s+-\s+(?P<body>.*)$", cleaned)
    if android_match:
        timestamp = _parse_export_timestamp(android_match.group("timestamp"))
        if timestamp is None:
            return None
        speaker, message = _split_speaker_message(android_match.group("body"))
        return ParsedWhatsAppLine(
            timestamp=timestamp,
            date_time_str=timestamp.strftime("%d/%m/%Y, %H:%M:%S"),
            speaker=speaker,
            message=message,
            source_format="android",
        )
    return None


def _split_speaker_message(body: str) -> tuple[str | None, str]:
    if ": " not in body:
        return None, body.strip()
    speaker, message = body.split(": ", 1)
    speaker = speaker.strip() or None
    return speaker, message.strip()


def find_whatsapp_attachment(message: str) -> WhatsAppAttachment | None:
    ios_match = re.search(r"<attached:\s*(?P<filename>.+?)>", message)
    if ios_match:
        filename = ios_match.group("filename").strip()
        return WhatsAppAttachment(filename=filename, is_audio=_is_audio_attachment(filename))

    android_match = re.search(r"(?P<filename>\S+\.(?:opus|ogg|m4a|mp3|wav|aac|flac|webm))\s+\(file attached\)", message, re.IGNORECASE)
    if android_match:
        filename = android_match.group("filename").strip()
        return WhatsAppAttachment(filename=filename, is_audio=_is_audio_attachment(filename))
    return None


def _is_audio_attachment(filename: str) -> bool:
    upper = filename.upper()
    suffix = Path(filename).suffix.lower()
    return "AUDIO" in upper or upper.startswith(("AUD-", "PTT-")) or suffix in {".opus", ".ogg", ".m4a", ".mp3", ".wav", ".aac", ".flac", ".webm"}

'''
def spellcheck(i: int, line: str) -> str:
    try:
        splitted = line.split("] ",1)
        if len(splitted) > 1:
            return i, splitted[0] + tool.correct(splitted[1])
        else: return i, splitted[0]
    except:
        print(f"Unhandled error on line {i}")
'''

def worker(queue):
    while True:
        task = queue.get()  # Get a task from the queue
        if task is None:  # None is the signal to stop
            break
        # Unpack the task tuple
        func, args = task
        func(*args)  # Execute the task

def process_chat_file_by_type(chat_file: str, audio_folder: str, model_prompt: str, file_out: list, to_type: str = "text", wkrs: list|None = None) -> None:
    if __ENABLE_TIMINGS: this_time = []
    if __ENABLE_TIMINGS: tt = this_time.append
    if __ENABLE_TIMINGS: now = datetime.now
    if __ENABLE_TIMINGS: tt(['start_processing_datetime', now()])
    # Process a chat f file and transcribe any audio messages found. 
    processed_file = f"{os.path.splitext(chat_file)[0]}-aud2txt{os.path.splitext(chat_file)[1]}"

    #######################################################
    # This section enables multiprocessing support
    # and better handles cases where large data transformations
    # are required outside of the AI-driven main thread
    # i.e. converting file types
    #######################################################
    global q
    tasks = q
    if tasks == None and __VERBOSE: print("Multithreading options not passed as arguments. Defaulting to a single")
    global pipe
    global wk_pool
    if __NUM_WORKERS is not None and __NUM_WORKERS > 0: num_workers = __NUM_WORKERS
    else: num_workers = round((cpu_count() + 1) / 2)    # this way ensures that this is never 0, and ends up allocating slightly more than half the cpus
    if wkrs is None: 
        global workers
        wkrs = workers
    
    # Start worker processes
    for i in range(0, num_workers):
        p = Process(target=worker, args=[tasks])
        wkrs.append(p)
        p.start()

    # End multiprocessing section
    #######################################################
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as wk_pool:
    
        if __ENABLE_TIMINGS: tt(['source_file_read_start', now()])
        with open(chat_file, "rb") as f:
            file_in_precorrected = f.readlines()
            fic = len(file_in_precorrected)
        file_in = file_in_precorrected.copy()
        if __ENABLE_TIMINGS: tt(['source_file_read_end', now()])
        
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
        
        spkr_profiles = {}
        
        model_dir = __VOICEBOX_MODEL

        prev_datetime = None
        prepend_text = None
        append_text = None
        prev_spkr = None
        cur_spkr_lines, prev_spr_lines = [], []

        chat_num = 0 # will use this to account for multi-lines chats

        if __ENABLE_TIMINGS: tt(['lines_loop_start',now()])
        for i in tqdm(range(0, fic), desc = os.path.split(audio_folder)[1], total=fic, miniters = 1, smoothing = 0.1, dynamic_ncols=True, disable=not __PROGRESS_BAR):
            if __ENABLE_TIMINGS: tt([f'{i}_loop_start_chat_{chat_num}',now()])
            firstline = i==0
            lastline = i==fic-1
            # Check if the line contains an audio attachment
            line = file_in[i].decode().strip()
            print(f"{line}") if __VERBOSE else None
            line = re.sub(r'[^\x20-\x7E\u00A0-\uD7FF\uF900-\uFFFF\u200e]', '', line)
            if len(line) < 1: continue
            line = line.replace("\u200e","")
            parsed = parse_whatsapp_line(line)
            if parsed is None:
                if file_out:
                    file_out[-1] = file_out[-1].rstrip("\n") + f"\n{line}\n"
                else:
                    file_out.append(line.strip() + "\n")
                continue

            date_time_str = parsed.date_time_str
            cur_datetime = parsed.timestamp
            spkrname = parsed.speaker
            line = parsed.message

            attachment = find_whatsapp_attachment(line)
            if attachment:
                #re.search(r"\[(.*?)\]", line).group(1)
                #spkrname = re.search(r'^\[(\w+)\W', line).group(1)
                if attachment.is_audio and to_type == "text":
                    transcribe_audio_line(audio_folder, attachment, model_dir, model_prompt, date_time_str, spkrname, file_out, chat_num)
                elif attachment.is_audio and to_type == "audio":
                    #
                    # this is adding too pool
                    #
                    #ctr=chat_num
                    #wkrs.append(wk_pool.submit(move_audio_file, line, audio_folder, ctr))
                    #move_audio_file(line, audio_folder, chat_num)
                    tasks.put((_move_audio_file_mt, (attachment, audio_folder, chat_num, __VERBOSE)))
                    print(f"Task queued: {line}") if __VERBOSE else None
                else:
                    print(f"Need to move: {line}") if __VERBOSE else None
                    file_out.append(line.strip() + "\n")
            elif to_type == "audio": 
                if prev_datetime is not None:
                    if (cur_datetime - prev_datetime) > timedelta(hours=12):
                        if (cur_datetime - prev_datetime) > timedelta(days=1):
                            prepend_text = f"{(cur_datetime-prev_datetime).days} days have passed since the last message."
                            prepend_text = f"{round((cur_datetime - prev_datetime).total_seconds() / 3600)} hours have passed since the last message"
                            cur_spkr_lines.append(prepend_text)
                # to audio by line, generate each line individually
                os.makedirs(os.path.join(audio_folder, "audio_out"), exist_ok=True)

                if prev_spkr is None:
                    prev_spkr = spkrname

                # actually generate the audio file for lines
                if prev_spkr != spkrname:
                    spkr_multi_lines = ". ".join(prev_spr_lines)
                    #
                    # this is adding to pool
                    #
                    #wkrs.append(wk_pool.submit(line_to_audio, spkr_multi_lines, prev_spkr, date_time_str, audio_folder, chat_num, spkr_profiles))
                    if spkr_multi_lines:
                        line_to_audio(spkr_multi_lines, prev_spkr or "", date_time_str, audio_folder, chat_num, spkr_profiles)
                    prev_spkr = spkrname
                    prev_spr_lines.clear()
                    for ln in cur_spkr_lines: prev_spr_lines.append(ln)
                    cur_spkr_lines.clear()
                    print(f"Done making audio for chat {chat_num}\n") if __VERBOSE else None
                prev_spr_lines.append(line)
                if lastline:
                    spkr_multi_lines = ". ".join(prev_spr_lines)
                    if spkr_multi_lines:
                        line_to_audio(spkr_multi_lines, prev_spkr or spkrname or "", date_time_str, audio_folder, chat_num, spkr_profiles)
                    prev_spr_lines.clear()

                prepend_text = None
            else:
                print(f"Don't know what to do with this line {i}")
            prev_datetime = cur_datetime
            if __ENABLE_TIMINGS: tt([f'{i}_loop_end',now()])
            chat_num+=1
        if __ENABLE_TIMINGS: tt(['lines_loop_end',now()])

    #######################################################
    # Wrap up section. Close subprocesses and 
        # Signal workers to stop
    for _ in wkrs:
        tasks.put(None)
    # Wait for all workers to finish
    for w in wkrs:
        w.join()
    if __ENABLE_TIMINGS: tt(['cleanup_end_start',now()])
    cleanup_end(processed_file, file_out)
    if __ENABLE_TIMINGS: tt(['cleanup_end_end', now()])
    if __ENABLE_TIMINGS: timings_measurement.append([os.path.abspath(os.path.join(chat_file, os.pardir)), this_time])

def _move_audio_file_mt(attachment, audio_folder, ctr, __VERBOSE) -> None:
    '''Multithread wrapper for move_audio_file()
    '''
    move_audio_file(attachment, audio_folder, ctr)

def move_audio_file(attachment, audio_folder, ctr) -> None:
    from pydub import AudioSegment

    filename = attachment.filename if isinstance(attachment, WhatsAppAttachment) else re.search(r"<attached:.+?>", attachment).group(0)[10:-1].strip()
    audio_file = os.path.join(audio_folder, filename)
    
    target_file = os.path.join(audio_folder, "audio_out", filename)[:-4]+"mp3"
    print(f"Moving to: {target_file}") if __VERBOSE else None
    if os.path.exists(target_file): 
        print(f"Already exists: {target_file}") if __VERBOSE else None
    tmpaudio = AudioSegment.empty()
    tmpaudio += AudioSegment.from_file(audio_file)
    with open(target_file, "wb") as a:
        tmpaudio.export(a, format="mp3")

def line_to_audio(line, spkrname, date_time_str, audio_folder, ctr: str, spkr_profiles: dict = {}):
    day, mon, yr = date_time_str.split(",")[0].split("/")
    os.makedirs(os.path.join(audio_folder, "audio_out"), exist_ok=True)
    audio_out_file = os.path.join(audio_folder, "audio_out", f"{str(ctr+1).zfill(8)}-AUDIO-{yr}-{mon}-{day}-auto-generated.mp3")
    if os.path.exists(audio_out_file) and os.path.isfile(audio_out_file):
        print(f"File exists: {audio_out_file}") if __VERBOSE else None
        return audio_out_file

    global __VOICEBOX_CLIENT
    global __VOICEBOX_PROFILE
    global __VOICEBOX_PROFILE_MAP
    global __VOICEBOX_LANGUAGE
    if __VOICEBOX_CLIENT is None:
        __VOICEBOX_CLIENT = VoiceboxClient(base_url=__VOICEBOX_URL or "http://127.0.0.1:17493")
    profile_id = (__VOICEBOX_PROFILE_MAP or {}).get(spkrname) or __VOICEBOX_PROFILE

    print(f"Audio out file: {audio_out_file}") if __VERBOSE else None
    line_list = line.strip().split(" ")

    # deal with locations being sent
    locreplaced = False
    uline_list = [_replace_location(sent) for sent in line_list]
    if uline_list != line_list:
        locreplaced = True
        line_list = uline_list.copy()

    print(f"  Updated location in chat no. {ctr+1}") if locreplaced else None

    line_text = " ".join(line_list)
    if line_text == "": print("No line text.") if __VERBOSE else None; return
    #else: None

    # substitute numbers for words
    # line_text = _deal_with_numbers(line_text) if not locreplaced else line_text

    if not locreplaced:
        # spell check and inflect the whole sentence
        line_text = line_text.replace("`", "'")
        line_text = line_text.replace("O' ", "O'")
        try:
            line_text = _check_spelling(line_text)
        except LookupError:
            print("NLTK data unavailable; using uncorrected text for Voicebox generation.") if __VERBOSE else None
    

    # split a long text into pieces
    sent_cnt = 1
    sentence = line_text
    print(f"Processing chat {ctr+1}...") if __VERBOSE else None
    if len(sentence) < 8: sentence = f"Line was too short: {sentence}"
    print(f"  Sentence: {spkrname}: {sentence}\n")
    sentence = sentence if len(sentence.split(" ")) > 1 else sentence + " " + sentence
    __VOICEBOX_CLIENT.generate_speech(
        sentence,
        Path(audio_out_file),
        profile_id=profile_id,
        language=__VOICEBOX_LANGUAGE,
    )
    sent_cnt += 1
    
    '''
    consolidated_audio = AudioSegment.empty()
    for i in range(1, sent_cnt):
        tmp_file = f"{audio_folder}/audio_tmp/line{ctr}_sent{i}.mp3"
        consolidated_audio += AudioSegment.from_file(tmp_file) + AudioSegment.silent(250)
        os.remove(tmp_file)

    with open(audio_out_file, "wb") as a:
        out_path = consolidated_audio.export(a, format="mp3")
        print(f"Saved to {out_path}") if __VERBOSE else None
    '''
    return audio_out_file

def transcribe_audio_line(audio_folder, match, model_dir, model_prompt, date_time_str, spkrname, file_out, i):
    # Extract the audio file name from the line
    filename = match.filename if isinstance(match, WhatsAppAttachment) else match.group(0)[10:-1].strip()
    audio_file = os.path.join(audio_folder, filename)

    global __VOICEBOX_CLIENT
    global __VOICEBOX_MODEL
    if __VOICEBOX_CLIENT is None:
        __VOICEBOX_CLIENT = VoiceboxClient(base_url=__VOICEBOX_URL or "http://127.0.0.1:17493")

    result = __VOICEBOX_CLIENT.transcribe_audio(
        audio_file,
        model=normalize_voicebox_transcription_model(__VOICEBOX_MODEL or model_dir),
    )
    language = result.language or "unk"

    # build the transcription line now
    transcription = f"[{date_time_str}] {spkrname}: [Transcribed]: {result.text} ({language}) [File: {os.path.basename(audio_file)}]\n"

    # Print the result to stdout
    print(f"Transcription (line {i}):\n{transcription}") if __VERBOSE else None

    file_out.append(transcription)
    return transcription


def cleanup_end(processed_file: str, file_out: list) -> None:
    with open(processed_file, "w") as f:
        for line in file_out:
            f.write(line)

def parse_voicebox_profile_map(items: list[str] | None, json_file: str | None = None) -> dict[str, str]:
    profile_map: dict[str, str] = {}
    if json_file:
        with open(json_file, "r") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError("--voicebox-profile-map-file must contain a JSON object")
        profile_map.update({str(k): str(v) for k, v in loaded.items()})
    for item in items or []:
        if "=" not in item:
            raise ValueError("--voicebox-profile-map entries must use Speaker=profile-id")
        speaker, profile_id = item.split("=", 1)
        speaker = speaker.strip()
        profile_id = profile_id.strip()
        if not speaker or not profile_id:
            raise ValueError("--voicebox-profile-map entries must include both speaker and profile id")
        profile_map[speaker] = profile_id
    return profile_map

def process_directories(directory: str, model_input: str, to_type: str, num_workers: int = None) -> None:
    # Process all _chat.txt files in a directory and its subdirectories 
    if os.path.isfile(directory):
        print(f"{directory} is not a valid directory.")
        return
    
    chat_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "_chat.txt":
                chat_paths.append((root, file))

    for root, file in tqdm(chat_paths, desc="WhatsApp exports", total=len(chat_paths), dynamic_ncols=True, disable=not __PROGRESS_BAR):
        file_out = []
        try:
            chat_file = os.path.join(root, file)
            if num_workers:
                print("Multi-threading support not implemented. Defaulting to normal process")
                # process_chat_file_by_type_mt(chat_file, root, model_input, file_out, to_type=to_type, num_workers=num_workers)
                process_chat_file_by_type(chat_file, root, model_input, file_out, to_type=to_type)
            else:
                process_chat_file_by_type(chat_file, root, model_input, file_out, to_type=to_type)
        except KeyboardInterrupt:
            error_out = [f"Transcription interrupted at: {datetime.now()}"]
            for line in file_out:
                error_out.append(line)
            cleanup_end(os.path.join(root, "_transcription_interrupted.txt"), error_out)

# a simple function to get parser to do some of the input validation for us
def __parser_allowed_dir(input):
    if os.path.isdir(input) and os.access(input, os.R_OK):
        return input
    else:
        AssertionError("ERROR: The given argument in --input_file is not a directory or is not accessible:", input)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="chatparser", 
        description="Process WhatsApp Messenger exported chats and transcribe audio messages, creating one text file.")
    parser.add_argument(
        "-m",
        "--model",
        type=str, 
        default="turbo",
        choices=[
            "base",
            "small",
            "medium",
            "large",
            "turbo",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large",
            "whisper-turbo",
        ],
        help="The Voicebox transcription model to request.")
    parser.add_argument(
        "--voicebox-url",
        type=str,
        default="http://127.0.0.1:17493",
        help="Local Voicebox REST API base URL.")
    parser.add_argument(
        "--voicebox-profile",
        type=str,
        default=None,
        help="Fallback Voicebox voice profile id for generated speech.")
    parser.add_argument(
        "--voicebox-profile-map",
        action="append",
        default=None,
        help="Speaker-to-Voicebox profile mapping as 'Speaker Name=profile-id'. Can be used multiple times.")
    parser.add_argument(
        "--voicebox-profile-map-file",
        type=str,
        default=None,
        help="JSON object mapping WhatsApp speaker display names to Voicebox profile ids.")
    parser.add_argument(
        "--voicebox-language",
        type=str,
        default="en",
        help="Language code to send to Voicebox for generated speech.")
    parser.add_argument(
        "-i", 
        "--input-directory", 
        type=__parser_allowed_dir,
        action="append",
        help="Directory containing the _chat.txt files or subdirectories with _chat.txt files. This can be used multiple times to input multiple directories.")
    parser.add_argument(
        "-v", 
        "--verbose", 
        dest="VERBOSE",
        action="store_true", 
        help="There are a couple spots where sometimes there's some extra info sent to stdout.")
    parser.add_argument(
        "-p", 
        "--prompt-file", 
        type=str, 
        default="chatparser.model_prompt.txt",
        help="List a text file which contains a string to give to the whisper.transformer model as a initial prompt. The default file and location is ./chatparser.model_prompt.txt. If you set this to a file that returns an empty string, chatparser will still attempt to give the model some context by extracting your locale. If locale can not be determined, a default of enUS is used. Text is: 'The locale of the user owning this audio file is {locale} so assume a higher likelihood that speech is in the language and accent common to that locale.'")
    parser.add_argument(
        "-P", 
        "--progress-bar",
        dest="PROGRESS_BAR", 
        action="store_true",
        help="If set, displays a progress bar for each directory that is being converted.")
    parser.add_argument(
        "--force-redo",
        dest="FORCE_REDO",
        action="store_true",
        help="Should already transcribed chats be overwritten? If False, the name of the skipped file will be printed to stdout.")
    parser.add_argument(
        "--to-type",
        dest="to_type",
        type=str,
        default="text",
        help="What format will be output for the given chat(s)?",
        choices=["audio", "text"],
    )
    parser.add_argument(
        "--num-workers",
        dest="NUM_WORKERS",
        type = int,
        default = None,
        help = "Set to any number to enable the multi-threaded processing. Note that setting this to 1 will enable the processing on a single processor, thereby skipping some elements that only supported in single-threaded processing (e.g. comparing timestamp between messages). Setting to a negative value will default to single-thread.",
    )
    parser.add_argument(
        "--enable-timings",
        dest="ENABLE_TIMINGS",
        action="store_true",
        help="When set, this will save timings information for all files in a global variable that is printed to stdout regardless of __VERBOSE settings."
    )
    parser.add_argument(
        "-G", "--generate_tensor",
        dest="generate_tensor",
        action="store_true"
    )
    args = parser.parse_args()

    set_globals()

    __FORCE_REDO, __VERBOSE, __PROGRESS_BAR, to_type, __NUM_WORKERS, __ENABLE_TIMINGS = args.FORCE_REDO, args.VERBOSE, args.PROGRESS_BAR, args.to_type, args.NUM_WORKERS, args.ENABLE_TIMINGS

    model_input = ""
    if not os.path.isfile(args.prompt_file):
        print(f"Invalid PROMPT_FILE path: {args.prompt_file} is not a file")
    elif not os.access(args.prompt_file, os.R_OK):
        print(f"Invalid PROMPT_FILE path: user does not have access to {args.prompt_file}")
    else:
        # Read the contents of the input file into a string variable
        with open(args.prompt_file, 'r') as f:
            model_input = f.read()
            print(f"The prompt-file imported as: {model_input}") if __VERBOSE else None
    # Get the locale of this process, or if not set assume US English
    # and give that as a default prompt to the translator
    if model_input == "":
        locale = getlocale()[0]
        locale = "enUS" if locale == "" else None
        model_input = f"The locale of the user owning this audio file is {locale}, so assume a higher likelihood that speech is in the language and accent common to that locale."

    __MODEL_NAME = args.model
    __VOICEBOX_URL = args.voicebox_url
    __VOICEBOX_MODEL = args.model
    __VOICEBOX_PROFILE = args.voicebox_profile
    __VOICEBOX_PROFILE_MAP = parse_voicebox_profile_map(args.voicebox_profile_map, args.voicebox_profile_map_file)
    __VOICEBOX_LANGUAGE = args.voicebox_language
    __VOICEBOX_CLIENT = VoiceboxClient(base_url=__VOICEBOX_URL)
    __NUM_WORKERS = __NUM_WORKERS if __NUM_WORKERS is not None and __NUM_WORKERS > 1 else None

    # this for-loop handles multiple --input-directory uses in the CLI
    # process_directories handles recursion on its own
    for dir in args.input_directory:
        try:
            process_directories(dir, model_input, to_type, num_workers=__NUM_WORKERS)
        finally:
            with open(os.path.join(dir, "_debug_dict.json"), 'a') as f:
                json.dump(debug_dict,f)
    if __ENABLE_TIMINGS: print(json.dumps(timings_measurement))
