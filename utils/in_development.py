from datetime import datetime
from typing import List
import re, os

from chatparser import line_to_audio, move_audio_file, transcribe_audio_line


def process_chat_file_by_type_mt(chat_file: str, audio_folder: str, model_prompt: str, file_out: List, to_type: str = "text", num_workers: int = 1) -> None:
    return None
    """
    # Process a chat f file and transcribe any audio messages found. 
    processed_file = f"{os.path.splitext(chat_file)[0]}-aud2txt{os.path.splitext(chat_file)[1]}"

    with open(chat_file, "rb") as f:
        file_in = f.readlines()
        fic = len(file_in)
    with open(processed_file, 'wb') as f:
        None
    file_out = [None] * fic

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

    if not __FORCE_REDO and os.path.isfile(processed_file):
        print(f"Quitting early: {chat_file} has already been transcribed.")
        return
    '''

    if to_type == "audio":
        spkr_profiles = {}

    model_dir = "mlx-community/whisper-"
    model_dir = model_dir + __MODEL_NAME + "-mlx-4bit"
    _ = whisper.load_models.load_model(model_dir, mx.float16)
    spkr_profiles = {}
    processes = []
    # Process lines using ThreadPoolExecutor
    with Pool(processes=num_workers) as pool:  # Adjust based on your system's capabilities
        # Submit all lines as separate tasks
        print("Creating all \'Futures\' tasks...")
        argslist = []
        for i in file_in:
            argslist.append([i, file_in[i], audio_folder, model_prompt, to_type, spkr_profiles, model_dir])
        results = pool.map(process_line_mt_pool, argslist, chunksize=10)


        print("Starting tqdm loop...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing lines"):
            i, processed_line = future.result()
            file_out[i] = processed_line
            # If order matters, you could handle ordering here based on the original line index or sort after all tasks complete.

    with open(processed_file, "w") as file:
        file.writelines(file_out)
"""


def process_line_mt(i:int, line:str, audio_folder: str, model_prompt: str, to_type: str = "text", spkr_profiles: dict = {}, model_dir: str = "mlx-community/whisper-small-mlx-4bit"):
    prev_datetime = None
    prepend_text = None
    append_text = None
    line=line.decode('utf-8')

    #print(f"{line}") if __VERBOSE else None
    line = re.sub(r'[^\x20-\x7E\u00A0-\uD7FF\uF900-\uFFFF\u200e]', '', line)
    date_time_str, spkrname = line.split("]", 1)
    date_time_str = date_time_str.split("[", 1)[1]
    cur_datetime = datetime.strptime(date_time_str, "%d/%m/%Y, %H:%M:%S")

    spkrname = spkrname.split(":")[0].split(" ")[1:]
    match_audio = re.search(r"<attached: \d+-AUDIO-.+>", line)
    match = re.search(r"<attached:.+>", line)
    if match:
        #re.search(r"\[(.*?)\]", line).group(1)
        #spkrname = re.search(r'^\[(\w+)\W', line).group(1)
        if match_audio and to_type == "text":
            transcription = transcribe_audio_line(audio_folder, match, model_dir, model_prompt, date_time_str, spkrname, file_out=None)
        elif match_audio and to_type == "audio":
            print(f"Trying to move: {line}")
            move_audio_file(line, audio_folder=audio_folder, ctr=i)
            transcription = line
        else:
            print(f"Need to move: {line}")
            # file_out.append(line.strip() + "\n")
            transcription = line
    elif to_type == "audio":
        '''
        # This won't work with MT...
        if prev_datetime is not None:
            if (cur_datetime - prev_datetime) > timedelta(hours=12):
                prepend_text = f"{round((cur_datetime - prev_datetime).total_seconds() / 3600)} hours have passed since the last message. "
        '''
        # to audio by line, generate each line individually
        os.makedirs(os.path.join(audio_folder, "audio_out"), exist_ok=True)
        os.makedirs(os.path.join(audio_folder, "audio_tmp"), exist_ok=True)
        audio_file_out = line_to_audio(line, spkrname, date_time_str, audio_folder, ctr=i, spkr_profiles=spkr_profiles, prepend_text=prepend_text, append_text=append_text)
       #print(f"Created audio file: {audio_file_out}\n") if __VERBOSE else None
        prepend_text = None
        append_text = None
        transcription = line
    else:
        print(f"Don't know what to do with this line {i}")
        transcription = line
    prev_datetime = cur_datetime

    return i, transcription