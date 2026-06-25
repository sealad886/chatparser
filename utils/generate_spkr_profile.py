#
# included file: generate_spkr_profile.py
# usage: standalone CLI utility
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

from __future__ import annotations

import argparse

from tqdm import tqdm

from voicebox_client import VoiceboxClient


def list_voicebox_profiles(base_url: str = "http://127.0.0.1:17493") -> list[dict]:
    """Return local Voicebox voice profiles usable by ChatParser."""
    return VoiceboxClient(base_url=base_url).list_profiles()


def create_voicebox_profile_from_files(
    name: str,
    audio_files: list[str],
    reference_texts: list[str],
    base_url: str = "http://127.0.0.1:17493",
    language: str = "en",
    description: str | None = None,
) -> dict:
    """Create a cloned Voicebox profile and upload local reference samples."""
    if len(audio_files) != len(reference_texts):
        raise ValueError("audio_files and reference_texts must have the same length")
    if not audio_files:
        raise ValueError("at least one audio file is required")

    client = VoiceboxClient(base_url=base_url)
    profile = client.create_voice_profile(name=name, language=language, description=description)
    profile_id = profile["id"]
    samples = [
        client.add_profile_sample(profile_id, audio_file, reference_text)
        for audio_file, reference_text in tqdm(
            zip(audio_files, reference_texts, strict=True),
            desc="Voice samples",
            total=len(audio_files),
            unit="sample",
            dynamic_ncols=True,
        )
    ]
    return {"profile": profile, "samples": samples}


def print_voicebox_profiles(base_url: str = "http://127.0.0.1:17493") -> None:
    profiles = list_voicebox_profiles(base_url)
    if not profiles:
        print("No Voicebox profiles found. Create or import one in Voicebox first.")
        return

    for profile in profiles:
        profile_id = profile.get("id", "")
        name = profile.get("name", "")
        language = profile.get("language", "")
        sample_count = profile.get("sample_count", 0)
        print(f"{profile_id}\t{name}\t{language}\tsamples={sample_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="generate_spkr_profile",
        description="List local Voicebox profiles for ChatParser voice cloning.",
    )
    parser.add_argument(
        "--voicebox-url",
        default="http://127.0.0.1:17493",
        help="Local Voicebox REST API base URL.",
    )
    parser.add_argument(
        "--create-profile",
        metavar="NAME",
        help="Create a cloned Voicebox profile from local sample audio files.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help="Local audio sample path. Repeat once per sample.",
    )
    parser.add_argument(
        "--reference-text",
        action="append",
        default=[],
        help="Transcript/reference text for the corresponding --sample. Repeat once per sample.",
    )
    parser.add_argument("--language", default="en", help="Voicebox profile language code.")
    parser.add_argument("--description", default=None, help="Optional Voicebox profile description.")
    args = parser.parse_args()
    if args.create_profile:
        result = create_voicebox_profile_from_files(
            name=args.create_profile,
            audio_files=args.sample,
            reference_texts=args.reference_text,
            base_url=args.voicebox_url,
            language=args.language,
            description=args.description,
        )
        profile = result["profile"]
        print(f"{profile.get('id', '')}\t{profile.get('name', '')}\tsamples={len(result['samples'])}")
    else:
        print_voicebox_profiles(args.voicebox_url)


if __name__ == "__main__":
    main()
