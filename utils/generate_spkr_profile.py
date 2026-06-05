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

from voicebox_client import VoiceboxClient


def list_voicebox_profiles(base_url: str = "http://127.0.0.1:17493") -> list[dict]:
    """Return local Voicebox voice profiles usable by ChatParser."""
    return VoiceboxClient(base_url=base_url).list_profiles()


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
    args = parser.parse_args()
    print_voicebox_profiles(args.voicebox_url)


if __name__ == "__main__":
    main()
