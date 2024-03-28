from geopy.geocoders import Nominatim
from utils.default_values import default_spkr_profile, replacement_dict

import emoji
import nltk
import nltk.data
import spellchecker
import torch
import os
from collections import defaultdict

debug_dict = defaultdict(list)
added_splChkr_words = [
    "Dublin",
    "Ireland",
    "Andy",
]

def _check_spelling(line_text: str) -> bool:
    # spell check and inflect sentence properly for better TTS
    splChkr = spellchecker.SpellChecker('en')
    splChkr.word_frequency.load_words(added_splChkr_words)
    corwords = []
    sen_diag = []
    end_punctuation = (".","!","?",",")
    #punktChkr = nltk.PunktSentenceTokenizer(line_text)
    punktChkr = nltk.data.load('tokenizers/punkt/english.pickle')
    interim_line_text = []

    # just really ignore very short length sentences or sentences that use overly aggressive text-type abbreviations (e.g. 'u', 'lol', 'bday')
    if len(line_text) > 8:
        sentences = punktChkr.tokenize(text=line_text)
        for sentence in sentences:
            sen_diag.append(dict(nltk.pos_tag(sentence.split(" "))))
            #sen_diag = dict(nltk.pos_tag(sentence))
        for wrdpr in sen_diag:
            for i, word in enumerate(wrdpr):
                if word == "": continue
                isProperNoun = wrdpr[word] == "NNP"
                try:
                    word = replacement_dict[word]
                except:
                    # word is not in replacement_dict, just move on
                    pass
                if not isProperNoun and len(word) > 2:
                    corword = splChkr.correction(emoji.demojize(word).replace(":", " "))
                else:
                    corword = word
                corwords.append(corword if corword else word)
                if word in set(nltk.corpus.stopwords.words('english')): continue
                debug_dict[word].append(corword)
            #look back and add end punctuation
            if word.endswith(end_punctuation): corwords[-1] = corwords[-1] + word[-1]
            #else: corwords[i] += "."
            interim_line_text.append(" ".join(corwords))
            corwords.clear()

    return " ".join(interim_line_text)


def _get_spkr_profile(spkrname, spkr_profiles, audio_folder):
    spkr_profile = None
    if spkrname in spkr_profiles:
        spkr_profile = spkr_profiles[spkrname]
    elif spkrname is not None:
        spkr_profile_filename = os.path.join(audio_folder, "audio_out", spkrname + "_profile.pt")
        if os.path.exists(spkr_profile_filename) and os.path.isfile(spkr_profile_filename):
            spkr_profile = torch.load(os.path.join(audio_folder, "audio_out", spkrname + "_profile.pt"))
        else:
            spkr_profile = default_spkr_profile
        spkr_profiles[spkrname] = spkr_profile
    else:
        spkr_profile = default_spkr_profile
    if spkr_profile == "" or spkr_profile == None: spkr_profile = default_spkr_profile

    return spkr_profile


def _replace_location(line_list: list) -> str:
    '''Handle WhatsApp Location sent functionality.
    Input
        line_list: list - list of words in the sentence
    Output
        Boolean True if Location detected and word output manipulated
        Boolean False if location not detected and should continue processing
        Note: can use output of this function to skip futher checks (e.g. spell-check) and other costly string manipulations
    '''
    if len(line_list) < 1: return line_list
    if line_list[0].replace('\u200e', "") == "Location:":
        geo = Nominatim(user_agent="ChatParser")
        tmptxt = line_list[1]
        if tmptxt[-1] == '.': tmptxt = tmptxt[:-1]
        loc_info = geo.reverse(tmptxt.split("?q=")[1], exactly_one=True).raw['address']
        road = loc_info.get('road', '')
        city = loc_info.get('city', '')
        country = loc_info.get('country', '')
        country = country.split(" / ")[1] if len(country.split(" / ")) > 1 else country
        tourism = loc_info.get('tourism', '')
        line_list = [line_list[0], f"sent is in {road}, {city}, {country}{f' Tourist site: ({tourism})' if tourism else None}."]
    else:
        print('This should never print. If you see this message, please first see if a new version has been released. If not, please contact the developer for debugging assistance. Line:', line_list)
    return line_list