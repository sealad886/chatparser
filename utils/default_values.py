#
# included file: default_values.py
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
import torch


replacement_dict = {
            'ty': "thank you",
            'brb': "be right back",
            'cts': "cunts",
            '  ': " ",
            'u': "you",
            'bb': "bareback",
            'bday': "birthday",
            'bj': "blow job",
            'bs': "bull shit",
            'btm': "bottom",
            'btw': "by the way",
            'didnt': "didn't",
            'dont': "don't",
            'dr': "doctor",
            'etc': "et cetera",
            'ffs': "for fuck sake",
            'fml': "fuck my life",
            'fs': "fuck sake",
            'fyi': "eff why eye",
            'isnt': "isn't",
            'ive': "I've",
            'k': "okay",
            'lol': "ell oh ell",
            'mins': "minutes",
            'np': "no problem",
            'num': "number",
            'ok': "okay",
            'ur': "your",
            'urs': "yours",
            'urself': "yourself",
            'vers': "versatile",
            'wasnt': "wasn't",
            'werent': "weren't",
            'whos': "whose",
            'xx': "hugs and kisses",
            'y': "why",
            'yday': "yesterday",
            'youd': "you'd",
            'youll': "you'll",
            'youre': "you're",
            'youve': "you've"
            }
default_spkr_profile = torch.tensor(
       [  3.3030, -26.8433,  11.5359, -11.3118,  50.6838,  -6.7650,  11.1486,
        -35.3899,  -5.8200, -16.4554,  31.1431,  67.7724, -19.2651, -15.6688,
        -11.0584,  29.1230,   9.5967, -12.0261,  -1.2262,   4.1948,  27.0633,
        -35.0959, -18.1568, -11.7088,   0.9919,  15.8290, -19.1572, -10.9074,
         -3.1697, -13.7624,  22.8617,  42.4551, -23.7573, -13.2931,   5.2166,
         18.3597,  13.6541,   8.7048,  -3.1167,   0.5206,  -8.0101, -17.2544,
        -15.1042,   7.0201,  19.8669,  10.5034,  53.6193, -26.8914,  21.5167,
          7.1972,  -5.0493,  15.3830,  -8.4843, -26.6469, -21.4733,   0.7113,
        -23.1216,  30.7590, -26.6894,   3.9000, -11.8147, -17.7994,  24.8402,
        -26.5748,  22.2962, -11.7148, -12.3652,   2.3795,   7.6798, -22.2336,
        -25.1569, -17.6484, -19.7486,   3.1402, -17.8605, -13.5138,  13.5349,
          1.3882,  33.9230,  -9.2085,  -6.5632,  -8.3163, -27.4731, -32.5096,
          6.4662,  14.0257,   4.0577, -10.7495, -10.4640,  33.1163, -21.2907,
        -27.6860,   3.4410,  -3.3653, -11.7509,  16.2152, -39.5942, -15.1898,
        -35.6131,   6.9573, -23.8600,   0.0818,  18.9825, -12.7931, -27.9797,
          1.6694,  -5.3829,  -0.8177,   5.8408,  -1.3645,  10.3066, -15.7771,
         12.8296, -17.2714, -34.7069,  28.7387,  41.3817,  27.9243,   6.5860,
         14.5791, -52.8733,  17.7706,  -3.5106, -13.6991,  10.5716,   1.9264,
         42.7409,   3.3306, -33.8983, -26.3936, -46.0700, -19.0354,  -2.6876,
        -52.7824,  -3.0759, -13.9642,  10.3853,  -1.7947,  17.7504,  13.0380,
          5.5325,  21.6279,  -5.6648, -13.2786,  -0.2836, -24.7143,  12.0942,
        -30.5527,   5.9480, -32.6845,  -1.3377,  20.5250,  -8.9241,  -9.2027,
        -31.2268, -18.0490,  -4.4949, -18.4595, -19.8005,  11.0795, -21.4954,
        -10.7590,  -1.4602,  -4.9541,   8.0809,   0.8277, -15.9250,  11.6963,
         -4.0755,  15.7040, -24.6733,   1.6899,  -8.3766,  15.0254,   3.0448,
         43.9724,  30.6890, -15.3496, -10.0466,   4.9025,  38.8629,  -2.4365,
        -23.2030, -16.8381,  -9.2210, -39.2661,  36.5700,   3.9311, -20.3710,
         11.8976,   2.5863, -22.2250]
    )