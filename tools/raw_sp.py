#!/usr/bin/env python3


import torch
import os
import glob
import argparse

import scipy.io.wavfile
from scipy import fftpack
import math
import numpy as np
import os
from subprocess import check_output,DEVNULL
import subprocess


def wav_scp_cmd2signal(cmd_wav_scp_wo_utt_id, data_type="<i2", start_ind=0, end_ind=-1, remove_header=True, header_bytes=44):

    cmd_wav_scp_wo_utt_id = cmd_wav_scp_wo_utt_id.strip()
    if cmd_wav_scp_wo_utt_id[-1] == "|":
        cmd_wav_scp_wo_utt_id = cmd_wav_scp_wo_utt_id[:-1]

    if end_ind != -1:
      if start_ind > end_ind:
        raise ValueError(f"Start index ({start_ind}) must be -lt end index ({end_ind})!")

    count = end_ind - start_ind
    if remove_header:
        offset = 2 * start_ind + header_bytes
    else:
        offset = 2 * start_ind
    if data_type.startswith(("<", ">")):
        data_type = data_type[1:]

    return np.frombuffer(
        check_output(cmd_wav_scp_wo_utt_id, shell=True, stderr=DEVNULL),
                     data_type, count, offset)

#output=wav_scp_cmd2signal("sox -t wav /shared/spandh1/Shared/data/TORGO/torgo_d3/data/MC04/Session2/wav_arrayMic/0996.wav -t wav - tempo 1.1 |")
output=wav_scp_cmd2signal("sox -t wav /shared/spandh1/Shared/data/TORGO/torgo_d3/data/MC04/Session2/wav_arrayMic/0996.wav -t wav - speed 1.1 |")
