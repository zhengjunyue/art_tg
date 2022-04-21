# Multi-modal Acoustic-articulatory Feature Fusion for Dysarthric Speech Recognition

Copyright 2022 Zhengjun Yue, Heidi Christensen, Jon Barker

# Description

This is a Pytorch-Kaldi recipe to build automatic speech recognition systems on the
[Torgo corpus](http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) of
dysarthric speech.
"Multi-modal Acoustic-articulatory Feature Fusion for Dysarthric Speech Recognition"


## Setup

Run the following:

```sh
. ./env_setup.sh

. ./cmd.sh
. ./path.sh
. ./parse_options.sh

```


## Usage

The following instructions allow to train ASR systems on Torgo and to reproduce
results from the paper.

### Process EMA data



### Train ASR systems

```sh
# MFCC baseline
python -u run_exp.py cfg/fd1_mfcc.cfg 

# Multi-stream fusion
python -u run_exp.py cfg/fd1_mfcc_concat2.cfg 

```




## Citation 

Please cite the following [paper](https://kclpure.kcl.ac.uk/portal/en/publications/multimodal-acousticarticulatory-feature-fusion-for-dysarthric-speech-recognition(24b17d9d-4f62-4b59-a5ef-f887d29df3e4).html) if you use this script for your research or are 
interested in this paper.

```BibTeX
@inproceedings{yue2022multi,
  title={MULTI-MODAL ACOUSTIC-ARTICULATORY FEATURE FUSION FOR DYSARTHRIC SPEECH RECOGNITION},
  author={Yue, Zhengjun and Loweimi, Erfan and Cvetkovic, Zoran and Christensen, Heidi and Barker, Jon},
  booktitle={ICASSP, IEEE International Conference on Acoustics, Speech and Signal Processing},
  year={2022}
}
```
The code is based on [an earlier recipe](https://github.com/mravanelli/pytorch-kaldi) by Mirco Ravanelli, Titouan Parcollet and Yoshua Bengio.
