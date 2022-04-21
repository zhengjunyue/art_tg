awk '{print $1,$6}' wav.scp > wav_lst.scp
python save_raw_fea_libri.py
scp -r /data/ac1zy/kaldi/egs/librispeech/s5/data/raw_TIMIT_200ms/train_clean_100/feats_raw.scp /data/ac1zy/kaldi/egs/librispeech/s5/data/train_clean_100/
scp -r /data/ac1zy/kaldi/egs/librispeech/s5/data/raw_TIMIT_200ms/dev/feats_raw.scp data/dev_clean/
scp -r /data/ac1zy/kaldi/egs/librispeech/s5/data/raw_TIMIT_200ms/test/feats_raw.scp data/test_clean/
