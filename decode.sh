#!/bin/bash
#$ -l h_rt=50:00:00
#$ -l rmem=20G
#$ -M z.yue@sheffield.ac.uk
#$ -m bea
# -P rse
# -q rse.q
#$ -l gpu=1
# -P tapas 
# -q tapas.q

# usage: qsub -V -o qsub/decode_${fd}_${feat} -e qsub/decode_${fd}_${feat} -j y ./decode.sh --fd ${fd} --feat ${feat}
fold="source_filter"
feat="vt_sp"
fd="fd1"

. ./cmd.sh
. ./path.sh
. ./parse_options.sh


spks=(F01 M01 M02 M04 M05 F03 F04 M03 FC01 FC02 FC03 FC04 MC01 MC02 MC03 MC04)
eps=(19 24)
for ep in ${eps[@]}; do
for spk in ${spks[@]}; do
   kaldi_decoding_scripts/decode_dnn.sh /data/ac1zy/pytorch-kaldi-new/pytorch-kaldi/exp/${fold}/${fd}_${feat}/decoding_${spk}_test_lm200_out_dnn4.conf /data/ac1zy/pytorch-kaldi-new/pytorch-kaldi/exp/${fold}/${fd}_${feat}/decode_${spk}_test_lm200_out_dnn4 /data/ac1zy/pytorch-kaldi-new/pytorch-kaldi/exp/${fold}/${fd}_${feat}/exp_files/forward_${spk}_test_lm200_ep${ep}_ck*_out_dnn4_to_decode.ark

/data/ac1zy/kaldi/egs1/torgo/s9/local/score.sh --min-lmwt 10 --max-lmwt 20 /data/ac1zy/kaldi/egs1/torgo/s9/data/${fd}/${spk}/test /data/ac1zy/kaldi/egs1/torgo/s9/exp/${fd}/train/tri3b/graph_test_lm200 /data/ac1zy/pytorch-kaldi-new/pytorch-kaldi/exp/${fold}/${fd}_${feat}/decode_${spk}_test_lm200_out_dnn4

done
done
