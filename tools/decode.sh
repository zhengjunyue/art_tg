#!/bin/bash

fds=(fd1 fd2 fd3 fd4 fd5)
spks=(F01 F03 F04 M01 M02 M03 M04 M05)
testsets=(test_sentence test_word)
spks1=(F01)
for spk in ${spks[@]}; do
 for fd in ${fds[@]}; do 
for testset in ${testsets[@]}; do 
decode_ark=/data/ac1zy/pytorch-kaldi/exp/test/${fd}_cd/decode_${fd}_${spk}_test_lm200_out_dnn2
data=data/${fd}/${spk}/${testset}
graphdir=exp/${fd}/train/tri3b_cleaned/graph_test_lm200
mkdir ${decode_ark}/${testset}
cp ${decode_ark}/lat.1.gz  ${decode_ark}/lat.2.gz ${decode_ark}/${testset}
dir=${decode_ark}/${testset}
local/score.sh --cmd "run.pl" $data $graphdir $dir; done; done; done

