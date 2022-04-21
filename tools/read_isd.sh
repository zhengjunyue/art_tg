#!/bin/bash

feat=mag_sp
fd=fd1
. ./parse_options.sh

spks=(F01 M01 M02 M04 M05 F03 F04 M03 FC01 FC02 FC03 MC01 MC02 MC03 MC04)

for spk in "${spks[@]}";do
file=exp/source_filter/${fd}_${feat}/decode_${spk}_test_lm200_out_dnn4/scoring_kaldi/best_wer
wer=awk '{print $2}' $file
ins=awk '{print $7}' $file
del=awk '{print $9}' $file
sub=awk '{print $11}' $file
err=awk '{print $4}' $file
total=awk '{print $5}' $file | awk -F "" '{print $1}'

echo $spk $wer $ins $del $sub $err $total
done
