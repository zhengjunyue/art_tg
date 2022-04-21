#!/bin/bash

exp="s9"
scp="s9"
. ./parse_options.sh
   for fd in ${fds[@]}; do
     for feat in ${feats[@]}; do

      if [  ! -f cfg/${exp}/${fd}_${feat}.cfg ]; then
        scp -r cfg/${exp}/fd1_${feat}.cfg cfg/${exp}/${fd}_${feat}.cfg
        python chang_spk.py cfg/${exp}/${fd}_${feat}.cfg fd1 ${fd}
      fi
       qsub -V -o result_tg/${exp}/qsub_${fd}_${feat} -e result_tg/${exp}/qsub_${fd}_${feat} -j y ./run_${scp}.sh --feat $feat --fd ${fd}
