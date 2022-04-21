#! /bin/bash
#$ -l h_rt=10:00:00
#$ -l rmem=20G
#$ -M z.yue@sheffield.ac.uk
#$ -m bea
# -P rse
# -q rse.q
#$ -l gpu=1
#$ -P tapas 
#$ -q tapas.q

# Author: Erfan Loweimi, CSTR, University of Edinburgh
# usage:. ./tools/pyk_decode.sh --exp-dir exp/source_filter/fd1_mag_sp_tempo --decoding-epochs '1 3 5 7 10 13 15 17 20 22 25 27 28 30'


set -euo pipefail

# ============================================================================ #
# Setting the parameters and pathes here ...
pyk_dir="/data/ac1zy/pytorch-kaldi-new/pytorch-kaldi"


# When options are empty and are not supplied by input default value is taken from pytorch config file
# Default values can be set through setting the followig variables

exp_dir= # name inside pyk-dir/exp, e.g. WSJ/WSJ_CNN_raw_tri4b_ali_bd
decoding_epochs="all" # "all" or an array, e.g. '1 3 5 7' WITHOUT Parentheses
acwt=0.1            # ~ 0.1; almost 1/optimal-lmwt
min_lmwt=5        # ~ 7; for TIMIT 1
max_lmwt=20       # ~ 18
beam=20.0            # ~ 13; for AMI 20
latbeam=8.0         # ~ 8; fro AMI 12 
scoring_script=/data/ac1zy/kaldi/egs1/torgo/s9/local/score.sh  # Each database has its own
gpu_device=1
# ============================================================================ #
parse_options=$pyk_dir/kaldi_decoding_scripts/parse_options.sh

echo -e "\n $0 $@ \n"
source $parse_options || exit 1;

if [[ $# -gt 14 ]] || [[ -z $exp_dir ]]; then
    echo -n "Usage: $0 --exp-dir <path/to/exp-dir-mdl> --decoding-epochs <array|all>"
    echo -n "[--acwt <acwt>] [--min-lmwt <min-lmwt>] [--max-lmwt <max-lmwt>]"
    echo "[--beam <beam>] [--latbeam <latbeam>] [--gpu-device 1]"
    echo "Example 1: $0 --exp-dir exp/WSJ/WSJ_CNN_raw_tri4b_ali_bd --decoding-epochs all --acwt 0.1 --min-lmwt 7 --max-lmwt 18 --beam 13 --latbeam 8 --gpu-device 1" 
    echo "Example 2: $0 --exp-dir exp/WSJ/WSJ_CNN_raw_tri4b_ali_bd --decoding-epochs '1 3 5 7' # <- without parentheses \n"  
    exit 1;
fi
# ============================================================================ #

# Modify the config file with new decoding setting ...
scoring_opts=
[[ ! -z $acwt ]] && acwt_line="acwt = $acwt"
[[ ! -z $min_lmwt ]] && scoring_opts="$scoring_opts --min-lmwt $min_lmwt"
[[ ! -z $max_lmwt ]] && scoring_opts="$scoring_opts --max-lmwt $max_lmwt"
[[ ! -z $beam ]] && beam_line="beam = $beam"
[[ ! -z $latbeam ]] && latbeam_line="latbeam = $latbeam"
[[ ! -z $scoring_opts ]] && scoring_opts_line="scoring_opts = \"$scoring_opts\""
[[ ! -z $scoring_script ]] && scoring_script_line="scoring_script = $scoring_script" 
require_decoding_line="require_decoding = True"


# Set CUDA_VISIBLE_DEVICES ...
#export CUDA_VISIBLE_DEVICES=$gpu_device
echo -e "\n [info] Set CUDA_VISIBLE_DEVICES to '$gpu_device'"


# Initial parameters ...
pyk_run_exp=$pyk_dir/run_exp_modified.py
mdl_exp_dir=$pyk_dir/$exp_dir
exp_res_res=$mdl_exp_dir/res.res
mdl_exp_files=$mdl_exp_dir/exp_files
multi_decode_dir=$mdl_exp_dir/multiple_decoding_results
multi_decode_mdl_dir=$multi_decode_dir/models
multi_decode_config_dir=$multi_decode_dir/configs
multi_decode_results_file="$multi_decode_dir/multiple_decoding_res.res-`date +%Y-%m-%d_%H-%M-%S`"
multi_decode_log_file=$multi_decode_dir/multiple_decoding.log
config_main=$mdl_exp_dir/conf_main.cfg


# Find train data_name ...
train_data_name=`grep train_with ${mdl_exp_dir}/conf.cfg | awk {'print $3'}`
echo -e "\n [info] Train data name is '$train_data_name'\n"


# Find number of chunks ...
num_chunks=`ls ${mdl_exp_files}/train_${train_data_name}_ep00_ck*.info | wc -l`
if [[ $num_chunks == 0 ]]; then
    num_chunks=`grep n_chunks $mdl_exp_dir/conf.cfg | awk 'NR==1 {print $3}'`
    if [[ `ls -1 $mdl_exp_files/*.lst 2>/dev/null | wc -l` != 0 ]]; then
        echo "Create (empty) .info files using *.lst files to skip the training stage ..."
        for i in *lst; do touch `echo $i | sed s/.lst/.info/`; done
    elif [[ `ls -1 $mdl_exp_files/*.cfg 2>/dev/null | wc -l` != 0 ]]; then
        echo "Create (empty) .info files using *.cfg files to skip the training stage ..."
        for i in *lst; do touch `echo $i | sed s/.cfg/.info/`; done
    else
        echo ".info, .lst and .cfg files do not exit in '$mdl_exp_files'!"
        exit 1;
    fi
fi
echo -e " [info] Number of chunks for training data is '$num_chunks'\n"


# Find number of architectrues ...
if [[ ! -f $mdl_exp_files/final_architecture1.pkl ]]; then
    cp $mdl_exp_files/final_archs/* $mdl_exp_files
fi


num_archs=$(ls $mdl_exp_files/final_architecture*pkl | wc -l)
archs_idx=$(ls $mdl_exp_files/final_architecture*pkl | awk -F "architecture" {'print $2'} | awk -F ".pkl" '{print $1}')
echo -e " [info] Number of architectures in $mdl_exp_files is $num_archs"
echo -e " [info] Architecture Indeices are " $archs_idx
echo


mkdir -p $mdl_exp_files/final_archs
if [[ -f $mdl_exp_files/final_architecture1.pkl ]]; then
    for final in $mdl_exp_files/final_architecture*pkl; do
        cp $final $mdl_exp_files/final_archs
        mv $final `echo $final | sed s/final_arch/FINAL_arch/`
    done
fi


# Find the decoding epochs (if all is chosen) ... 
if [ "$decoding_epochs" == "all" ]; then
    echo -e " [info] Decode for all the epoches ...$multi_decode_config_dir\n"
    cfg_files=($(ls $multi_decode_config_dir))
    epochs=($(for i in ${cfg_files[@]}; do echo ${i%*.cfg} | awk -F'_' '{print $5}'; done | sort -n))
else
    epochs=(${decoding_epochs[@]})
fi
echo -e " [info] Decoding will be done for the following EPOCHs: ${epochs[@]}\n"

#if [ -f $multi_decode_results_file ]; then
#    echo -e " [info] $multi_decode_results_file exists ... \n"
#    old_multi_decode_file="$multi_decode_dir/old_multiple_decoding_res-`date +%Y-%m-%d.%H:%M:%S`.res"
#    mv $multi_decode_results_file $old_multi_decode_file
#    echo -e " ... Moved it to $old_multi_decode_file \n"
touch $multi_decode_results_file


# Decoding ...
cp $mdl_exp_dir/conf.cfg $config_main

for ep in ${epochs[@]}; do

  # 1. Create link to models at given epoch in exp_files dir
  echo -e "\n Decoding for epoch $ep started ..."
  ep1=$((ep-1))
  ep2=`[[ $ep1 -lt 10 ]] && echo 0$ep1 || echo $ep1`

  num_chunks1=$((num_chunks-1))
  [[ $num_chunks1 -lt 10 ]] && num_chunks1=`echo 0$num_chunks1`
  for i in $archs_idx; do
    src_arch=$multi_decode_mdl_dir/EPOCH${ep}_architecture${i}.pkl
    tgt_arch=$mdl_exp_dir/exp_files/train_${train_data_name}_ep${ep2}_ck${num_chunks1}_architecture${i}.pkl
    tgt_arch2=$mdl_exp_dir/exp_files/final_architecture${i}.pkl
    [[ -L $tgt_arch ]] && rm $tgt_arch
    [[ -L $tgt_arch2 ]] && rm $tgt_arch2
    [[ ! -f $tgt_arch ]] && ln -s $src_arch $tgt_arch 
    #[[ ! -f $tgt_arch2 ]] && ln -s $src_arch $tgt_arch2 
 done

  # 2. Update the decoding config for the given parameters, e.g. acwt, lmwt, ...
  config_file=$multi_decode_config_dir/multi_decoding_config_ep_$ep.cfg
  sed s:"^n_epochs_tr.*":"n_epochs_tr = $ep": $config_main > $config_file
  [[ ! -z $acwt ]] && sed -i s:"^acwt.*":"$acwt_line": $config_file
  [[ ! -z $scoring_opts ]] && sed -i s:"^scoring_opts.*":"$scoring_opts_line": $config_file
  [[ ! -z $beam ]] && sed -i s:"^beam.*":"$beam_line": $config_file
  [[ ! -z $latbeam ]] && sed -i s:"^latbeam.*":"$latbeam_line": $config_file
  [[ ! -z $scoring_script ]] && sed -i s:"^scoring_script.*":"$scoring_script_line": $config_file
  sed -i s:"^require_decoding.*":"$require_decoding_line": $config_file
  
  # 3. Remove the forward and compute the quasi-likelihood
  rm -rf $mdl_exp_dir/exp_files/forward_*dnn4_to_decode.ark
rm -rf $mdl_exp_dir/exp_files/forward_*{.cfg,.lst,.info}

  # 4. Decode
  python -u $pyk_run_exp $config_file

  # 5. Update multi-decoding-results files
  wer_ep=$(grep WER $exp_res_res)
  echo -e "\n ep=$ep\n  $wer_ep" >> $multi_decode_results_file

  echo -e "\n Decoding for $ep finished successfully ..."
done | tee $multi_decode_log_file

less $multi_decode_log_file |& stdbuf -oL tr '\r' '\n' | grep -v '\[\-\|\-\]' > ${multi_decode_log_file}_tmp
mv ${multi_decode_log_file}_tmp $multi_decode_log_file

echo -e "\n Decoding results are in $multi_decode_results_file"
echo -e "\n Log file -> $multi_decode_log_file \n"


cp $config_main $mdl_exp_dir/conf.cfg || rm $config_main
rm -rf $mdl_exp_files/FINAL_architecture*pkl
cp $mdl_exp_files/final_archs/* $mdl_exp_files
