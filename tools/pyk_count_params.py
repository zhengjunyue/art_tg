#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:08:19 2020

@author: erfanloweimi
"""

import torch
import os
import glob
import argparse


parser = argparse.ArgumentParser("Compute number of parameters per model per layer for Pytorh-Kaldi")
parser.add_argument("exp_dir", type=str, help="path to the exp_files dir or the out_folder in cfg file")
args = parser.parse_args()

# ========================================================================== #
# ========================================================================== #
def load_pytorch_model(pyt_model_path, map_location='cpu'):
    pyt_mdl = torch.load(pyt_model_path, map_location=map_location)
    mdl_param = pyt_mdl['model_par']
    mdl_optimizer_param = pyt_mdl['optimizer_par']
    return mdl_param, mdl_optimizer_param
# ========================================================================== #
def dnn_num_params(pyt_model_or_path, print_num_params=True):
    if isinstance(pyt_model_or_path, str):
        pyt_model_path = pyt_model_or_path
        pyt_mdl_param, _ = load_pytorch_model(pyt_model_path, map_location='cpu')
        num_params = sum(param.numel() for param in pyt_mdl_param.values())
    else:
        pyt_model = pyt_model_or_path
        num_params = sum(param.numel() for param in pyt_model.parameters() if param.requires_grad) 
    
    if print_num_params:
        print("Number of parameters of the model is {:,}".format(num_params))

    return num_params
# ========================================================================== #

# ========================================================================== #
def pyk_dnn_num_params(pyk_exp_dir, print_num_params=True):
    
    if isinstance(pyk_exp_dir, str):
        arch_list = pyk_get_arch_list(pyk_exp_dir)
        num_archs = len(arch_list)
    elif not isinstance(pyk_exp_dir, list):
        raise ValueError("input should be either a list of pathes to arch.pkl "\
                         "or the path to the pyk-exp-dir")
    
    num_param_list = []
    for i, arch_path in enumerate(arch_list):
        num_param_list.append(dnn_num_params(arch_path, print_num_params=False))
    
    if print_num_params:
        print("\n==============================")
        if isinstance(pyk_exp_dir, str):
            print(" * EXPeriment dir: {}".format(pyk_exp_dir))
        #print(" * Number of architectures is {}".format(num_archs))
        print(" * Number of parameters (#Param) per architecture ...")
        #print("------------------------------")
        for i in range(num_archs):
            print("  -- {}: {:,}".format(os.path.basename(arch_list[i]), num_param_list[i]))
        print("------------------------------")
        print("  -> Total #Param is {:,}".format(sum(num_param_list)))
        print("==============================\n")

    return num_param_list
# ========================================================================== #
def pyk_get_arch_list(pyk_exp_dir):
    
    arch_list = glob.glob("{}/final_architecture*pkl".format(pyk_exp_dir))
    if len(arch_list) == 0:
        exp_files = "{}/exp_files".format(pyk_exp_dir)
        arch_list = glob.glob("{}/final_architecture*pkl".format(exp_files))

    if len(arch_list) == 0:
        raise ValueError("{} does not include any final_architecture*.pkl file!".format(pyk_exp_dir))
    else:
        return sorted(arch_list)
# ========================================================================== #
# ========================================================================== #
if __name__ == "__main__":
    pyk_dnn_num_params(args.exp_dir)

# pyt_model_path = "/Users/erfanloweimi/Desktop/final_architecture2.pkl"
# mdl_param, mdl_optimizer_param = load_pytorch_model(pyt_model_path)

# dnn_num_params = dnn_num_params(pyt_model_path)
# print("Number of parameters of the model is {:,}".format(dnn_num_params))



    
    
