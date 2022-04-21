#!/bin/env python
# -*- coding:utf-8 -*-
 
import sys
 
def replace(file_path, old_str, new_str):
  f = open(file_path,'r+')
  all_lines = f.readlines()
  f.seek(0)
  f.truncate()
  for line in all_lines:
    line = line.replace(old_str, new_str)
    f.write(line)
  f.close() 

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("need 3 params")
    sys.exit(1)
  file_name = sys.argv[1]
  src_str = sys.argv[2]
  dst_str = sys.argv[3]
  replace(file_name, src_str, dst_str)
