#!/bin/bash

# This requires the use of clang++ as CXX compiler!

# the record dim has 1095 entries
for i in {0..1000..20}; do
  cmake -DCMAKE_CXX_FLAGS=-ftime-trace -DLLAMA_COMPTIME_RECORD_DIM_SIZE=$i .. > /dev/null 2>&1
  make llama-comptime > /dev/null 2>&1
  s=$(grep -E -o "Instantiate(Class|Function)" examples/comptime/CMakeFiles/llama-comptime.dir/comptime.cpp.json | wc -l)
  echo $i $s
done
