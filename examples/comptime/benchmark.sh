#!/bin/bash

# the record dim has 1095 entries
repetitions=20
for i in {0..1000..20}; do
  cmake -DLLAMA_COMPTIME_RECORD_DIM_SIZE=$i .. > /dev/null 2>&1
  for ((j=0; j < $repetitions; j++)); do
    make clean
    s=$( { TIMEFORMAT=%R; time make -j1 llama-comptime > /dev/null 2>&1 ; unset TIMEFORMAT; } 2>&1 )
    echo $i $s
  done
done
