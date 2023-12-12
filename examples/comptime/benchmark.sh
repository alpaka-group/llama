#!/bin/bash

# the record dim has 1095 entries
repetitions=20
for i in {0..1000..20}; do
  cmake -DLLAMA_COMPTIME_RECORD_DIM_SIZE=$i .. > /dev/null 2>&1
  for ((j=0; j < $repetitions; j++)); do
    make clean
    s=$(\time -f "%e" make llama-comptime 2>&1 > /dev/null)
    echo $i $s
  done
done
