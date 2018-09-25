#!/bin/sh
for build in "$@"
do
    cd "$build"
    make -j
    cd ..
done
