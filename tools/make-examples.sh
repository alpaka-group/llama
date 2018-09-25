#!/bin/sh
if [ "$#" -eq 0 ]; then
    echo -e "make-examples.sh calls make -j in every given build directory"
    echo -e "Usage:\tmake-examples.sh build-directory [another] [another] […] "
    exit 0
fi
for build in "$@"
do
    cd "$build"
    make -j
    cd ..
done
