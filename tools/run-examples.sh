#!/bin/bash
if [ "$#" -eq 0 ]; then
    echo -e "run-examples.sh runs examples in given build directory and prints the archieved time"
    echo -e "Usage:\trun-examples.sh build-directory [another] [another] […] "
    exit 0
fi
printf "                      Build directories: "
for build in "$@"
do
    printf "%16s " $build
done
printf "\n"
printf "      async example blur kernel runtime: "
for build in "$@"
do
    RESULT=`./$build/examples/asynccopy/llama-asynccopy | grep "Blur kernel:" | awk '{ print $3 }'`
    printf "%16s " $RESULT
done
printf "\n"
printf "    nbody example update kernel runtime: "
for build in "$@"
do
    RESULT=`./$build/examples/nbody/llama-nbody | grep "Update kernel:" | tail -n 1 | awk '{ print $3 }'`
    printf "%16s " $RESULT
done
printf "\n"
printf "       treemap example complete runtime: "
for build in "$@"
do
	TIME="$(time ( ./$build/examples/treemaptest/llama-treemaptest ) 2>&1 1>/dev/null )"
    RESULT=`echo $TIME | tail -n 2 | head -n 1 | awk '{ print $2 }' | sed s/0m// | sed s/,/./ | sed s/s//`
    printf "%16s " $RESULT
done
printf "\n"
printf "vectoradd example update kernel runtime: "
for build in "$@"
do
    RESULT=`./$build/examples/vectoradd/llama-vectoradd | grep "Add kernel:" | tail -n 1 | awk '{ print $3 }'`
    printf "%16s " $RESULT
done
printf "\n"
