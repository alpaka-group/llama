#!/bin/bash

folder=$1
example=$2

tempfile1=$(mktemp "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")
tempfile2=$(mktemp "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")

cat $folder/examples/$example/CMakeFiles/llama-$example.dir/$example.ptx > $tempfile1
cat $tempfile1 | sed -n -e '/cvta.to.global.u64/,$p' > $tempfile2
while [ -s "$tempfile2" ]
do
	mv $tempfile2 $tempfile1
	cat $tempfile1 | sed '/}/q' > $tempfile2
	echo "loc: " `cat $tempfile2 | wc -l`
	cat $tempfile1 | sed -n -e '/}/,$p' | sed -n -e '/cvta.to.global.u64/,$p' > $tempfile2
done

rm $tempfile1
rm $tempfile2
