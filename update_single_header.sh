#!/bin/bash

amalgamate_tmp_dir=$(pwd)/_amalgamate_tmp
destination_dir=$(pwd)/single_header

git clone https://github.com/shrpnsld/amalgamate.git --depth 1 $amalgamate_tmp_dir/clone
cd include/llama
$amalgamate_tmp_dir/clone/amalgamate -o $amalgamate_tmp_dir -H -v -a -n 'llama'
cd ../..
mkdir $destination_dir
mv $amalgamate_tmp_dir/llama-amalgamated/llama.hpp $destination_dir
rm -rf $amalgamate_tmp_dir
