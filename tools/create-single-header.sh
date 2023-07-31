#!/bin/bash

workingcopy_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )/.."
amalgamate_tmp_dir=$workingcopy_dir/_amalgamate_tmp
destination_dir=$workingcopy_dir/single-header

git clone https://github.com/shrpnsld/amalgamate.git --depth 1 $amalgamate_tmp_dir/clone
mkdir $amalgamate_tmp_dir
$amalgamate_tmp_dir/clone/amalgamate -i include/llama -o $amalgamate_tmp_dir -v -a -n 'llama' -x 'hpp,cpp'
mkdir -p $destination_dir
mv $amalgamate_tmp_dir/llama-amalgamated/llama.hpp $destination_dir
rm -rf $amalgamate_tmp_dir
