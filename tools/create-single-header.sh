#!/bin/bash

workingcopy_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )/.."
amalgamate_tmp_dir=$workingcopy_dir/_amalgamate_tmp
destination_dir=$workingcopy_dir/single_header

git clone https://github.com/shrpnsld/amalgamate.git --depth 1 $amalgamate_tmp_dir/clone
cd include/llama
$amalgamate_tmp_dir/clone/amalgamate -o $amalgamate_tmp_dir -H -v -a -n 'llama'
cd ../..
mv $amalgamate_tmp_dir/llama-amalgamated/llama.hpp $workingcopy_dir
rm -rf $amalgamate_tmp_dir
