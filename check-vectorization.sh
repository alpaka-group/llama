#!/bin/bash

function set_color {
	if [ "$1" -eq "0" ]; then
		echo -ne "\e[31m"
	else
		echo -ne "\e[32m"
	fi
}

function reset_color {
	echo -e "\e[39m"
}

function check_fma
{
	number_ss=$(grep $2 $1 | grep ss | wc -l)
	number_sd=$(grep $2 $1 | grep sd | wc -l)
	number_full=$(grep $2 $1 | grep p | wc -l)
	set_color $number_full
	echo -ne "$3: ps/pd \e[1m$number_full\e[21m (ss $number_ss, sd $number_sd)"
	reset_color
}

grep_output=$(mktemp "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")
objdump_output=$(mktemp "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")

objdump -DSC $1 > $objdump_output

cat $objdump_output | grep vfmadd > $grep_output
check_fma $grep_output "xmm" "SSE1…4 FMA"
check_fma $grep_output "ymm" "AVX1/2 FMA"
check_fma $grep_output "zmm" "AVX512 FMA"

cat $objdump_output | grep vadd > $grep_output
check_fma $grep_output "xmm" "SSE1…4 ADD"
check_fma $grep_output "ymm" "AVX1/2 ADD"
check_fma $grep_output "zmm" "AVX512 ADD"

cat $objdump_output | grep vmul > $grep_output
check_fma $grep_output "xmm" "SSE1…4 MUL"
check_fma $grep_output "ymm" "AVX1/2 MUL"
check_fma $grep_output "zmm" "AVX512 MUL"

rm $grep_output
rm $objdump_output
