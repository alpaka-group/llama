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
	echo -ne "$3 FMA: ps/pd \e[1m$number_full\e[21m (ss $number_ss, sd $number_sd)"
	reset_color
}

tempfile=$(mktemp "${TMPDIR:-/tmp/}$(basename $0).XXXXXXXXXXXX")
objdump -DSC $1 | grep vfmadd > $tempfile

check_fma $tempfile "xmm" "SSE1…4"
check_fma $tempfile "ymm" "AVX1/2"
check_fma $tempfile "zmm" "AVX512"

rm $tempfile
