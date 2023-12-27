#!/bin/bash

# run this on the generated n-body executable to see where rsqrt is generated
objdump -SC $1 | awk '
/^000.+/ {
  if ($0 ~ /[Uu]pdate/) {
    f = $0;
  } else {
    f = ""
  }
}

/r?sqrt/ {
  if (f != "") {
    if ($0 ~ "rsqrt")
      printf "\033[32m"
    else
      printf "\033[31m"
    print substr(f, 0, 80), "\n", $0
  }
}
'
