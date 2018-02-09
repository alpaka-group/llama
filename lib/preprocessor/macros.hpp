#pragma once

#define LLAMA_INDEPENDENT_DATA \
    _Pragma ("ivdep") \
    _Pragma ("GCC ivdep")
