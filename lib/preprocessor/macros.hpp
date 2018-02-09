#pragma once

#define LLAMA_MARK_ASSUMED_DEPENDENCIES_AS_INDEPENDENT \
    _Pragma ("ivdep") \
    _Pragma ("GCC ivdep")
