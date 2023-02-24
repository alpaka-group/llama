This is a compliant version of the [STREAM benchmark](https://www.cs.virginia.edu/stream/).
There are a few official such versions, which you can download [here](https://www.cs.virginia.edu/stream/FTP/Code/).
The [original version](https://www.cs.virginia.edu/stream/FTP/Code/stream.c) places the input and ouput arrays into the static program segment.
This causes the linker to complain because it breaks limits of ELF/PE (Linux/Windows executable file format) for big array sizes.
We therefore took the [version using `posix_memalign`](https://www.cs.virginia.edu/stream/FTP/Code/Versions/stream_5-10_posix_memalign.c) for allocation.
Some fixes were applied to make it compile with MSVC.
