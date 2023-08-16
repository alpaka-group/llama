This is a compliant version of the [STREAM benchmark](https://www.cs.virginia.edu/stream/).
There are a few official such versions, which you can download [here](https://www.cs.virginia.edu/stream/FTP/Code/).
This example is based on the [original version](https://www.cs.virginia.edu/stream/FTP/Code/stream.c) which places the input and ouput arrays into the static program segment.
This requires to set corresponding command line flags, notable `-mcmodel=medium` for gcc, clang and icpx.
Unfortunately, there is no equivalent for MSVC und you are therefore limited in the values of `STREAM_ARRAY_SIZE`.
For MSVC, you can checkout the STREAM [version using `posix_memalign`](https://www.cs.virginia.edu/stream/FTP/Code/Versions/stream_5-10_posix_memalign.c) for allocation.
Some fixes were applied to this version regardless to make it compile with MSVC.
