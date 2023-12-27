# n-body simulation

This example shows an n-body simulation, comparing a LLAMA implementation with manually written versions.

## OpenMP

All kernels can be run scalar or using multithreading+SIMD.
For the latter, enable `LLAMA_NBODY_OPENMP` in cmake,
and specify OpenMP thread affinity when executing:
`OMP_NUM_THREADS=x OMP_PROC_BIND=true OMP_PLACES=cores llama-nbody`,
where x is the number of cores on your system.

## rsqrt

The use of the `rsqrt` instruction is disabled,
which is required for comparable benchmarks (so all versions use sqrt and divison).
You can enable `rsqrt` by setting the appropriate variable in the C++ source file,
and compile with `-ffast-math`.
