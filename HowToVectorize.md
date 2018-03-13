How to vectorize
================

* ICC is able to recognize more complex access pattern as vectorizable
* especially stores needs to be in the inner loop, e.g.: ```C++
float a[256];
float b[256];
float c[256];
for (int i = 0; i < 256; ++i)
    for (int j = 0; j < 256; ++j)
        c[j] = a[i] * b[j];
```
Although loop switching should be easy for the compiler, at least the GCC does
not notice the possibility and although icc vectorizes an outer j-loop, the
inner j-loop version is faster.
* Use `#pragma`s to mark arrays inside loop independent
* Calculating of loop borders outside of loop instead of `if (i < max_value)`
constructs inside loop
