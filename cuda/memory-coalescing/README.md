# Memory coalescing

A CUDA exercise on how contiguous and strided memory reads affect kernel performance.

From the repository root, configure once and build:

```bash
cmake -S . -B build
cmake --build build --target memory_coalescing
```

Run the exercise:

```bash
./build/memory_coalescing
```
