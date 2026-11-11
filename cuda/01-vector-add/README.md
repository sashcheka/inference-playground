# Vector addition

Write a CUDA kernel that adds two arrays element by element. Then launch it from the CPU and check the result.

Start with 1,000 `float` elements and 256 threads per block. Make sure threads outside the array bounds do not access memory.

From the repository root, configure once and build:

```bash
cmake -S . -B build
cmake --build build
```

Run the exercise:

```bash
./build/vector_add
```
