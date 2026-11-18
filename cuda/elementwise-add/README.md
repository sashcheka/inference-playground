# Elementwise addition

A CUDA exercise on elementwise addition and 1D and 2D thread indexing. The executable adds two matrices and verifies the result on the CPU.

From the repository root, configure once and build:

```bash
cmake -S . -B build
cmake --build build --target elementwise_add
```

Run the exercise:

```bash
./build/elementwise_add
```
