# Elementwise addition

Add two inputs element by element on the GPU and verify the result on the CPU. The source includes a 1D vector kernel and a 2D matrix kernel. The executable runs the matrix version.

The matrix uses a contiguous buffer. Threads outside its row or column bounds skip the write.

From the repository root, configure once and build:

```bash
cmake -S . -B build
cmake --build build
```

Run the exercise:

```bash
./build/elementwise_add
```

Or run the registered test on a machine with a CUDA-capable GPU:

```bash
ctest --test-dir build --output-on-failure
```

GitHub Actions builds the exercise, but its standard runner has no GPU and does not run this test.
