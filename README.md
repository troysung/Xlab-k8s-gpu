<img align="right" height="60" src="https://raw.githubusercontent.com/1751200/Xlab-k8s-gpu/master/images/Sage_logo_new.png"/>

# Sage Linear System Solver

## Introduction

GPU-accelerated algorithms for solving large sparse linear algebraic equations using C++ language, implementing *Jacobi Method*, *Gauss—Seidel Method*, *Successive Over-Relaxation (SOR) Method*, and *Conjugate Gradient Method*. Among these, we also implemented *Jacobi Method* and *Conjugate Gradient* using Nvidia CUDA API to accelerate these algorithms.

It is a collaborative, interdisciplinary project drawing on expertise from School of Software Engineering and College of Civil Engineering, Tongji University, Shanghai.

## Getting Started

### Environment Requirements

- NVIDIA Graphics Card (Support at least versions after CUDA 10.0)
- Microsoft Windows 10 (NVIDIA has ceased CUDA driver support for Apple MacOS X)
- Microsoft Visual Studio (Special support for CUDA application)

### Get the Project

- Get the source code from GitHub

    > git clone [https://github.com/1751200/Xlab-k8s-gpu.git](https://github.com/1751200/Xlab-k8s-gpu.git)

### Import the Project to IDE

- Basic CPU versions (jacobi, gauss, SOR, conjugate gradient)
    - Use Visual Studio (2015/16/17/18) to create a `Command Line Application` project.
    - Copy the relevent `.cpp`, `.h` files to the project path `xxx/src`
    - You can also use compilers other than Visual Studio such as dev c++, llvm, ant etc.
    
- CUDA GPU versions (jacobi_gpu, cg_gpu)
    - Use Visual Studio to create a blank `CUDA` project.
    - Copy the relevant `.cu`, `.cuh`, `.cpp`, `.h`, `.lib`, `.exp` files to the project path `xxx/src`
    - You can also use ncvv compilers (requires gcc and g++ on Linux, clang and clang++ on Mac OS X, and cl.exe on Windows)

### Build the Project

In Visual Studio:
- After configuring the compilation, click `Build` to build the project.

In Command Line:
- If using g++ compiler, we strongly recommend you to turn on O3 compiler optimization using `g++ -o3` command.

## Running the Project

- Install npm if missing npm environment
- Enter `project` folder
    ```
    cd project
    ```
- Run `npm start`
- Open your browser and visit `localhost://3000`

## Project Functionalities

- Upload your matrix files (each includes a dense matrix and vector)
- Solve the linear system using different algorithms

## Documentation

We use **GitHub Wiki** for organizing documentation. For the documentation available, see the [homepage](https://github.com/1751200/Xlab-k8s-gpu/wiki) of our Wiki.

## Code Structure

```
.
├── MatMul                         # Solve large matrix multiplication problem
│   ├── CUDA                       # Example provided by CUDA tutorial
│   ├── PyTorch&CuPy               # Cope with matrix multiplication using python libararies
│   └── cuBLAS                     # Solve large matrix multiplication using cuBLAS API
├── Iterative-Methods              # Implementation of iterative methods
│   ├── Basic                      # initialize solvers
│   ├── ConjugateGradient          # Conjugate Gradient method (both CPU and GPU)
│   ├── Jacobi                     # Jacobi method (both CPU and GPU)
│   ├── Gauss-Seidel               # Gauss Seidel method (both CPU and GPU)
│   ├── Sparse-Matrix-CG-Solver    # Solve sparse linear system using cuBLAS and cuSOLVER
│   └── Utils                      # Utils, helps to read matrix data from given file
├── Non-iterative-Methods          # Implementation of iterative methods
│   ├── Eigen                      # Gaussian Elimination, LU, SVD methods using Eigen
│   ├── Gaussian-elimination       # Gaussian Elimination method
│   ├── LU-Decomposition           # LU Decomposition method
│   └── svd-solver                 # SVD Decomposition method
├── project                        # A Node.js app for testing these algorithms
├── Report                         # Documentation for this project
├── README.md
├── LICENSE
└── images
```

## License

This project is licensed under the Apache2.0 License - see the [LICENSE.md](https://github.com/1751200/Xlab-k8s-gpu/blob/master/LICENSE) file for details.

