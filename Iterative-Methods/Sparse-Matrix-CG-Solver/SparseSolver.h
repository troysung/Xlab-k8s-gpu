#ifndef SPARSESOLVER_H_
#define SPARSESOLVER_H_

#include <string>

extern int conjugateGradientUM(int argc, char **argv, int size, std::string rfile,float eps, std::string wFile );

/* genTridiag: generate a random tridiagonal symmetric matrix */
extern int conjugateGradient(int argc, char **argv, int size, std::string rfile, float eps, std::string wFile);

extern int conjugateGradientCudaGraphs(int argc, char **argv, int size, std::string rfile, float eps, std::string wFile);

#endif
