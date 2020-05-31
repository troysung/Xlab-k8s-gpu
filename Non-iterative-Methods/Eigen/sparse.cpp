// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include <iostream>
using namespace std;
#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
//映入Cholesky
#include <Eigen/Cholesky>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/SparseQR>
#include <Eigen/SparseLU>
#include <Eigen/Sparse>
using namespace Eigen;
using namespace std;

#define  MATRIX_SIZE 1500
int main(int argc, char** argv)
{
    srand((unsigned)time(NULL));
    // 解方程
    // 我们求解 A * x = b 这个方程
    // 直接求逆自然是最直接的，但是求逆运算量大

    //Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > A1;
    //A1 = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    //SparseMatrix<double> A1(MATRIX_SIZE, MATRIX_SIZE);
    //Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > b1;
    //b1 = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    //Cholesky 解方程
    Eigen::SparseMatrix<double> smA(MATRIX_SIZE, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            if (i == j) {
                smA.insert(i, j) = rand();
                continue;
            }
            if (rand() / double(RAND_MAX) < 0.002) {
                smA.insert(i, j) = rand();
            }
        }
    }

    smA.makeCompressed();
    Eigen::SparseMatrix<double> b(MATRIX_SIZE, 1);
    for (int i = 0; i < MATRIX_SIZE; i++) {
        b.insert(i, 0) = rand();
    }
    b.makeCompressed();
    Eigen::SparseMatrix<double> x(MATRIX_SIZE, 1);
    cout << "已经赋值" << endl;


    SparseLU < SparseMatrix < double >, AMDOrdering < int > > lu;
    clock_t time_stt = clock();
    lu.analyzePattern(smA);
    //lu.compute(smA);
    lu.factorize(smA);
    x = lu.solve(b);
    cout << "time use in sparse LU decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

    SparseQR < SparseMatrix < double >, AMDOrdering < int > > qr;
    qr.compute(smA);
    x=qr.solve(b);
    cout << "time use in sparse QR decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;

    SimplicialLDLT < SparseMatrix < double >> chloe;
    time_stt = clock();
    chloe.compute(smA);
    x = chloe.solve(b);
    cout << "time use in sparse Cholesky decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;


    //solver.analyzePattern(A);
    // Compute the numerical factorization 
    //solver.factorize(A);
    //Use the factors to solve the linear system 
    //x = solver.solve(b);


}
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
