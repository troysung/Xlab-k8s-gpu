// ConsoleApplication1.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
#include <iostream>
using namespace std;
#include <ctime>
// Eigen 部分
#include <Eigen/Core>
//映入Cholesky
#include <Eigen/Cholesky>
// 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Sparse>
#include <Eigen/Dense>
using namespace Eigen;     // 改成这样亦可 using Eigen::MatrixXd; 
using namespace std;

#define  MATRIX_SIZE 4
/*
int main(int argc, char** argv)
{
    // 解方程
    // 我们求解 A * x = b 这个方程
    // 直接求逆自然是最直接的，但是求逆运算量大

    //Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > A1;
    //A1 = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    SparseMatrix<double> A1(MATRIX_SIZE, MATRIX_SIZE);
    Eigen::Matrix< double, Eigen::Dynamic, Eigen::Dynamic > b1;
    b1 = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_stt = clock(); // 计时
    //Cholesky 解方程


    // 直接求逆
    Eigen::Matrix<double, MATRIX_SIZE, 1> x = A1.inverse() * b1;
    cout << "time use in normal inverse is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    //cout << x << endl;
    // QR分解colPivHouseholderQr()
    //time_stt = clock();
    x = A1.colPivHouseholderQr().solve(b1);
    cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    //cout << x << endl;
    //QR分解fullPivHouseholderQr()
    //time_stt = clock();
    //x = A1.fullPivHouseholderQr().solve(b1);
    //cout << "time use in Qr decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    //cout << x << endl;
    //llt分解 要求矩阵A正定
    time_stt = clock();
    x = A1.llt().solve(b1);
    cout <<"time use in llt decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;
    //ldlt分解  要求矩阵A正或负半定
    time_stt = clock();
    x = A1.ldlt().solve(b1);
    cout <<"time use in ldlt decomposition is " <<1000*(clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
    cout <<x<<endl;
    //lu分解 partialPivLu()
    time_stt = clock();
    x = A1.partialPivLu().solve(b1);
    cout << "time use in lu decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    //cout << x << endl;
    //lu分解（fullPivLu()
    // = clock();
    //x = A1.fullPivLu().solve(b1);
    //cout << "time use in lu decomposition is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    //cout << x << endl;

    //x = A1.bdcSvd(ComputeThinU | ComputeThinV).solve(b1);
    cout << "time use in svd is " << 1000 * (clock() - time_stt) / (double)CLOCKS_PER_SEC << "ms" << endl;
    //cout << x << endl;
    return 0;

}
*/
// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
