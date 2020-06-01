#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector <double **> submatrix;
vector <int> matrix_size;
vector <double> b;
vector <double> expected_solution;
int g_size, num_submatrix;

int read_file(const string &filename);

int main() {

    read_file("../Equ192.stiff");
//    read_file("../Equ4800.stiff");

    return 0;
}

int read_file(const string &filename) {
    ifstream infile(filename);
    string line;
    double num;

    if(!infile) {
        cerr << "No such file!" << endl;
        return -1;
    }

    // 忽略文件的前两行
    for (int i = 0; i < 3; ++i) {
        getline(infile, line);
    }
    // 将文件中的矩阵元素总个数和子矩阵个数保存
    stringstream in(line);
    in >> g_size >> num_submatrix;

    // 忽略掉格式信息的说明
    for (int i = 0; i < 2; ++i) {
        getline(infile, line);
    }

    // 分别读取各个子矩阵
    for(int m = 0; m < num_submatrix; ++m) {
        // 忽略掉第一行子矩阵的编号
        for (int i = 0; i < 2; ++i) {
            getline(infile, line);
        }
        // 记录并保存下子矩阵的大小
        int size = stoi(line);
        matrix_size.push_back(size);

        // 子矩阵的各行|列元素在总体矩阵A中的位置
        getline(infile, line);
        stringstream s(line);
        int pos;

        // 打印矩阵信息
        cout << "Matrix " << m + 1 << ": " << endl;
        cout << "Size: " << size << "\t Position: ";
        for (int i = 0; i < size; ++i) {
            s >> pos;
            cout << pos << " ";
        }
        cout << endl << endl;

        // 将子矩阵的的各元素值保存在 matirx 数组中
        auto ** matrix = new double*[size];
        for (int i = 0; i < size; ++i) {
            matrix[i] = new double[size];
            getline(infile, line);
            stringstream sin(line);
            for (int j = 0; j < size; ++j) {
                sin >> num;
                matrix[i][j] = num;
            }
        }
        submatrix.push_back(matrix);
    }

    // 保存等号右边向量 F 的值
    for (int i = 0; i < 2; ++i) {
        getline(infile, line);
    }
    stringstream bin(line);
    while (bin >> num) {
        b.push_back(num);
    }
    cout << "The size of F：" << b.size() << endl;

    int cnt_24 = 0;
    int cnt_36 = 0;
    for (auto s : matrix_size) {
        s == 24 ? cnt_24 ++ : cnt_36++;
    }

    // 保存预期解向量的值
    for (int i = 0; i < 2; ++i) {
        getline(infile, line);
    }
    stringstream sin(line);
    while(sin >> num) {
        expected_solution.push_back(num);
    }
    cout << "The size of expected solution: " << expected_solution.size() << endl;

    // 输出大小为 24 的矩阵和 36 的矩阵的个数
    // 并计算总个数与 192 * 192 之间的差距
//    cout << "size_24: " << cnt_24 << "\t size_36: " << cnt_36 << endl;
//    cout << "diff = " << cnt_24 * 24 * 24 + cnt_36 * 36 * 36 - 192 * 192 << endl;

    /**
     * 示例用法
     *
     * double ** matrix = submatrix.at(0);
     * int size = matrix_size.at(0);
     *
     * for (int i = 0; i < size; ++i) {
     *     for (int j = 0; j < size; ++j) {
     *         TODO：sample code here.
     *     }
     * }
     */

    return 0;
}