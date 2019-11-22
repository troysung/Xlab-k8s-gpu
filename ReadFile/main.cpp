#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

vector <double **> submatrix;
vector <int> matrix_size;
int g_size, num_submatrix;

int read_file(const string &filename);

int main() {

    read_file("../Equation193.stiff");

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
        int pos0 = 0, posl = 0;
        s >> pos0;
        for (int i = 0; i < size - 1; ++i) {
            s >> posl;
        }

        // 打印矩阵信息
        cout << "Matirx " << m + 1 << ": " << endl;
        cout << "Size: " << size << "\t Position: " << pos0 << "~" << posl << endl << endl;

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