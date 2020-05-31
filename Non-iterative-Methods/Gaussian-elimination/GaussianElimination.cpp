#include <iostream>
#include <vector>
using namespace std;

vector<vector<double>> gaussianElimination(vector<vector<double>> matrixIn, int n)
{
    vector<vector<double>> matrix = matrixIn;

    // Rearrange rows to make sure none of the values on the main diagonal are 0.
    for (int row = 0; row < n; ++row)
    {
        if (matrix[row][row] == 0.0)
        {
            // Iterates to find suitable row to swap with.
            int rowToSwapWith = 0;
            while (rowToSwapWith < n)
            {
                if (matrix[rowToSwapWith][row] != 0.0)
                {
                    break;
                }
                ++rowToSwapWith;
            }

            // Swaps rows
            vector<double> rowTemp = matrix[row];
            matrix[row] = matrix[rowToSwapWith];
            matrix[rowToSwapWith] = rowTemp;
        }
    }

    /*
    Elimination from top to bottom:
        1. Divide the row by a number such that the value on the main diagonal becomes 1.
        2. Subtract a multiple of the row from all rows under it to set all the values under the aformentioned value of the main diagonal to 0.
        3. The matrix should end up in row echelon form.
    */
     for (int row = 0; row < n; ++row)
    {
        double mainDiagonalValue = matrix[row][row];
        for (int col = 0; col < n + 1; ++col)
        {
            matrix[row][col] /= mainDiagonalValue;
        }

        if (row != n - 1)
        {
            for (int rowUnder = row + 1; rowUnder < n; ++rowUnder)
            {
                double multiple = matrix[rowUnder][row];
                for (int col = 0; col < n + 1; ++col)
                {
                    matrix[rowUnder][col] -= matrix[row][col] * multiple;
                }
            }
        }
    }

   /*
    Back substitution from bottom to top:
        1. Subtract a multiple of the row from all rows above it to set all the values above the value on the main diagonal to 0.
        2. The matrix should end up in reduced row echelon form.
    */
    for (int row = n - 1; row > 0; --row)
    {
        for (int rowAbove = row - 1; rowAbove >= 0; --rowAbove)
        {
            double multiple = matrix[rowAbove][row];
            for (int col = 0; col < n + 1; ++col)
            {
                matrix[rowAbove][col] -= matrix[row][col] * multiple;
            }
        }
    }

    // Convert all negative zeroes to zero
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n + 1; ++col)
        {
            if (matrix[row][col] == -0.0)
                {
                matrix[row][col] = 0.0;
                }
        }
    }

    return matrix;
}

int main()
{
    // Gets user input
    cout << "Enter number of variables:" << endl;
    int n;
    cin >> n;
    
    cout << "\nEnter coefficients:" << endl;
    vector<vector<double>> matrix;
    for (int i = 0; i < n; ++i)
    {
        vector<double> row;
        for (int j = 0; j < n + 1; ++j)
        {
            double value;
            cin >> value;
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    matrix = gaussianElimination(matrix, n);

    // Outputs answer
    cout << "\nAnswer:" << endl;
    for (int row = 0; row < n; ++row)
    {
        for (int col = 0; col < n + 1; ++col)
        {
            cout << matrix[row][col] << ' ';
        }
        cout << endl;
    }  

    return 0;
}