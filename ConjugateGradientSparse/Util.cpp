#include"Util.h"

void input(int *I, int *J, float *val, std::string rFile, int nz, int N) {
	std::ifstream  fin(rFile);
	int temp;
	fin >> temp;
	for (int i = 0; i < N + 1; i++) {
		fin >> I[i];
	}
	for (int i = 0; i < nz; i++) {
		fin >> J[i];
	}
	for (int i = 0; i < nz; i++) {
		fin >> val[i];
	}
	fin.close();
}
/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, float *val, int N, int nz)
{
	I[0] = 0, J[0] = 0, J[1] = 1;
	val[0] = (float)rand() / RAND_MAX + 10.0f;
	val[1] = (float)rand() / RAND_MAX;
	int start;

	for (int i = 1; i < N; i++)
	{
		if (i > 1)
		{
			I[i] = I[i - 1] + 3;
		}
		else
		{
			I[1] = 2;
		}

		start = (i - 1) * 3 + 2;
		J[start] = i - 1;
		J[start + 1] = i;

		if (i < N - 1)
		{
			J[start + 2] = i + 1;
		}

		val[start] = val[start - 1];
		val[start + 1] = (float)rand() / RAND_MAX + 10.0f;

		if (i < N - 1)
		{
			val[start + 2] = (float)rand() / RAND_MAX;
		}
	}

	I[N] = nz;
}
void output(std::string wfile, float *x, int size)
{
	int n = size;
	std::ofstream fout(wfile);
	for (int i = 0; i < n; i++)
	{
		fout << x[i] << "   ";
		//cout << x[i] << "   ";
	}
	fout.close();
}
