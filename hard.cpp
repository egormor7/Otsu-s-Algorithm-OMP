#include <iostream>
#include <fstream>
#include <omp.h>
#include <cstdint>
#include <vector>
#include <string>

constexpr auto L = 256;

using namespace std;

int parseToInt(string s)
{
	int res = 0;
	int n = s.length();
	int i = 0;
	bool isNegative = false;
	if (s[i] == '-') {
		isNegative = true;
		i += 1;
	}
	while (i < n)
	{
		if (s[i] > '9' || s[i] < '0') {
			throw invalid_argument("Error, Illegal argument for number of threads");
		}
		res *= 10;
		res += s[i] - '0';
		i += 1;
	}
	return isNegative ? -res : res;
}

int main(int argc, char** argv)
{
	if (argc != 4) {
		throw invalid_argument("Error, Illegal number of aruments. Expected: 4 Actual: " + 
			to_string(argc));
	}
	int N_THREADS = parseToInt(string(argv[1]));
	if (N_THREADS < -1) {
		throw invalid_argument("Error, Illegal number of threads. Expected integer >= -1");
	}
	string inputFileName = string(argv[2]);
	string outputFileName = string(argv[3]);

	fstream inputFile(inputFileName, ios::in | ios::out | ios::binary);
	if (!inputFile) {
		inputFile.close();
		throw invalid_argument("Error, no such file in the directory");
	}
	char ch1, ch2;
	inputFile >> ch1 >> ch2;			// reading P5 
	if (ch1 != 'P' || ch2 != '5') {
		inputFile.close();
		throw invalid_argument("Illegal file format, expected P5 pgm file");
	}
	int width, height;
	inputFile >> width >> height;		// reading width, height
	int r;
	inputFile >> r;						// reading 255
	char currentByte;
	inputFile.read(&currentByte, 1);	// reading /n
	vector<char> inputBytesArray(width * height);
	for (int i = 0; i < width * height; i++) {		// reading bytes from file
		inputFile.read(&currentByte, 1);
		inputBytesArray[i] = currentByte;
	}
	double timeStart = omp_get_wtime();
	if (N_THREADS > 0) {
		omp_set_num_threads(N_THREADS);
	}
	vector<double> gnf(L);						// global nf - shared for each thread
#pragma omp parallel if (N_THREADS != -1)
	{
		vector<double> lnf(L);					// local nf for each thread
#pragma omp for schedule(static, 1)
		for (int i = 0; i < width * height; i+=1) {		// filling nf array 
			// ( number of occurances of each brightness in file )
			lnf[(unsigned char)inputBytesArray[i]]++;
		}
#pragma omp critical 
		{
			for (int i = 0; i < L; i++) {
				gnf[i] += lnf[i];
			}
		}
	}
	vector<double> p(L);		// creating and filling array of frequenties 
	double N = width * height;
	for (int i = 0; i < L; i++) {
		p[i] = gnf[i] / N;
	}
	vector<double> prefSumOfP_i(L);			// prefix sum array of p[i]
	prefSumOfP_i[0] = p[0];
	for (int i = 1; i < L; i++) {
		prefSumOfP_i[i] = p[i] + prefSumOfP_i[i - 1];
	}

	vector<double> prefSumOfIMulP_i(L);		// prefix sum array of i * p[i]
	for (int i = 1; i < L; i++) {
		prefSumOfIMulP_i[i] = p[i] * i + prefSumOfIMulP_i[i - 1];
	}

	int gf1, gf2, gf3;
	double gmaxDisp = 0;
#pragma omp parallel if (N_THREADS != -1)
	{
		int lf1, lf2, lf3;
		double maxDisp = 0;
#pragma omp for schedule(static, 1)
		for (int f1 = 0; f1 < L - 3; f1 += 1) {

			double q1 = prefSumOfP_i[f1];
			double mu1 = prefSumOfIMulP_i[f1] / q1;

			for (int f2 = f1 + 1; f2 < L - 2; f2++) {

				double q2 = prefSumOfP_i[f2] - prefSumOfP_i[f1];
				double mu2 = (prefSumOfIMulP_i[f2] - prefSumOfIMulP_i[f1]) / q2;

				for (int f3 = f2 + 1; f3 < L - 1; f3++) {
					double q3 = prefSumOfP_i[f3] - prefSumOfP_i[f2];
					double mu3 = (prefSumOfIMulP_i[f3] - prefSumOfIMulP_i[f2]) / q3;

					double q4 = prefSumOfP_i[L - 1] - prefSumOfP_i[f3];
					double mu4 = (prefSumOfIMulP_i[L - 1] - prefSumOfIMulP_i[f3]) / q4;

					double curDisp = (q1 * mu1 * mu1) + (q2 * mu2 * mu2) + 
						(q3 * mu3 * mu3) + (q4 * mu4 * mu4);

					if (curDisp > maxDisp) {
						maxDisp = curDisp;
						lf1 = f1;
						lf2 = f2;
						lf3 = f3;
					}
				}
			}
		}
#pragma omp critical
		{
			if (maxDisp > gmaxDisp) {
				gmaxDisp = maxDisp;
				gf1 = lf1;
				gf2 = lf2;
				gf3 = lf3;
			}
		}
	}
	printf("%u %u %u\n", gf1, gf2, gf3);
#pragma omp parallel if (N_THREADS != -1)
	{
		int id = omp_get_thread_num();
		if (id == 0) {
			N_THREADS = omp_get_num_threads();
		}
#pragma omp for schedule(static, 1)
		for (int i = 0; i < width * height; i++) {
			int currentBrightness = (unsigned char)inputBytesArray[i];
			if (0 <= currentBrightness && currentBrightness <= gf1) {
				inputBytesArray[i] = 0;
			}
			else if (gf1 < currentBrightness && currentBrightness <= gf2) {
				inputBytesArray[i] = 84;
			}
			else if (gf2 < currentBrightness && currentBrightness <= gf3) {
				inputBytesArray[i] = 170;
			}
			else if (gf3 < currentBrightness && currentBrightness <= 255) {
				inputBytesArray[i] = 255;
			}
		}
	}
	double timeEnd = omp_get_wtime();
	printf("Time (%i thread(s)): %g ms\n", N_THREADS, (timeEnd - timeStart) * 1000);
	ofstream outputFile(outputFileName, std::ios::out | std::ios::binary);
	outputFile << "P5\n";
	outputFile << width << " " << height << "\n";
	outputFile << 255 << "\n";

	outputFile.write((char*)&inputBytesArray[0], 
		(sizeof inputBytesArray[0]) * inputBytesArray.size());
	outputFile.close();
	inputFile.close();
	return 0;
}
