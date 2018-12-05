#include <iostream>
#include <fstream>
#include "CNN_tools.h"
using namespace std;

int main()
{
	// 初始化权重矩阵
	CNN_weights model = {
		#include "CNN_weights.h"
		{0}, {0}, {0}, {0}, {0}, {0}
	};
	
	ifstream fin("test_data_2_input.txt");
	float input[6 + 2][12 + 2][1] = {0};
	for (int i = 1; i <= 6; i++)
	{
		for (int j = 1; j <= 12; j++)
		{
			fin >> input[i][j][0];
		}
	}
	Inference(&input[0][0][0], &model);// 推理
	for (int i = 0; i < 4; i++)
	{
		cout << model.result[i] << ' ';
	}

	return 0;
}