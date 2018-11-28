#include <iostream>
#include <fstream>
#include "ANN_tools.h"
using namespace std;

int main()
{
	// 初始化权重矩阵
	ANN_weights model = {
		#include "ANN_weights.h"
		{0}, {0}, {0}, {0}
	};
	
	ifstream fin("test_data_1_input.txt");
	float input[72];
	for (int i = 0; i < 72; i++)
	{
		fin >> input[i];
	}
	Inference(input, &model);// 推理
	for (int i = 0; i < 4; i++)
	{
		cout << model.result[i] << ' ';
	}

	return 0;
}