#include "CNN_tools.h"
#include <iostream>
using namespace std;

/*卷积计算*/
inline float Conv_3x3_compute(float *x1, float *x2, int x, int y, int c, int X, int Y, int C)
{
	float prod = 0;
	int pos = y * X * C + x * C + c;

	prod += x1[pos - (X + 1) * C] * x2[0];
	prod += x1[pos - X * C] * x2[1];
	prod += x1[pos - (X - 1) * C] * x2[2];
	prod += x1[pos - C] * x2[3];
	prod += x1[pos] * x2[4];
	prod += x1[pos + C] * x2[5];
	prod += x1[pos + (X - 1) * C] * x2[6];
	prod += x1[pos + X * C] * x2[7];
	prod += x1[pos + (X + 1) * C] * x2[8];
	//cout << y << ' '<<x<<' '<<c<<' '<<pos - (X + 1) * C << ' ' << pos - X * C << ' ' << pos - (X - 1) * C << ' ' << pos - C << ' ' << pos << ' ' << pos + C << ' ' << pos + (X - 1) * C << ' ' << pos + X * C << ' ' << pos + (X + 1) * C << ' '<<prod << endl;

	return prod;
}

/*矩阵点积*/
float compute_innerprod(float x1[], float x2[], int len)
{
	float prod = 0;
	int i = 0;

	for (i = 0; i < len; i++)
	{
		prod += x1[i] * x2[i];
	}
	return prod;
}

/*ReLu激活函数*/
void relu_activation(float inp[], int len)
{
	int i = 0;
	for (i = 0; i < len; i++)
	{
		if (inp[i] < 0.0f)
		{
			inp[i] = 0.0f;
		}
	}
}

/*Softmax函数*/
void soft_max(float input[], float result[], int len)
{
	float maxval = input[0];
	float sum = 0.0f;
	int i = 0;
	// 找最大值
	for (i = 1; i < len; i++)
	{
		if (input[i] > maxval) maxval = input[i];
	}
	// 计算指数函数及指数函数和
	for (i = 0; i < len; i++)
	{
		input[i] = (float)exp((double)(input[i] - maxval)); // exp()以double作为输入输出
		sum += input[i];
	}
	for (i = 0; i < len; i++)
	{
		result[i] = input[i] / sum;
	}
}

/*数据归一化*/
void normalize_feature_zero_mean_unit_sigma(float inp[], float op[], int len)
{
	int i;
	float sum = 0, sumsq = 0, eps = 1e-16, mean = 0, sigma = 0;

	for (i = 0; i < len; i++)
	{
		sum += inp[i];
		sumsq += inp[i] * inp[i];
	}
	mean = sum / len;
	sigma = (float)sqrt((double)(sumsq / len - mean * mean)) + eps;

	for (i = 0; i < len; i++)
	{
		op[i] = (inp[i] - mean) / sigma;
	}
}

inline float max_of_two(float a, float b)
{
	if (a >= b) return a;
	else return b;
}

inline float max_of_three(float a, float b, float c)
{
	return max_of_two(max_of_two(a, b), c);
}

inline float max_of_four(float a, float b, float c, float d)
{
	return max_of_two(max_of_three(a, b, c), d);
}

void Inference(float *input, CNN_weights* pANN_weights)
{
	float prod = 0.0;
	int i = 0, w = 0, c = 0, x = 0, y = 0;

	// 网络结构:全连接层->ReLu->全连接层->ReLu->全连接层->Softmax
	memset(pANN_weights->output_1, 0, sizeof(pANN_weights->output_1));
	for (w = 0; w < LAYER_1_CNN_W; w++)
	{
		for (y = 1; y <= INPUT_Y; y++)
		{
			for (x = 1; x <= INPUT_X; x++)
			{
				pANN_weights->output_1[y][x][w] = pANN_weights->b_1[w];
				for (c = 0; c < LAYER_1_CNN_C; c++)
				{
					prod = Conv_3x3_compute(input, &pANN_weights->w_1[w][c][0][0], x, y, c, INPUT_X + 2, INPUT_Y + 2, LAYER_1_CNN_C);
					pANN_weights->output_1[y][x][w] += prod;
				}
			}
		}
	}

	relu_activation(&pANN_weights->output_1[0][0][0], (INPUT_X + 2) * (INPUT_Y + 2) * LAYER_1_CNN_W);

	memset(pANN_weights->output_2, 0, sizeof(pANN_weights->output_2));
	for (w = 0; w < LAYER_2_CNN_W; w++)
	{
		for (y = 1; y <= INPUT_Y; y++)
		{
			for (x = 1; x <= INPUT_X; x++)
			{
				pANN_weights->output_2[y][x][w] = pANN_weights->b_2[w];
				for (c = 0; c < LAYER_2_CNN_C; c++)
				{
					prod = Conv_3x3_compute(&pANN_weights->output_1[0][0][0], &pANN_weights->w_2[w][c][0][0], x, y, c, INPUT_X + 2, INPUT_Y + 2, LAYER_2_CNN_C);
					pANN_weights->output_2[y][x][w] += prod;
				}
			}
		}
	}

	relu_activation(&pANN_weights->output_2[0][0][0], (INPUT_X + 2) * (INPUT_Y + 2) * LAYER_2_CNN_W);

	/*
	//池化层
	for (y = 0; y < LAYER_3_POOL_OUTPUT_Y; y++)
	{
		for (x = 0; x < LAYER_3_POOL_OUTPUT_X; x++)
		{
			for (c = 0; c < LAYER_3_POOL_C; c++)
			{
				pANN_weights->max_pool[y][x][c] = max_of_four(pANN_weights->output_2[y * 2 + 1][x * 2 + 1][c],
															  pANN_weights->output_2[y * 2 + 1][x * 2 + 2][c],
															  pANN_weights->output_2[y * 2 + 2][x * 2 + 1][c],
															  pANN_weights->output_2[y * 2 + 2][x * 2 + 2][c]);
			}
		}
	}
	*/

	//池化+展平层
	for (y = 0; y < LAYER_3_POOL_OUTPUT_Y; y++)
	{
		for (x = 0; x < LAYER_3_POOL_OUTPUT_X; x++)
		{
			for (c = 0; c < LAYER_3_POOL_C; c++)
			{
				pANN_weights->faltten[y * LAYER_3_POOL_OUTPUT_X * LAYER_3_POOL_C + x * LAYER_3_POOL_C + c] = max_of_four(pANN_weights->output_2[y * 2 + 1][x * 2 + 1][c],
					pANN_weights->output_2[y * 2 + 1][x * 2 + 2][c],
					pANN_weights->output_2[y * 2 + 2][x * 2 + 1][c],
					pANN_weights->output_2[y * 2 + 2][x * 2 + 2][c]);
			}
		}
	}

	for (i = 0; i < LAYER_4_ANN_NODES; i++)
	{
		pANN_weights->output_3[i] = pANN_weights->b_3[i];
		prod = compute_innerprod(pANN_weights->faltten, &pANN_weights->w_3[i][0], FALTTEN_NODES);
		pANN_weights->output_3[i] += prod;
	}

	relu_activation(pANN_weights->output_3, LAYER_4_ANN_NODES);

	for (i = 0; i < LAYER_5_ANN_NODES; i++)
	{
		pANN_weights->output_4[i] = pANN_weights->b_4[i];
		prod = compute_innerprod(pANN_weights->output_3, &pANN_weights->w_4[i][0], LAYER_4_ANN_NODES);
		pANN_weights->output_4[i] += prod;
	}

	soft_max(pANN_weights->output_4, pANN_weights->result, LAYER_5_ANN_NODES);
}