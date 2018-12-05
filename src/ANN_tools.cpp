#include "ANN_tools.h"

/*矩阵点积*/
float ANN_compute_innerprod(float x1[], float x2[], int len)
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
void ANN_relu_activation(float inp[], int len)
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
void ANN_soft_max(float input[], float result[], int len)
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
void ANN_normalize_feature_zero_mean_unit_sigma(float inp[], float op[], int len)
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

void ANN_Inference(float *input, ANN_weights* pANN_weights)
{
	float prod = 0.0;
	int i = 0;

	// 网络结构:全连接层->ReLu->全连接层->ReLu->全连接层->Softmax
	for (i = 0; i < NUM_FIRST_LAYER_NODES; i++)
	{
		pANN_weights->output_1[i] = pANN_weights->b_1[i];
		prod = ANN_compute_innerprod(input, &pANN_weights->w_1[i][0], NUM_INPUT_NODES);
		pANN_weights->output_1[i] += prod;
	}

	ANN_relu_activation(pANN_weights->output_1, NUM_FIRST_LAYER_NODES);

	for (i = 0; i < NUM_SECOND_LAYER_NODES; i++)
	{
		pANN_weights->output_2[i] = pANN_weights->b_2[i];
		prod = ANN_compute_innerprod(pANN_weights->output_1, &pANN_weights->w_2[i][0], NUM_FIRST_LAYER_NODES);
		pANN_weights->output_2[i] += prod;
	}

	ANN_relu_activation(pANN_weights->output_2, NUM_SECOND_LAYER_NODES);

	for (i = 0; i < NUM_THIRD_LAYER_NODES; i++)
	{
		pANN_weights->output_3[i] = pANN_weights->b_3[i];
		prod = ANN_compute_innerprod(pANN_weights->output_2, &pANN_weights->w_3[i][0], NUM_SECOND_LAYER_NODES);
		pANN_weights->output_3[i] += prod;
	}

	ANN_soft_max(pANN_weights->output_3, pANN_weights->result, NUM_THIRD_LAYER_NODES);
}