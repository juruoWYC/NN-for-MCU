#ifndef CNN_C_H
#define CNN_C_H

#define NUM_LAYERS 5

#define INPUT_X 12
#define INPUT_Y 6

#define LAYER_1_CNN_W 16
#define LAYER_1_CNN_C 1
#define LAYER_1_CNN_X 3
#define LAYER_1_CNN_Y 3

#define LAYER_2_CNN_W 8
#define LAYER_2_CNN_C 16
#define LAYER_2_CNN_X 3
#define LAYER_2_CNN_Y 3

#define LAYER_3_POOL_INPUT_X INPUT_X
#define LAYER_3_POOL_INPUT_Y INPUT_Y
#define LAYER_3_POOL_OUTPUT_X INPUT_X/2
#define LAYER_3_POOL_OUTPUT_Y INPUT_Y/2
#define LAYER_3_POOL_C LAYER_2_CNN_W

#define FALTTEN_NODES (LAYER_3_POOL_OUTPUT_X * LAYER_3_POOL_OUTPUT_Y * LAYER_3_POOL_C)
#define LAYER_4_ANN_NODES 32
#define LAYER_5_ANN_NODES 4


struct CNN_weights
{
	float w_1[LAYER_1_CNN_W][LAYER_1_CNN_C][LAYER_1_CNN_Y][LAYER_1_CNN_X];
	float b_1[LAYER_1_CNN_W];
	float w_2[LAYER_2_CNN_W][LAYER_2_CNN_C][LAYER_2_CNN_Y][LAYER_2_CNN_X];
	float b_2[LAYER_2_CNN_W];
	float w_3[LAYER_4_ANN_NODES][FALTTEN_NODES];
	float b_3[LAYER_4_ANN_NODES];
	float w_4[LAYER_5_ANN_NODES][LAYER_4_ANN_NODES];
	float b_4[LAYER_5_ANN_NODES];
	float output_1[INPUT_Y + 2][INPUT_X + 2][LAYER_1_CNN_W];
	float output_2[INPUT_Y + 2][INPUT_X + 2][LAYER_2_CNN_W];
	//float max_pool[LAYER_3_POOL_OUTPUT_Y][LAYER_3_POOL_OUTPUT_X][LAYER_3_POOL_C];
	float faltten[FALTTEN_NODES];
	float output_3[LAYER_4_ANN_NODES];
	float output_4[LAYER_5_ANN_NODES];
	float result[LAYER_5_ANN_NODES];
};

#endif