#ifndef ANN_C_H
#define ANN_C_H

#define MAX_LAYERS 32

#define NUM_LAYERS 3
#define NUM_INPUT_NODES 72
#define NUM_FIRST_LAYER_NODES 64
#define NUM_SECOND_LAYER_NODES 16
#define NUM_THIRD_LAYER_NODES 4

struct ANN_model
{
	int num_layers = NUM_LAYERS;
	int num_input_nodes = NUM_INPUT_NODES;
	int num_first_layer_nodes = NUM_FIRST_LAYER_NODES;
	int num_second_layer_nodes = NUM_SECOND_LAYER_NODES;
	int num_third_layer_nodes = NUM_THIRD_LAYER_NODES;
};

struct ANN_weights
{
	float w_1[NUM_FIRST_LAYER_NODES][NUM_INPUT_NODES];
	float b_1[NUM_FIRST_LAYER_NODES];
	float w_2[NUM_SECOND_LAYER_NODES][NUM_FIRST_LAYER_NODES];
	float b_2[NUM_SECOND_LAYER_NODES];
	float w_3[NUM_THIRD_LAYER_NODES][NUM_SECOND_LAYER_NODES];
	float b_3[NUM_THIRD_LAYER_NODES];
	float output_1[NUM_FIRST_LAYER_NODES];
	float output_2[NUM_SECOND_LAYER_NODES];
	float output_3[NUM_THIRD_LAYER_NODES];
	float result[NUM_THIRD_LAYER_NODES];
};

#endif