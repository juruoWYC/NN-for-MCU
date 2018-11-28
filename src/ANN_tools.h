#ifndef ANN_TOOLS_H
#define ANN_TOOLS_H

#include <math.h>
#include "ANN_C.h"

float compute_innerprod(float x1[], float x2[], int len);
void relu_activation(float inp[], int len);
void soft_max(float input[], float result[], int len);
void normalize_feature_zero_mean_unit_sigma(float inp[], float op[], int len);
void Inference(float *input, ANN_weights* pANN_weights);

#endif