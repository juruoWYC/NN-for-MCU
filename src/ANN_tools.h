#ifndef ANN_TOOLS_H
#define ANN_TOOLS_H

#include <math.h>
#include "ANN_C.h"

float ANN_compute_innerprod(float x1[], float x2[], int len);
void ANN_relu_activation(float inp[], int len);
void ANN_soft_max(float input[], float result[], int len);
void ANN_normalize_feature_zero_mean_unit_sigma(float inp[], float op[], int len);
void ANN_Inference(float *input, ANN_weights* pANN_weights);

#endif