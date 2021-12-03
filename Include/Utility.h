#ifndef UTILITY_H
#define UTILITY_H

#define MAX_DIM 15
#define NUM_GENERATIONS 1000
#define RANK_RETENTION_RATE 0.5
#define RANDOM_RETENTION_RATE 0.2
#define RUN_SEQUENTIAL true

#define GET_SUB_GRID(tileIndex, subDim) ((tileIndex / subDim / subDim) / subDim) * subDim + ((tileIndex % (subDim * subDim)) / subDim)

#endif

