#ifndef UTILITY_H
#define UTILITY_H

const int MAX_DIM = 15;

#define GET_SUB_GRID(tileIndex, subDim) ((tileIndex / subDim / subDim) / subDim) * subDim + ((tileIndex % (subDim * subDim)) / subDim)
//#define GET_SUB_GRID(tileIndex, subDim, dim) ((tileIndex / dim) / subDim) * subDim + ((tileIndex % dim) / subDim)
// These are the same

#endif

