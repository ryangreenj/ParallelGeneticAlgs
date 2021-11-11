#ifndef SUDOKU_KERNEL_CUH
#define SUDOKU_KERNEL_CUH

#include "Utility.h"
#include "Board.h"

// Determine if a tile can be filled in immediately based on the other tiles in the row/column/subgrid
Board* PredetermineTiles(Board *boardIn);

#endif