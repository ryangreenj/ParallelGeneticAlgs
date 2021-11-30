#ifndef SUDOKU_KERNEL_CUH
#define SUDOKU_KERNEL_CUH

#include "Utility.h"
#include "Sudoku/Board.h"
#include "Sudoku/Population.h"

// Determine if a tile can be filled in immediately based on the other tiles in the row/column/subgrid
Board* PredetermineTiles(Board *boardIn);

// Calculate the fitness of a given population, return scores of length popIn->GetSize()
int* RankFitness(int numChromosomes, int numGenes, char *flattenedPop, int *fitnessCount);

Population* Breed(Population *popIn);

#endif