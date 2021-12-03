#ifndef SUDOKU_SEQUENTIAL_CUH
#define SUDOKU_SEQUENTIAL_CUH

#include "Utility.h"
#include "Sudoku/Board.h"
#include "Sudoku/Population.h"

// Copy and paste most of the parallel code and make some modifications so it will run sequentially

// Determine if a tile can be filled in immediately based on the other tiles in the row/column/subgrid
//Board *PredetermineTiles(Board *boardIn);

namespace Sequential
{
    // Calculate the fitness of a given population, return scores of length popIn->GetSize()
    int *RankFitness(int numChromosomes, int numGenes, char *flattenedPop, int *fitnessCount, int &fitnessTime);

    Population *Breed(Population *popIn, int &bestrank, char *bestboard);
}
#endif
