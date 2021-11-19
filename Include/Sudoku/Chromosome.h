#ifndef SUDOKU_CHROMOSOME_H
#define SUDOKU_CHROMOSOME_H

#include "Utility.h"

class Chromosome
{
public:
    Chromosome() {}
    Chromosome(int numGenesIn, byte *boardIn) { numGenes = numGenesIn; board = boardIn; }
    ~Chromosome() { if (board) { delete[]board; board = nullptr; } }

    int GetNumGenes() { return numGenes; }
    byte *GetBoardPointer() { return board; }
private:
    int numGenes = 0;
    byte *board = nullptr;
};

#endif