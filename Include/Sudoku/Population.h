#ifndef SUDOKU_POPULATION_H
#define SUDOKU_POPULATION_H

#include <memory>
#include <ostream>
#include <vector>

#include "Sudoku/Board.h"
#include "Utility.h"

class Population
{
public:
    Population() {}
    Population(const Population &popToCopy);
    Population(Board *boardIn, int numChromosomesIn) { GeneratePopulation(boardIn, numChromosomesIn); }
    Population(int numGenesIn, int numChromosomesIn, std::shared_ptr<bool[]> lockedGenesIn, char *flattenedPopulationIn);
    ~Population() { if (flattenedPopulation) { delete[]flattenedPopulation; flattenedPopulation = nullptr; } }

    int GetSize();
    int GetNumGenes();
    std::shared_ptr<bool[]> GetLockedGenes(); // Array of size numGenes, false=unlocked  true=locked

    bool GeneratePopulation(Board *boardIn, int numChromosomes);
    char* FlattenPopulationToArray(int &popSizeOut, int &numGenesOut, bool doCopy=true); // Returned pointer NEEDS to be deleted by user IF doCopy == true
    char* FlattenPopulationToArrayShuffle(int &popSizeOut, int &numGenesOut, bool doCopy=true); // Returned pointer NEEDS to be deleted by user IF doCopy == true
    void PrintPopulation(std::ostream &out);

private:
    int numGenes = 0;
    int numChromosomes = 0;
    std::shared_ptr<bool[]> lockedGenes = nullptr;
    char *flattenedPopulation = nullptr;
};

#endif