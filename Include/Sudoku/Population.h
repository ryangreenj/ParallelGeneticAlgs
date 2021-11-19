#ifndef SUDOKU_POPULATION_H
#define SUDOKU_POPULATION_H

#include <ostream>
#include <vector>

#include "Sudoku/Board.h"
#include "Sudoku/Chromosome.h"
#include "Utility.h"

class Population
{
public:
    Population() {}
    Population(Board *boardIn, int numChromosomes) { GeneratePopulation(boardIn, numChromosomes); }
    Population(int numGenesIn, bool *lockedGenesIn, std::vector<Chromosome *> chromosomesIn);
    ~Population() { if (lockedGenes) { delete[]lockedGenes; lockedGenes = nullptr; } }

    bool AddChromosome(Chromosome *chromosomeIn);
    int GetSize();
    int GetNumGenes();
    bool* GetLockedGenes(); // Array of size numGenes, false=unlocked  true=locked

    bool GeneratePopulation(Board *boardIn, int numChromosomes);
    byte* FlattenPopulationToArray(int &popSizeOut, int &numGenesOut); // Returned pointer NEEDS to be deleted by user
    void PrintPopulation(std::ostream &out);

private:
    int numGenes = 0;
    bool *lockedGenes = nullptr;
    std::vector<Chromosome*> chromosomes;
};

#endif