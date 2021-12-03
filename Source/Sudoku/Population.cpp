#include "Sudoku/Population.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <numeric>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <array>
#include <vector>

#include <iterator>

Population::Population(int numGenesIn, int numChromosomesIn, std::shared_ptr<bool[]> lockedGenesIn, char *flattenedPopulationIn)
{
    numGenes = numGenesIn;
    numChromosomes = numChromosomesIn;
    lockedGenes = lockedGenesIn;
    flattenedPopulation = flattenedPopulationIn;
}

Population::Population(const Population &popToCopy)
{
    numGenes = popToCopy.numGenes;
    numChromosomes = popToCopy.numChromosomes;
    lockedGenes = popToCopy.lockedGenes;
    flattenedPopulation = new char[numChromosomes * numGenes];

    for (int i = 0; i < (numChromosomes * numGenes); ++i)
    {
        flattenedPopulation[i] = popToCopy.flattenedPopulation[i];
    }
}

int Population::GetSize()
{
    return numChromosomes;
}

int Population::GetNumGenes()
{
    return numGenes;
}

std::shared_ptr<bool[]> Population::GetLockedGenes()
{

    return lockedGenes;
}

bool Population::GeneratePopulation(Board *boardIn, int numChromosomesIn)
{
    numChromosomes = numChromosomesIn;
    int dim = boardIn->GetDimension();
    int subDim = sqrt(dim);
    numGenes = dim * dim;

    flattenedPopulation = new char[numChromosomes * numGenes];

    lockedGenes = std::shared_ptr<bool[]>(new bool[numGenes]);

    std::vector<int> vals = std::vector<int>(dim);
    std::iota(vals.begin(), vals.end(), 1); // Fill vals with [1, 2, ..., dim]

    std::vector<std::vector<int>> subGridVals = std::vector<std::vector<int>>(dim, vals);

    char *board = boardIn->GetBoardPointer();

    for (int iTile = 0; iTile < numGenes; ++iTile)
    {
        if (board[iTile] != 0)
        {
            lockedGenes[iTile] = true;
            int subGrid = GET_SUB_GRID(iTile, subDim);
            subGridVals[subGrid].erase(std::remove(subGridVals[subGrid].begin(), subGridVals[subGrid].end(), board[iTile]), subGridVals[subGrid].end());
        }
        else
        {
            lockedGenes[iTile] = false;
        }
    }

    for (int iChromosome = 0; iChromosome < numChromosomes; ++iChromosome)
    {
        auto subGridValsCopy = subGridVals;

        for (int iTile = 0; iTile < numGenes; ++iTile)
        {
            if (board[iTile] != 0)
            {
                flattenedPopulation[iChromosome * numGenes + iTile] = board[iTile];
            }
            else
            {
                int subGrid = GET_SUB_GRID(iTile, subDim);
                int tryIndex = std::rand() % subGridValsCopy[subGrid].size();

                flattenedPopulation[iChromosome * numGenes + iTile] = subGridValsCopy[subGrid][tryIndex];
                subGridValsCopy[subGrid].erase(subGridValsCopy[subGrid].begin() + tryIndex);
            }
        }
    }
    return true;
}

char* Population::FlattenPopulationToArray(int &popSizeOut, int &numGenesOut, bool doCopy)
{
    popSizeOut = GetSize();
    numGenesOut = GetNumGenes();

    if (doCopy)
    {
        char *flattened = new char[popSizeOut * numGenesOut];

        for (int i = 0; i < (popSizeOut * numGenesOut); ++i)
        {
            flattened[i] = flattenedPopulation[i];
        }

        return flattened;
    }
    else
    {
        return flattenedPopulation;
    }
}

char* Population::FlattenPopulationToArrayShuffle(int &popSizeOut, int &numGenesOut, bool doCopy)
{
    popSizeOut = GetSize();
    numGenesOut = GetNumGenes();

    if (doCopy)
    {
        char *flattened = new char[popSizeOut * numGenesOut];

        int *order = new int[popSizeOut];

        std::iota(order, order + popSizeOut, 0); // Fill vals with [0, 1, ..., popSizeOut-1]

        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::shuffle(order, order + popSizeOut, std::default_random_engine(seed));

        for (int i = 0; i < popSizeOut; ++i)
        {
            for (int j = 0; j < numGenesOut; ++j)
            {
                flattened[(order[i] * GetNumGenes()) + j] = flattenedPopulation[(i * GetNumGenes()) + j];
            }
        }

        delete order;
        return flattened;
    }
    else
    {
        return flattenedPopulation;
    }
}

void Population::PrintPopulation(std::ostream &out)
{
    int dimension = sqrt(numGenes);

    for (int i = 0; i < numChromosomes; ++i)
    {
        out << "Chromosome " << i + 1 << "\n";

        for (int y = 0; y < dimension; ++y)
        {
            for (int x = 0; x < dimension; ++x)
            {
                out << (int)flattenedPopulation[i * numGenes + y * dimension + x] << ' ';
            }
            out << '\n';
        }
        out << '\n';
    }
}
