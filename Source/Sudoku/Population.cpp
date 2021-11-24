#include "Sudoku/Population.h"

#include <algorithm>
#include <cstdlib>
#include <math.h>
#include <numeric>

Population::Population(int numGenesIn, int numChromosomesIn, std::shared_ptr<bool[]> lockedGenesIn, char *flattenedPopulationIn)
{
    numGenes = numGenesIn;
    numChromosomes = numChromosomesIn;
    lockedGenes = lockedGenesIn;
    flattenedPopulation = flattenedPopulationIn;
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
