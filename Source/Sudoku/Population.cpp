#include "Sudoku/Population.h"

#include <cstdlib>
#include <numeric>

Population::Population(int numGenesIn, bool *lockedGenesIn, std::vector<Chromosome *> chromosomesIn)
{
    numGenes = numGenesIn;
    lockedGenes = lockedGenesIn;
    for (Chromosome *c : chromosomesIn)
    { 
       chromosomes.push_back(c);
    }
}

bool Population::AddChromosome(Chromosome *chromosomeIn)
{
    if (numGenes == 0)
    {
        numGenes = chromosomeIn->GetNumGenes();
    }

    if (numGenes == chromosomeIn->GetNumGenes())
    {
        chromosomes.push_back(chromosomeIn);
        return true;
    }
    else
    {
        return false;
    }
}

int Population::GetSize()
{
    return chromosomes.size();
}

int Population::GetNumGenes()
{
    return numGenes;
}

bool* Population::GetLockedGenes()
{
    return lockedGenes;
}

bool Population::GeneratePopulation(Board *boardIn, int numChromosomes)
{
    int dim = boardIn->GetDimension();
    int subDim = sqrt(dim);
    numGenes = dim * dim;

    lockedGenes = new bool[numGenes];

    std::vector<int> vals = std::vector<int>(dim);
    std::iota(vals.begin(), vals.end(), 1); // Fill vals with [1, 2, ..., dim]

    std::vector<std::vector<int>> subGridVals = std::vector<std::vector<int>>(dim, vals);

    byte *board = boardIn->GetBoardPointer();

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
        byte *newBoard = new byte[numGenes];

        for (int iTile = 0; iTile < numGenes; ++iTile)
        {
            if (board[iTile] != 0)
            {
                newBoard[iTile] = board[iTile];
            }
            else
            {
                int subGrid = GET_SUB_GRID(iTile, subDim);
                int tryIndex = std::rand() % subGridValsCopy[subGrid].size();

                newBoard[iTile] = subGridValsCopy[subGrid][tryIndex];
                subGridValsCopy[subGrid].erase(subGridValsCopy[subGrid].begin() + tryIndex);
            }
        }

        Chromosome *newChromosome = new Chromosome(numGenes, newBoard);

        chromosomes.push_back(newChromosome);
    }
    return true;
}

byte* Population::FlattenPopulationToArray(int &popSizeOut, int &numGenesOut)
{
    popSizeOut = GetSize();
    numGenesOut = GetNumGenes();

    byte *flattened = new byte[popSizeOut * numGenesOut];

    for (int iChromosome = 0; iChromosome < popSizeOut; ++iChromosome)
    {
        byte *board = chromosomes[iChromosome]->GetBoardPointer();
        
        if (board)
        {
            for (int iGene = 0; iGene < numGenesOut; ++iGene)
            {
                flattened[iChromosome * numGenesOut + iGene] = board[iGene];
            }
        }
        else // board should never be null
        {
            delete[] flattened;
            return nullptr;
        }
    }

    return flattened;
}

void Population::PrintPopulation(std::ostream &out)
{
    int dimension = sqrt(numGenes);

    for (int i = 0; i < chromosomes.size(); ++i)
    {
        out << "Chromosome " << i + 1 << "\n";

        for (int y = 0; y < dimension; ++y)
        {
            for (int x = 0; x < dimension; ++x)
            {
                out << (int)chromosomes[i]->GetBoardPointer()[y * dimension + x] << ' ';
            }
            out << '\n';
        }
        out << '\n';
    }
}