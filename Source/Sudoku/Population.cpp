#include "Sudoku/Population.h"

#include <cstdlib>

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
    // Can probably parellelize this function as well
    int dim = boardIn->GetDimension();
    numGenes = dim * dim;

    lockedGenes = new bool[numGenes];

    std::vector<int> numValsLeft = std::vector<int>(dim, dim); // When dim=9 [9 ones left, 9 twos left, ..., 9 nines left]

    byte *board = boardIn->GetBoardPointer();

    for (int iTile = 0; iTile < numGenes; ++iTile)
    {
        if (board[iTile] != 0)
        {
            lockedGenes[iTile] = true;
            --numValsLeft[board[iTile] - 1];
        }
        else
        {
            lockedGenes[iTile] = false;
        }
    }

    for (int iChromosome = 0; iChromosome < numChromosomes; ++iChromosome)
    {
        std::vector<int> numValsLeftCopy = numValsLeft;
        byte *newBoard = new byte[numGenes];

        std::vector<int> valsLeft = std::vector<int>(dim); // [1, 2, 3, ..., dim]
        for (int i = 0; i < dim; ++i)
        {
            valsLeft[i] = i + 1;
        }

        for (int iTile = 0; iTile < numGenes; ++iTile)
        {
            if (board[iTile] != 0)
            {
                newBoard[iTile] = board[iTile];
            }
            else
            {
                int tryIndex = std::rand() % valsLeft.size();
                int tryVal = valsLeft[tryIndex];

                --numValsLeftCopy[tryVal - 1];
                newBoard[iTile] = tryVal;

                if (numValsLeftCopy[tryVal - 1] <= 0)
                {
                    valsLeft.erase(valsLeft.begin() + tryIndex);
                }
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