#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Sudoku/Kernel.cuh"
#include <iostream>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <map>
#include <curand.h>
#include <curand_kernel.h>
#include "..\..\Include\Sudoku\Sequential.h"

/*
void PredetermineTilesKernel(int tileId, int subDim, int dimension, char *boardIn, char *boardOut)
{
    int row = tileId / dimension;
    int col = tileId % dimension;
    int subGrid = GET_SUB_GRID(tileId, subDim);
    int rowOffset = subDim * (subGrid / subDim); // Subgrid tiles logic
    int colOffset = subDim * (subGrid % subDim);

    __shared__ bool modified[MAX_DIM * MAX_DIM];
    __shared__ bool madeChange;

    do
    {
        modified[threadIdx.x] = false;

        if (boardIn[tileId] != 0)
        {
            // Tile already has a set value, skip it
            boardOut[tileId] = boardIn[tileId];
        }
        else
        {
            bool usedNums[MAX_DIM];
            for (int iVal = 0; iVal < dimension; ++iVal)
            {
                // Every tile in column
                char tile = boardIn[iVal * dimension + col];
                if (tile > 0)
                {
                    usedNums[tile - 1] = true;
                }

                // Every tile in row
                tile = boardIn[row * dimension + iVal];
                if (tile > 0)
                {
                    usedNums[tile - 1] = true;
                }

                // Every tile in subgrid, convert iVal into 1D index of board
                tile = boardIn[(rowOffset + (iVal / subDim)) * dimension + colOffset + (iVal % subDim)];
                if (tile > 0)
                {
                    usedNums[tile - 1] = true;
                }
            }

            char candidate = 0;
            for (int i = 0; i < dimension; i++)
            {
                if (!usedNums[i])
                {
                    if (candidate != 0)
                    {
                        // More than one candidate value
                        candidate = 0;
                        break;
                    }
                    else
                    {
                        candidate = i + 1;
                    }
                }
            }

            boardOut[tileId] = candidate;
            modified[threadIdx.x] = candidate != 0;
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            madeChange = false;
            for (int i = 0; i < dimension * dimension; ++i)
            {
                if (modified[i])
                {
                    madeChange = true;
                    break;
                }
            }
        }

        __syncthreads();

        // Running another iteration, copy current output for next input
        if (madeChange)
        {
            boardIn[tileId] = boardOut[tileId];
        }

        __syncthreads();

    } while (madeChange);
}

Board *PredetermineTiles(Board *boardIn)
{
    int dimension = boardIn->GetDimension();
    int subDim = sqrt(dimension);
    char *boardArrIn = boardIn->GetBoardPointer();

    char *dev_boardIn, *dev_boardOut;

    cudaMalloc((void **)&dev_boardIn, dimension * dimension * sizeof(char));
    cudaMalloc((void **)&dev_boardOut, dimension * dimension * sizeof(char));

    cudaMemcpy(dev_boardIn, boardArrIn, dimension * dimension * sizeof(char), cudaMemcpyHostToDevice);

    PredetermineTilesKernel << <1, dimension *dimension >> > (subDim, dimension, dev_boardIn, dev_boardOut);

    char *boardArrOut = new char[dimension * dimension];
    cudaMemcpy(boardArrOut, dev_boardOut, dimension * dimension * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(dev_boardIn);
    cudaFree(dev_boardOut);

    Board *out = new Board(dimension, boardArrOut);
    return out;
}
*/


void RankFitnessKernel(int chromIndex, int geneIndex, int numGenes, int dimension, char *flattenedPop, int *fitnessCount)
{
    int tileId = geneIndex;
    int chromOffset = chromIndex * numGenes;
    int row = tileId / dimension;
    int col = tileId % dimension;

    char currTile = flattenedPop[chromOffset + tileId];

    // Count how many tiles in col/row are same (conflicting) with current tile
    for (int iVal = 0; iVal < dimension; ++iVal)
    {
        int colTile = iVal * dimension + col;
        int rowTile = row * dimension + iVal;

        if (colTile != tileId && flattenedPop[chromOffset + colTile] == currTile)
        {
            ++fitnessCount[chromIndex];
        }

        if (rowTile != tileId && flattenedPop[chromOffset + rowTile] == currTile)
        {
            ++fitnessCount[chromIndex];
        }
    }
}

int *Sequential::RankFitness(int numChromosomes, int numGenes, char *flattenedPop, int *errorCountsOut)
{
    int dimension = sqrt(numGenes);

    for (int iChrom = 0; iChrom < numChromosomes; ++iChrom)
    {
        errorCountsOut[iChrom] = 0;
        for (int iGene = 0; iGene < numGenes; ++iGene)
        {
            RankFitnessKernel(iChrom, iGene, numGenes, dimension, flattenedPop, errorCountsOut);
        }
    }

    // Give chromosomes a unique rank based on their errorCount/fitnessScore

    // Using std::sort is O(n log n), previous algorithm was O(n^2)
    // <fitnessScore, chromosomeIndex>
    std::vector<std::pair<int, int>> scoreToIndex = std::vector<std::pair<int, int>>();

    for (int i = 0; i < numChromosomes; ++i)
    {
        scoreToIndex.push_back(std::make_pair(errorCountsOut[i], i));
    }

    // Sorts by fitnessScore, O(log n)
    std::sort(scoreToIndex.begin(), scoreToIndex.end());

    int *fitnessRank = new int[numChromosomes];

    int i = 0;
    for (auto &pair : scoreToIndex)
    {
        fitnessRank[pair.second] = i++;
    }

    return fitnessRank;
}

void BreedKernel1(int chromIndex, int geneIndex, int numChromosomes, int numGenes, int dimension, int subDim, int seed, char *flattenedPop, int *ranks, bool *lockedIn, int *dev_swap_index, int *dev_swap_candidates, char *dev_tempPopualtion)
{
    // Select top ranked solutions and place them in dev_tempPopualtion
    if (ranks[chromIndex] < (int)(numChromosomes * RANK_RETENTION_RATE))
    {
        int offset = (chromIndex * numGenes) + geneIndex;
        int rank = (ranks[chromIndex] * numGenes) + geneIndex;
        dev_tempPopualtion[rank] = flattenedPop[offset];
    }
}

void BreedKernel2(int chromIndex, int geneIndex, int numChromosomes, int numGenes, int dimension, int subDim, int seed, char *flattenedPop, int *ranks, bool *lockedIn, int *dev_swap_index, int *dev_swap_candidates, char *dev_tempPopualtion)
{
    // This block will pick the chromosomes that will be swaping, via a tournament style selection
    // The two solutions with the best rank will swap genes

    // This could easily be made to use an array and feature more than 4 prospective chomosomes

    if (geneIndex == 0)
    {
        // Select 4 random indexes that within the RANK_RETENTION_RATE
        int c1 = (rand() % (int)(numChromosomes * RANK_RETENTION_RATE));
        int c2 = (rand() % (int)(numChromosomes * RANK_RETENTION_RATE));
        int c3 = (rand() % (int)(numChromosomes * RANK_RETENTION_RATE));
        int c4 = (rand() % (int)(numChromosomes * RANK_RETENTION_RATE));

        // best ranks between c1/c2 and c3/c4 will swap
        dev_swap_candidates[(chromIndex * 2)] = c1 < c2 ? c1 : c2;
        dev_swap_candidates[(chromIndex * 2) + 1] = c3 < c4 ? c3 : c4;

        // decide which subgrid we will start the swap at
        // this number is in the range [2,dimension-1], don't start at first or last subgrid
        dev_swap_index[chromIndex] = (chromIndex % (dimension - 2)) + 1;
    }
}

void BreedKernel3(int chromIndex, int geneIndex, int numChromosomes, int numGenes, int dimension, int subDim, int seed, char *flattenedPop, int *ranks, bool *lockedIn, int *dev_swap_index, int *dev_swap_candidates, char *dev_tempPopualtion)
{

    int offset = ((chromIndex) * numGenes) + geneIndex; // location of gene in flattened population
    int grid = GET_SUB_GRID(geneIndex, subDim); // determines the grid of current thread within a block

    // Get new offset by picking the correct swap candidate
    // if the current grid is less than the swap index, its the first candidate, otherwise its the second
    int offset_new = (dev_swap_candidates[(chromIndex * 2) + (grid < dev_swap_index[chromIndex] ? 0 : 1)] * numGenes) + geneIndex;
    flattenedPop[offset] = dev_tempPopualtion[offset_new];
}

void BreedKernel4(int chromIndex, int geneIndex, int numChromosomes, int numGenes, int dimension, int subDim, int seed, char *flattenedPop, int *ranks, bool *lockedIn, int *dev_swap_index, int *dev_swap_candidates, char *dev_tempPopualtion)
{
    // This carries out a randome number of 'mutations' 
    // a mutation is just swapping to non-locked genes of a subgrid
    if (geneIndex == 0)
    {
        // 1 to 3 mutations will happen on every block (this was randomly decided)
        // I was messing with this a lot and this seems to be best so far
        for (int k = 0; k < (rand() % subDim) + 1; k++)
        {
            // randomly generates a swap index and makes sure that it is not locked gene
            int swap_index_1 = (rand() % numGenes);
            while (lockedIn[swap_index_1]) swap_index_1 = (rand() % numGenes);

            // determines the subgrid number of swap_index_1, then determines the center of that subgrid
            int swap_grid = GET_SUB_GRID(swap_index_1, subDim);

            int rowOffset = subDim * (swap_grid / subDim); // Subgrid tiles logic
            int colOffset = subDim * (swap_grid % subDim);

            int swap_index_2 = swap_index_1;
            int subGridTile = 0;

            do
            {
                // Randomly choose unlocked tile within subgrid
                // Thought of using incremental logic here to guarantee it completes in finite time but that makes certain tiles more probable for swaps depending on locked genes
                subGridTile = rand() % dimension;
                swap_index_2 = (rowOffset + (subGridTile / subDim)) * dimension + colOffset + (subGridTile % subDim);
            } while (swap_index_1 == swap_index_2 || lockedIn[swap_index_2]);

            // swaps swap_index_1 and swap_index_2
            char temp = flattenedPop[(chromIndex * numGenes) + swap_index_1];
            flattenedPop[(chromIndex * numGenes) + swap_index_1] = flattenedPop[(chromIndex * numGenes) + swap_index_2];
            flattenedPop[(chromIndex * numGenes) + swap_index_2] = temp;
        }
    }
}

Population *Sequential::Breed(Population * popIn, int &bestrank, char *bestboard)
{
    int numChromosomes = 0;
    int numGenes = 0;
    std::shared_ptr<bool[]> lockedGenesIn = popIn->GetLockedGenes();

    char *flattenedPop = popIn->FlattenPopulationToArray(numChromosomes, numGenes, true);

    int dimension = sqrt(numGenes);
    int subDim = sqrt(dimension);

    int *errorCounts = new int[numChromosomes];
    int *fitnessRanks = RankFitness(numChromosomes, numGenes, flattenedPop, errorCounts);

    // This is just used for printing the best solution at the end    
    int prev_best = bestrank;
    bestrank = INT_MAX;
    int index = 0;
    for (int e = 0; e < numChromosomes; e++)
    {
        if (errorCounts[e] < bestrank)
        {
            bestrank = errorCounts[e];
            index = e;
            if (bestrank < prev_best)
            {
                for (int i = 0; i < dimension; i++)
                {
                    for (int j = 0; j < dimension; j++)
                    {
                        bestboard[(i * dimension) + j] = flattenedPop[(index * numGenes) + (i * dimension) + j];
                    }
                }
            }
        }

    }

    std::cout << "Best error - " << bestrank << "\n";

    int retention_size = (int)(numChromosomes * RANK_RETENTION_RATE);

    char *dev_tempPopualtion = new char[retention_size * numGenes];
    int *dev_swap_index = new int[numChromosomes];
    int *dev_swap_candidates = new int[numChromosomes * 2];

    for (int iChrom = 0; iChrom < numChromosomes; ++iChrom)
    {
        for (int iGene = 0; iGene < numGenes; ++iGene)
        {
            BreedKernel1(iChrom, iGene, numChromosomes, numGenes, dimension, subDim, (rand() % dimension) + 1, flattenedPop, fitnessRanks, lockedGenesIn.get(), dev_swap_index, dev_swap_candidates, dev_tempPopualtion);
        }
    }

    for (int iChrom = 0; iChrom < numChromosomes; ++iChrom)
    {
        for (int iGene = 0; iGene < numGenes; ++iGene)
        {
            BreedKernel2(iChrom, iGene, numChromosomes, numGenes, dimension, subDim, (rand() % dimension) + 1, flattenedPop, fitnessRanks, lockedGenesIn.get(), dev_swap_index, dev_swap_candidates, dev_tempPopualtion);
        }
    }

    for (int iChrom = 0; iChrom < numChromosomes; ++iChrom)
    {
        for (int iGene = 0; iGene < numGenes; ++iGene)
        {
            BreedKernel3(iChrom, iGene, numChromosomes, numGenes, dimension, subDim, (rand() % dimension) + 1, flattenedPop, fitnessRanks, lockedGenesIn.get(), dev_swap_index, dev_swap_candidates, dev_tempPopualtion);
        }
    }

    for (int iChrom = 0; iChrom < numChromosomes; ++iChrom)
    {
        for (int iGene = 0; iGene < numGenes; ++iGene)
        {
            BreedKernel4(iChrom, iGene, numChromosomes, numGenes, dimension, subDim, (rand() % dimension) + 1, flattenedPop, fitnessRanks, lockedGenesIn.get(), dev_swap_index, dev_swap_candidates, dev_tempPopualtion);
        }
    }

    delete[] dev_tempPopualtion;
    delete[] dev_swap_index;
    delete[] dev_swap_candidates;

    delete[] errorCounts;
    delete[] fitnessRanks;

    Population *out = new Population(numGenes, numChromosomes, lockedGenesIn, flattenedPop);

    return out;
}
