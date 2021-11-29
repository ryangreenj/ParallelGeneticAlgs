#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Sudoku/Kernel.cuh"
#include <iostream>
#include <algorithm>    // std::shuffle
#include <array>        // std::array
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

__global__ void PredetermineTilesKernel(int subDim, int dimension, char *boardIn, char *boardOut)
{
    int tileId = threadIdx.x;
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

Board* PredetermineTiles(Board *boardIn)
{
    int dimension = boardIn->GetDimension();
    int subDim = sqrt(dimension);
    char *boardArrIn = boardIn->GetBoardPointer();

    char *dev_boardIn, *dev_boardOut;

    cudaMalloc((void **)&dev_boardIn, dimension * dimension * sizeof(char));
    cudaMalloc((void **)&dev_boardOut, dimension * dimension * sizeof(char));

    cudaMemcpy(dev_boardIn, boardArrIn, dimension * dimension * sizeof(char), cudaMemcpyHostToDevice);
    
    PredetermineTilesKernel<<<1, dimension * dimension>>>(subDim, dimension, dev_boardIn, dev_boardOut);

    char *boardArrOut = new char[dimension * dimension];
    cudaMemcpy(boardArrOut, dev_boardOut, dimension * dimension * sizeof(char), cudaMemcpyDeviceToHost);

    cudaFree(dev_boardIn);
    cudaFree(dev_boardOut);

    Board *out = new Board(dimension, boardArrOut);
    return out;
}



__global__ void RankFitnessKernel(int chromosomes, int dimension, char *flattenedPop, int *fitnessCount)
{
    int tileId = threadIdx.x;
    int chromOffset = blockIdx.x * blockDim.x;
    int row = tileId / dimension;
    int col = tileId % dimension;

    __shared__ int errors[MAX_DIM * MAX_DIM];
    errors[threadIdx.x] = 0;

    char currTile = flattenedPop[chromOffset + tileId];

    // Count how many tiles in col/row are same (conflicting) with current tile
    for (int iVal = 0; iVal < dimension; ++iVal)
    {
        int colTile = iVal * dimension + col;
        int rowTile = row * dimension + iVal;

        if (colTile != tileId && flattenedPop[chromOffset + colTile] == currTile)
        {
            ++errors[threadIdx.x];
        }

        if (rowTile != tileId && flattenedPop[chromOffset + rowTile] == currTile)
        {
            ++errors[threadIdx.x];
        }
    }

    __syncthreads();

    // Parallel reduction
    if (threadIdx.x % dimension == 0)
    {
        for (int i = 1; i < dimension; ++i)
        {
            errors[threadIdx.x] += errors[threadIdx.x + i];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        fitnessCount[blockIdx.x] = 0;
        for (int i = 0; i < dimension; ++i)
        {
            fitnessCount[blockIdx.x] += errors[i * dimension];
        }
    }

}

int* RankFitness(int numChromosomes, int numGenes, char *flattenedPop)
{
    // int numChromosomes = 0;
    // int numGenes = 0;

    // Arguments are output args, filled by function
    // char *flattenedPop = popIn->FlattenPopulationToArray(numChromosomes, numGenes, false);

    int dimension = sqrt(numGenes);

    char *dev_flattenedPop;
    int *dev_fitnessCount;

    cudaMalloc((void **)&dev_flattenedPop, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_fitnessCount, numChromosomes * sizeof(int));

    cudaMemcpy(dev_flattenedPop, flattenedPop, numChromosomes * numGenes * sizeof(char), cudaMemcpyHostToDevice);

    RankFitnessKernel<<<numChromosomes, numGenes>>>(numChromosomes, dimension, dev_flattenedPop, dev_fitnessCount);

    int *fitnessRank = new int[numChromosomes];
    int *fitnessCount = new int[numChromosomes];

    cudaMemcpy(fitnessCount, dev_fitnessCount, numChromosomes * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_flattenedPop);
    cudaFree(dev_fitnessCount);

    // This can be parallelized but I was having some mem issues
    for (int i = 0; i < numChromosomes; ++i)
    {
        fitnessRank[i] = 0;
        for (int j = 0; j < numChromosomes; ++j)
        {   
            if (fitnessCount[i] > fitnessCount[j])
            {
                fitnessRank[i] += 1;
            }
        }
    }

    int* rank_set = new int[numChromosomes];
    // for (int i = 0; i < numChromosomes; ++i)
    // {
    //     rank_set[i] = -999;
    // }
    
    for (int i = 0; i < numChromosomes; ++i)
    {
        bool inSet = false;
        for (int j = 0; j < i; ++j)
        {
            if (rank_set[j] == fitnessRank[i])
            {
                inSet = true;
                break;
            }            
        }

        if (inSet)
        {
            fitnessRank[i] += 1;
        } 
        rank_set[i] = fitnessRank[i];
    }
    return fitnessRank;
}

__global__ void BreedKernel(int numChromosomes, int numGenes, int seed, char *flattenedPop, int *subgrid_swaps, int *gridswap, char *flattenedPopOut)
{
    int grid = ((threadIdx.x % 9) / 3) + ((threadIdx.x / 27) * 3); // Determines the grid of current run
    int offset = ((blockIdx.x) * numGenes) + threadIdx.x;
    int offset_new = ((subgrid_swaps[blockIdx.x]) * numGenes) + threadIdx.x;

    // Determines random subgrid to swap if subgrid swap id is not the block id
    int grid_swap = (blockIdx.x == subgrid_swaps[blockIdx.x]) ? blockIdx.x : (blockIdx.x + subgrid_swaps[blockIdx.x] * seed) % 9;
    flattenedPopOut[offset] = flattenedPop[(grid == grid_swap) ? offset_new : offset ];

    __syncthreads();

    // Random mutations

}

Population* Breed(Population *popIn)
{
    double choices[2] = {0.20, 0.8}; // Keep top 20%, breed next 60%, drop final 20%

    int numChromosomes = 0;
    int numGenes = 0;
    std::shared_ptr<bool[]> lockedGenesIn = popIn->GetLockedGenes();

    // Arguments are output args, filled by function
    // Need to delete returned pointer at end
    char *flattenedPop = popIn->FlattenPopulationToArrayShuffle(numChromosomes, numGenes);

    int* fitnessRanks = RankFitness(numChromosomes, numGenes, flattenedPop);

    char *dev_flattenedPopBreed;
    char *dev_flattenedPopBreedOut;
    int *dev_fitnessRankBreed;
    int *dev_gridSwap;
    int *grid_swap = new int[numChromosomes];

    
    cudaMalloc((void **)&dev_flattenedPopBreed, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_flattenedPopBreedOut, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_fitnessRankBreed, numChromosomes * sizeof(int));
    cudaMalloc((void **)&dev_gridSwap, numChromosomes * sizeof(int));

    int* swaps = new int[numChromosomes];
    bool* stay = new bool[numChromosomes];

    for (int i = 0; i < numChromosomes; i++) stay[i] = fitnessRanks[i] < (int)(.2 * numChromosomes);

    for (int i = 0; i < numChromosomes; i++){
        if(fitnessRanks[i] < (int)(.2 * numChromosomes))
        {
            swaps[i] = i;
        }
        else 
        {
            int i1 = i;
            do
            {
                i++;
                if (i >= numChromosomes) break;
                swaps[i] = i;
            } while (fitnessRanks[i] < (int)(.2 * numChromosomes));
            
            int i2 = i;

            swaps[i1] = i2;
            swaps[i2] = i1;
        }
    }

    cudaMemcpy(dev_flattenedPopBreed, flattenedPop, numChromosomes * numGenes * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fitnessRankBreed, swaps, numChromosomes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gridSwap, grid_swap, numChromosomes * sizeof(int), cudaMemcpyHostToDevice);

    BreedKernel<<<numChromosomes, numGenes>>>(numChromosomes, numGenes, (rand() % 9) + 1, dev_flattenedPopBreed, dev_fitnessRankBreed, dev_gridSwap, dev_flattenedPopBreedOut);
    
    char *popout = new char[numChromosomes * numGenes];
    cudaMemcpy(popout, dev_flattenedPopBreedOut, numChromosomes * numGenes * sizeof(char), cudaMemcpyDeviceToHost);
    
    Population *out = new Population(numGenes, numChromosomes, lockedGenesIn, popout);
    return out;
}