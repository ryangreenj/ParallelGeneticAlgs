#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Sudoku/Kernel.cuh"
#include <iostream>

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

int* RankFitness(Population *popIn)
{
    int numChromosomes = 0;
    int numGenes = 0;

    // Arguments are output args, filled by function
    char *flattenedPop = popIn->FlattenPopulationToArray(numChromosomes, numGenes, false);

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
        std::cout << "(" << fitnessRank[i] << "\n";

    }
    return fitnessRank;
}


__global__ void OrderPopulation(int numChromosomes, int numGenes, char *flattenedPop, int *fitnessRank, int *gridswap, char *flattenedPopOut)
{
    
    int offset = ((blockIdx.x) * numGenes) + threadIdx.x;
    int rank_offset = (fitnessRank[blockIdx.x] * numGenes) + threadIdx.x;

    flattenedPopOut[rank_offset] = flattenedPop[offset];
} 

__global__ void BreedKernel(int numChromosomes, int numGenes, char *flattenedPop, int *fitnessRank, int *gridswap, char *flattenedPopOut)
{
    
    int swap = blockIdx.x % 2 ? -1 : 1;

    int grid = ((threadIdx.x % 9) / 3) + ((threadIdx.x / 27) * 3);
    int offset = ((blockIdx.x) * numGenes) + threadIdx.x;

    if (grid == gridswap[blockIdx.x])
    {
        int offset_new = ((blockIdx.x + swap) * numGenes) + threadIdx.x;
        flattenedPopOut[offset] = flattenedPop[offset_new];
    }
    else 
    {
        flattenedPopOut[offset] = flattenedPop[offset];
    }
}

Population* Breed(Population *popIn, int* rankings)
{
    double choices[2] = {0.20, 0.8}; // Keep top 20%, breed next 60%, drop final 20%

    int numChromosomes = 0;
    int numGenes = 0;
    std::shared_ptr<bool[]> lockedGenesIn = popIn->GetLockedGenes();

    // Arguments are output args, filled by function
    // Need to delete returned pointer at end
    char *flattenedPop = popIn->FlattenPopulationToArray(numChromosomes, numGenes);

    char *dev_flattenedPopBreed;
    char *dev_flattenedPopBreedOut;
    int *dev_fitnessRankBreed;
    int *dev_gridSwap;
    int *grid_swap = new int[numChromosomes];

    for (int i = 0; i < numChromosomes / 2; i++){
        int swap = rand() % 9;
        grid_swap[i * 2] = swap;
        grid_swap[(i * 2) + 1] = swap;

    }

    cudaMalloc((void **)&dev_flattenedPopBreed, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_flattenedPopBreedOut, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_fitnessRankBreed, numChromosomes * sizeof(int));
    cudaMalloc((void **)&dev_gridSwap, numChromosomes * sizeof(int));

    cudaMemcpy(dev_flattenedPopBreed, flattenedPop, numChromosomes * numGenes * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_fitnessRankBreed, rankings, numChromosomes * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gridSwap, grid_swap, numChromosomes * sizeof(int), cudaMemcpyHostToDevice);

    OrderPopulation<<<numChromosomes, numGenes>>>(numChromosomes, numGenes, dev_flattenedPopBreed, dev_fitnessRankBreed, dev_gridSwap, dev_flattenedPopBreedOut);
    // BreedKernel<<<numChromosomes, numGenes>>>(numChromosomes, numGenes, dev_flattenedPopBreed, dev_fitnessRankBreed, dev_gridSwap, dev_flattenedPopBreedOut);
    
    char *popout = new char[numChromosomes * numGenes];
    cudaMemcpy(popout, dev_flattenedPopBreedOut, numChromosomes * numGenes * sizeof(char), cudaMemcpyDeviceToHost);
    
    Population *out = new Population(numGenes, numChromosomes, lockedGenesIn, popout);
    // Population *out = new Population(numGenes, numChromosomes, lockedGenesIn, pop);
    return out;
}