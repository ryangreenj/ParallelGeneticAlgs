#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Sudoku/Kernel.cuh"

__global__ void PredetermineTilesKernel(int dimension, byte *boardIn, byte *boardOut)
{
    int tileId = threadIdx.x;
    int row = tileId / dimension;
    int col = tileId % dimension;

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
                byte tile = boardIn[iVal * dimension + col];
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

                // TODO subgrid logic
            }

            byte candidate = 0;
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
    byte *boardArrIn = boardIn->GetBoardPointer();

    byte *dev_boardIn, *dev_boardOut;

    cudaMalloc((void **)&dev_boardIn, dimension * dimension * sizeof(byte));
    cudaMalloc((void **)&dev_boardOut, dimension * dimension * sizeof(byte));

    cudaMemcpy(dev_boardIn, boardArrIn, dimension * dimension * sizeof(byte), cudaMemcpyHostToDevice);
    
    PredetermineTilesKernel<<<1, dimension * dimension>>>(dimension, dev_boardIn, dev_boardOut);

    byte *boardArrOut = new byte[dimension * dimension];
    cudaMemcpy(boardArrOut, dev_boardOut, dimension * dimension * sizeof(byte), cudaMemcpyDeviceToHost);

    cudaFree(dev_boardIn);
    cudaFree(dev_boardOut);

    Board *out = new Board(dimension, boardArrOut);
    return out;
}



__global__ void RankFitnessKernel(int numChromosomes, int numGenes, byte *flattenedPop, int *fitnessRankOut)
{
    //TODO
}

int* RankFitness(Population *popIn)
{
    int numChromosomes = 0;
    int numGenes = 0;

    // Arguments are output args, filled by function
    // Need to delete returned pointer at end
    byte *flattenedPop = popIn->FlattenPopulationToArray(numChromosomes, numGenes);

    byte *dev_flattenedPop;
    int *dev_fitnessRank;

    cudaMalloc((void **)&dev_flattenedPop, numChromosomes * numGenes * sizeof(byte));
    cudaMalloc((void **)&dev_fitnessRank, numChromosomes * sizeof(int));

    cudaMemcpy(dev_flattenedPop, flattenedPop, numChromosomes * numGenes * sizeof(byte), cudaMemcpyHostToDevice);

    RankFitnessKernel<<<numChromosomes, numGenes>>>(numChromosomes, numGenes, dev_flattenedPop, dev_fitnessRank);

    int *fitnessRank = new int[numChromosomes];
    cudaMemcpy(fitnessRank, dev_fitnessRank, numChromosomes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_flattenedPop);
    cudaFree(dev_fitnessRank);

    return fitnessRank;
}