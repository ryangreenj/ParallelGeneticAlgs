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

int* RankFitness(int numChromosomes, int numGenes, char *flattenedPop, int *errorCountsOut)
{
    int dimension = sqrt(numGenes);

    char *dev_flattenedPop;
    int *dev_fitnessCount;

    cudaMalloc((void **)&dev_flattenedPop, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_fitnessCount, numChromosomes * sizeof(int));

    cudaMemcpy(dev_flattenedPop, flattenedPop, numChromosomes * numGenes * sizeof(char), cudaMemcpyHostToDevice);

    RankFitnessKernel<<<numChromosomes, numGenes>>>(numChromosomes, dimension, dev_flattenedPop, dev_fitnessCount);

    cudaMemcpy(errorCountsOut, dev_fitnessCount, numChromosomes * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_flattenedPop);
    cudaFree(dev_fitnessCount);

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

__global__ void BreedKernel(int numChromosomes, int numGenes, int dimension, int subDim, int seed, char *flattenedPop, int *ranks, bool *lockedIn, int* dev_swap_index, int* dev_swap_candidates, char *dev_tempPopualtion)
{   

    curandState_t state;

    // Select top ranked solutions and place them in dev_tempPopualtion
    if (ranks[blockIdx.x] < (int)(numChromosomes * RANK_RETENTION_RATE))
    {
        int offset = (blockIdx.x * numGenes) + threadIdx.x;
        int rank = (ranks[blockIdx.x] * numGenes) + threadIdx.x;
        dev_tempPopualtion[rank] = flattenedPop[offset];
    }

    __syncthreads();
    // This block will pick the chromosomes that will be swaping, via a tournament style selection
    // The two solutions with the best rank will swap genes

    // This could easily be made to use an array and feature more than 4 prospective chomosomes

    if (threadIdx.x == 0) 
    {   
        /* we have to initialize the state */
        curand_init(seed, /* the seed controls the sequence of random values that are produced */
                blockIdx.x, /* the sequence number is only important with multiple cores */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);

        // Select 4 random indexes that within the RANK_RETENTION_RATE
        int c1 = (curand(&state) % (int)(numChromosomes * RANK_RETENTION_RATE));
        int c2 = (curand(&state) % (int)(numChromosomes * RANK_RETENTION_RATE));
        int c3 = (curand(&state) % (int)(numChromosomes * RANK_RETENTION_RATE));
        int c4 = (curand(&state) % (int)(numChromosomes * RANK_RETENTION_RATE));

        // best ranks between c1/c2 and c3/c4 will swap
        dev_swap_candidates[(blockIdx.x * 2)] = c1 < c2 ? c1 : c2;
        dev_swap_candidates[(blockIdx.x * 2) + 1] = c3 < c4 ? c3 : c4;

        // decide which subgrid we will start the swap at
        // this number is in the range [2,dimension-1], don't start at first or last subgrid
        dev_swap_index[blockIdx.x] = (blockIdx.x % (dimension - 2)) + 1;
    }

    __syncthreads();

    int offset = ((blockIdx.x) * numGenes) + threadIdx.x; // location of gene in flattened population
    int grid = GET_SUB_GRID(threadIdx.x, subDim); // determines the grid of current thread within a block
    
    // Get new offset by picking the correct swap candidate
    // if the current grid is less than the swap index, its the first candidate, otherwise its the second
    int offset_new = (dev_swap_candidates[(blockIdx.x * 2) + (grid < dev_swap_index[blockIdx.x] ? 0 : 1)] * numGenes) + threadIdx.x;
    flattenedPop[offset] = dev_tempPopualtion[offset_new];
    
    __syncthreads();

    // This carries out a randome number of 'mutations' 
    // a mutation is just swapping to non-locked genes of a subgrid
    if (threadIdx.x == 0)
    {
        // 1 to 3 mutations will happen on every block (this was randomly decided)
        // I was messing with this a lot and this seems to be best so far
        for (int k = 0; k < (curand(&state) % subDim) + 1; k++)
        {
            // randomly generates a swap index and makes sure that it is not locked gene
            int swap_index_1 = (curand(&state) % numGenes);
            while (lockedIn[swap_index_1]) swap_index_1 = (curand(&state) % numGenes);

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
                subGridTile = curand(&state) % dimension;
                swap_index_2 = (rowOffset + (subGridTile / subDim)) * dimension + colOffset + (subGridTile % subDim);
            } while (swap_index_1 == swap_index_2 || lockedIn[swap_index_2]);

            // swaps swap_index_1 and swap_index_2
            char temp = flattenedPop[(blockIdx.x * numGenes) + swap_index_1];
            flattenedPop[(blockIdx.x * numGenes) + swap_index_1] = flattenedPop[(blockIdx.x * numGenes) + swap_index_2];
            flattenedPop[(blockIdx.x * numGenes) + swap_index_2] = temp;
        }
    }
}

Population* Breed(Population *popIn, int &bestrank, char* bestboard)
{
    int numChromosomes = 0;
    int numGenes = 0;
    std::shared_ptr<bool[]> lockedGenesIn = popIn->GetLockedGenes();

    char *flattenedPop = popIn->FlattenPopulationToArray(numChromosomes, numGenes, false);

    int dimension = sqrt(numGenes);
    int subDim = sqrt(dimension);

    // START FITNESS RANK
    int *errorCounts = new int[numChromosomes];

    char *dev_flattenedPop;
    int *dev_fitnessCount;

    cudaMalloc((void **)&dev_flattenedPop, numChromosomes * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_fitnessCount, numChromosomes * sizeof(int));

    cudaMemcpy(dev_flattenedPop, flattenedPop, numChromosomes * numGenes * sizeof(char), cudaMemcpyHostToDevice);

    auto startFitness = std::chrono::high_resolution_clock::now();
    RankFitnessKernel<<<numChromosomes, numGenes >>>(numChromosomes, dimension, dev_flattenedPop, dev_fitnessCount);
    auto stopFitness = std::chrono::high_resolution_clock::now();

    cudaMemcpy(errorCounts, dev_fitnessCount, numChromosomes * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_fitnessCount);

    // Give chromosomes a unique rank based on their errorCount/fitnessScore

    // Using std::sort is O(n log n), previous algorithm was O(n^2)
    // <fitnessScore, chromosomeIndex>
    std::vector<std::pair<int, int>> scoreToIndex = std::vector<std::pair<int, int>>();

    for (int i = 0; i < numChromosomes; ++i)
    {
        scoreToIndex.push_back(std::make_pair(errorCounts[i], i));
    }

    // Sorts by fitnessScore, O(log n)
    std::sort(scoreToIndex.begin(), scoreToIndex.end());

    int *fitnessRanks = new int[numChromosomes];

    int i = 0;
    for (auto &pair : scoreToIndex)
    {
        fitnessRanks[pair.second] = i++;
    }

    // END FITNESS RANK

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
    
    char *dev_tempPopualtion;
    int *dev_ranks;
    bool *dev_lockedIn;
    int *dev_swap_index;
    int *dev_swap_candidates;

    int retention_size = (int)(numChromosomes * RANK_RETENTION_RATE);

    cudaMalloc((void **)&dev_tempPopualtion, retention_size * numGenes * sizeof(char));
    cudaMalloc((void **)&dev_ranks, numChromosomes * sizeof(int));
    cudaMalloc((void **)&dev_lockedIn, numChromosomes * sizeof(bool));
    cudaMalloc((void **)&dev_swap_index, numChromosomes * sizeof(int));
    cudaMalloc((void **)&dev_swap_candidates, numChromosomes * 2 * sizeof(int));

    cudaMemcpy(dev_ranks, fitnessRanks, numChromosomes * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemcpy(dev_lockedIn, lockedGenesIn.get(), numChromosomes * sizeof(bool), cudaMemcpyHostToDevice);

    auto startBreed = std::chrono::high_resolution_clock::now();
    BreedKernel<<<numChromosomes, numGenes>>>(numChromosomes, numGenes, dimension, subDim, (rand() % dimension) + 1, dev_flattenedPop, dev_ranks, dev_lockedIn, dev_swap_index, dev_swap_candidates, dev_tempPopualtion);
    auto stopBreed = std::chrono::high_resolution_clock::now();

    std::cout << "Parallel Time: " << ((std::chrono::duration_cast<std::chrono::microseconds>(stopFitness - startFitness)).count() + (std::chrono::duration_cast<std::chrono::microseconds>(stopBreed - startBreed)).count()) << "\n";

    char *popout = new char[numChromosomes * numGenes];
    cudaMemcpy(popout, dev_flattenedPop, numChromosomes * numGenes * sizeof(char), cudaMemcpyDeviceToHost);
    
    cudaFree(dev_flattenedPop);
    cudaFree(dev_ranks);
    cudaFree(dev_lockedIn);
    cudaFree(dev_tempPopualtion);
    cudaFree(dev_swap_index);
    cudaFree(dev_swap_candidates);

    delete[] errorCounts;
    delete[] fitnessRanks;

    Population *out = new Population(numGenes, numChromosomes, lockedGenesIn, popout);

    return out;
}