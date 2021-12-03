#include <chrono>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "Utility.h"
#include "Sudoku/Board.h"
#include "Sudoku/Kernel.cuh"
#include "Sudoku/Population.h"
#include "Sudoku/Sequential.h"

using namespace std;

int SudokuDriver(string fileIn);

int main(int argc, char **argv)
{
    std::srand(std::time(nullptr)); // Seed random generator with current time
    return SudokuDriver("Data/SudokuBoard9x9_2.txt");
}

// Good references: 
// http://micsymposium.org/mics_2009_proceedings/mics2009_submission_66.pdf
// https://www.math.uci.edu/~brusso/DengLiOptLett2013.pdf

int SudokuDriver(string fileIn)
{
    bool success = false;
    Board *board = new Board(fileIn, success);
    int dimension = board->GetDimension();

    if (!success)
    {
        delete board;
        return 1;
    }

    // Can parallelize initial step of determining any tiles from start board
    Board *out = PredetermineTiles(board);
    cout << "Initial Board\n";
    out->PrintBoard(cout);
    
    Population *pop = new Population(out, 250);
    delete out;
  
    int bestrank = 0;
    char* best_board = new char[pop->GetNumGenes()];

    for (int i = 0; i < NUM_GENERATIONS; i++){
        
#if USE_PARALLEL
        pop = Breed(pop, bestrank, best_board);
#else
        pop = Sequential::Breed(pop, bestrank, best_board);
#endif
        if (bestrank < 5) break;
    }

    for (int i = 0; i < dimension; i++)
    {
        for (int j = 0; j < dimension; j++)
        {
            std::cout << (int)best_board[(i * dimension) + j] << " ";
        }
        std::cout << "\n";
    }
    
    delete best_board;
    delete board;
    delete pop;

    return 0;
}