#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>

#include "Utility.h"
#include "Sudoku/Board.h"
#include "Sudoku/Kernel.cuh"
#include "Sudoku/Population.h"

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

    if (!success)
    {
        delete board;
        return 1;
    }

    // Can parallelize initial step of determining any tiles from start board
    Board *out = PredetermineTiles(board);
    cout << "Initial Board\n";
    out->PrintBoard(cout);
    
    Population *pop = new Population(out, 1000);
    delete out;

    pop->PrintPopulation(cout);

    // Fitness ranking is parallelized
    // Fitness score is number of errors : Lower score == better board
    // Population *pop_2 = nullptr;
    // Population *pop_2 = Breed(pop);
    // delete pop;

    for (int i = 0; i < 100; i++){
        pop = Breed(pop);
    }
    
    // pop->PrintPopulation(cout);

    // cout << "------------------\n";
    

    delete board;
    delete pop;

    return 0;
}