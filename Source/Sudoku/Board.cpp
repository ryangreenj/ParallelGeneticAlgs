#include "Sudoku/Board.h"

#include <fstream>

using namespace std;

bool Board::LoadFromFile(string fileName)
{
    ifstream fileIn;
    fileIn.open(fileName);

    if (!fileIn.is_open())
    {
        return false;
    }

    string lineIn;

    // Skip first line, contains source information
    getline(fileIn, lineIn);

    // Second line contains sub-grid dimension n, where board is (n^2) x (n^2). n=3 for 9x9 board
    getline(fileIn, lineIn);
    int intIn = stoi(lineIn);

    if (intIn <= MAX_DIM)
    {
        dimension = intIn * intIn;
    }
    else
    {
        return false;
    }

    board = new byte[dimension*dimension];

    // Read remaining lines of the board
    for (int y = 0; y < dimension; ++y)
    {
        for (int x = 0; x < dimension; ++x)
        {
            if (getline(fileIn, lineIn, ',') && (intIn = stoi(lineIn)) <=  dimension)
            {
                board[y * dimension + x] = intIn;
            }
            else
            {
                return false;
            }
        }
    }

    return true;
}

void Board::PrintBoard(ostream &out)
{
    for (int y = 0; y < dimension; ++y)
    {
        for (int x = 0; x < dimension; ++x)
        {
            out << (int)board[y * dimension + x] << ' ';
        }
        out << '\n';
    }
    out << '\n';
}