#ifndef SUDOKU_BOARD_H
#define SUDOKU_BOARD_H

#include <ostream>
#include <string>

#include "Utility.h"

class Board
{
public:
    Board() {}
    Board(int dimensionIn, char *boardIn) { dimension = dimensionIn; board = boardIn; }
    Board(std::string fileName, bool &success) { success = LoadFromFile(fileName); }
    ~Board() { if (board) { delete []board; board = nullptr; } }

    int GetDimension() { return dimension; }
    char *GetBoardPointer() { return board; }

    bool LoadFromFile(std::string fileName);
    void PrintBoard(std::ostream &out);

private:
    int dimension = 0;
    char *board = nullptr; // dimension*dimension length
};

#endif
