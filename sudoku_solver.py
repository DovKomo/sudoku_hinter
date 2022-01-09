# https://www.askpython.com/python/examples/sudoku-solver-in-python
import numpy as np
import time

def print_sudoku(a):
    """Prints sudoku."""
    for i in range(9):
        for j in range(9):
            print(a[i][j], end=" ")
        print()


def solve(grid, row, col, num):
    """Solves sudoku, here 0 indicates that no value is assigned."""
    for x in range(9):
        if grid[row][x] == num:
            return False

    for x in range(9):
        if grid[x][col] == num:
            return False

    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


def sudoku(grid, row, col):
    """Handles all sudoku calculations."""
    if row == 9 - 1 and col == 9:
        return True
    if col == 9:
        row += 1
        col = 0
    if grid[row][col] > 0:
        return sudoku(grid, row, col + 1)

    for num in range(1, 9 + 1, 1):
        if solve(grid, row, col, num):
            grid[row][col] = num
            if sudoku(grid, row, col + 1):
                return True
        grid[row][col] = 0
    return False


if __name__ == "__main__":
    grid = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
                     [6, 0, 0, 1, 9, 5, 0, 0, 0],
                     [0, 9, 8, 0, 0, 0, 0, 6, 0],
                     [8, 0, 0, 0, 6, 0, 0, 0, 3],
                     [4, 0, 0, 8, 0, 3, 0, 0, 1],
                     [7, 0, 0, 0, 2, 0, 0, 0, 6],
                     [0, 6, 0, 0, 0, 0, 2, 8, 0],
                     [0, 0, 0, 4, 1, 9, 0, 0, 5],
                     [0, 0, 0, 0, 8, 0, 0, 7, 9]])
    start_time = time.time()
    if sudoku(grid, 0, 0):
        print_sudoku(grid)
    else:
        print('Solution does not exist')
    print("--- %s seconds ---" % (time.time() - start_time))
