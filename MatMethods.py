from Matrix import Vector, SpEye
from MatCalc import matmul
from copy import deepcopy
from time import time


def Jacobi(a, b):
    if a.cols != b.rows and b.cols != 1:
        raise ValueError
    x = Vector(a.cols)
    for i in range(x.rows):
        x.vals[i][0] = 0
    iters = 0
    norm = 0
    norms = []
    startTime = time()
    while iters < 100:
        iters += 1
        newX = Vector(a.cols)
        for i in range(newX.rows):
            val = b.vals[i][0]
            for j in range(newX.rows):
                if i != j:
                    val -= a.vals[i][j] * x.vals[j][0]
            val /= a.vals[i][i]
            newX.vals[i][0] = val
        x = newX
        res = matmul(a, x) - b
        norm = res.norm()
        norms.append(norm)
        if norm < 10e-9:
            break
    endTime = time() - startTime
    print(f"Jacobi: {iters} iteracji")
    print(f"Czas: {endTime} s")
    print(f"Błąd: {norm}")
    return norms, endTime


def GaussSeidel(a, b):
    if a.cols != b.rows and b.cols != 1:
        raise ValueError
    x = Vector(a.cols)
    for i in range(x.rows):
        x.vals[i][0] = 0
    iters = 0
    norm = 0
    norms = []
    startTime = time()
    while iters < 100:
        iters += 1
        newX = Vector(a.cols)
        for i in range(newX.rows):
            val = b.vals[i][0]
            for j in range(i):
                val -= a.vals[i][j] * newX.vals[j][0]
            for j in range(i + 1, newX.rows):
                val -= a.vals[i][j] * x.vals[j][0]
            val /= a.vals[i][i]
            newX.vals[i][0] = val
        x = newX
        res = matmul(a, x) - b
        norm = res.norm()
        norms.append(norm)
        if norm < 10e-9:
            break
    endTime = time() - startTime
    print(f"Gauss-Seidel: {iters} iteracji")
    print(f"Czas: {endTime} s")
    print(f"Błąd: {norm}")
    return norms, endTime


def LU_Factorization(a, b):
    startTime = time()

    U = deepcopy(a)
    L = SpEye(a.cols)

    for i in range(1, a.cols):
        for j in range(i):
            L.vals[i][j] = U.vals[i][j] / U.vals[j][j]
            for k in range(U.cols):
                U.vals[i][k] -= L.vals[i][j] * U.vals[j][k]

    y = Vector(a.cols)
    x = Vector(a.cols)

    for i in range(y.rows):
        val = b.vals[i][0]
        for j in range(i):
            val -= L.vals[i][j] * y.vals[j][0]
        val /= L.vals[i][i]
        y.vals[i][0] = val

    for i in range(x.rows - 1, -1, -1):
        val = y.vals[i][0]
        for j in range(i + 1, x.rows):
            val -= U.vals[i][j] * x.vals[j][0]
        val /= U.vals[i][i]
        x.vals[i][0] = val

    res = matmul(a, x) - b
    norm = res.norm()
    endTime = time() - startTime
    print("Faktoryzacja LU:")
    print(f"Czas: {endTime} s")
    print(f"Błąd: {norm}")

    return endTime