from Matrix import BandMatrix, Vector
from math import sin
from MatMethods import Jacobi, GaussSeidel, LU_Factorization
import matplotlib.pyplot as plt

# Zadanie A
a1, a2, a3 = 10, -1, -1
N = 946
a = BandMatrix(N, a1, a2, a3)

b = Vector(N)
for i in range(b.rows):
    b.vals[i][0] = sin(4 * (i + 1))

# Zadanie B
print("\nZadanie B:")
norms_Jacobi, _ = Jacobi(a, b)
norms_GS, _ = GaussSeidel(a, b)

plt.semilogy(range(1, len(norms_Jacobi) + 1), norms_Jacobi, label="Jacobi")
plt.semilogy(range(1, len(norms_GS) + 1), norms_GS, label="Gauss-Seidel")
plt.title("Wykres wartości norm residuum dla metod iteracyjnych")
plt.xlabel("Iteracja")
plt.ylabel("Wartość normy")
plt.legend()
plt.show()

# Zadanie C
ca1, ca2, ca3 = 3, -1, -1
ca = BandMatrix(N, ca1, ca2, ca3)

print("\nZadanie C:")
norms_Jacobi, _ = Jacobi(ca, b)
norms_GS, _ = GaussSeidel(ca, b)

plt.semilogy(range(1, len(norms_Jacobi) + 1), norms_Jacobi, label="Jacobi")
plt.semilogy(range(1, len(norms_GS) + 1), norms_GS, label="Gauss-Seidel")
plt.title("Wykres wartości norm residuum dla metod iteracyjnych")
plt.xlabel("Iteracja")
plt.ylabel("Wartość normy")
plt.legend()
plt.show()

# Zadanie D
print("\nZadanie D:")
LU_Factorization(ca, b)

# Zadanie E
print("\nZadanie E:")
sizes = [100, 250, 500, 1000, 1500, 2000]
times_Jacobi = []
times_GS = []
times_LU = []
for size in sizes:
    print(f"\nRozmiar macierzy: {size}x{size}")
    ea = BandMatrix(size, a1, a2, a3)
    eb = Vector(size)
    for i in range(eb.rows):
        eb.vals[i][0] = sin(4 * (i + 1))
    _, time_Jacobi = Jacobi(ea, eb)
    _, time_GS = GaussSeidel(ea, eb)
    time_LU = LU_Factorization(ea, eb)
    times_Jacobi.append(time_Jacobi)
    times_GS.append(time_GS)
    times_LU.append(time_LU)
plt.plot(sizes, times_Jacobi, label="Jacobi")
plt.plot(sizes, times_GS, label="Gauss-Seidel")
plt.plot(sizes, times_LU, label="Faktoryzacja LU")
plt.title("Wykres zależności czasu wyznaczania rozw. od rozmiaru macierzy")
plt.xlabel("Rozmiar macierzy")
plt.ylabel("Czas [s]")
plt.legend()
plt.show()