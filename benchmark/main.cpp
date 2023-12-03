
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/PardisoSupport>
#include <chrono>
#include <iostream>

using namespace Eigen;
using namespace std;


bool test_solve() {
    auto a = SparseMatrix<double>(300000,300000);
    auto b = VectorXd::Random(300000);
    std::vector<Eigen::Triplet<double>> triplets;
    for (int i = 0; i < 300000; i++) {
        if (i % 101 == 0) {
            for (int j = 0; j < 300000; j++) {
                if (j % 101 == 0)
                    triplets.emplace_back(i, j, rand() * 10.0);
            }
        }
        triplets.emplace_back(i, i, rand() * 100000.0);
    }
    a.setFromTriplets(triplets.begin(), triplets.end());
   
    SimplicialLLT<SparseMatrix<double>> solver;
    auto t0 = std::chrono::steady_clock::now();
    solver.analyzePattern(a);

    solver.factorize(a);

    VectorXd c = solver.solve(b);
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count() << "microseconds" <<std::endl;

    return true;
}

int main(){
    Eigen::initParallel();
    auto res = test_solve();
    return 0;
}