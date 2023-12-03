
#define EIGEN_USE_MKL_ALL
#define EIGEN_VECTORIZE_SSE4_2

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/PardisoSupport>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace Eigen;
using namespace std;
using data_type = double;

void from_file(std::vector<Eigen::Triplet<data_type>> & target, const std::string& path){
    std::ifstream file(path, std::ios::binary);
    Eigen::Triplet<data_type> triplet;
    while(file.read(reinterpret_cast<char*>(&triplet), sizeof(triplet))){
        target.emplace_back(triplet);
    }
    std::cout << "read " << target.size() << " triplets" << std::endl;
}

void data_from_file(std::vector<data_type> & target, const std::string& path){
    std::ifstream file(path, std::ios::binary);
    data_type line;

    while (file.read(reinterpret_cast<char*>(&line), sizeof(line)))
    {
        target.emplace_back(line);
    }
    std::cout << "read " << target.size() << " data" << std::endl;
}

bool test_solve() {
    auto rhs = std::vector<data_type>();
    auto lhs = std::vector<data_type>();
    auto triplet = std::vector<Eigen::Triplet<data_type>>();
    auto triplet_m = std::vector<Eigen::Triplet<data_type>>();
    triplet.reserve(7000000);
    rhs.reserve(120000);
    lhs.reserve(120000);
    triplet_m.reserve(120000);

    std::string prefix = DATA_PATH;
    from_file(triplet, prefix + "N.matrix");
    from_file(triplet_m, prefix + "mass.matrix");
    data_from_file(rhs, prefix + "rhs.data");
    data_from_file(lhs, prefix + "lhs.data");

    std::cout << "read data done" << std::endl;
    std::cout << "dof: " << rhs.size() << std::endl;
    std::cout << "sparse rate: " << 100.0 - (static_cast<data_type>(triplet.size()) * 100.0/ static_cast<data_type>((rhs.size() * rhs.size()))) << "%" << std::endl;
    Eigen::VectorXd rhs_eigen(rhs.size());
    Eigen::VectorXd lhs_eigen(lhs.size());
    std::cout << std::endl;

    for(int i = 0; i < rhs.size(); i++){
        rhs_eigen(i) = rhs[i];
        lhs_eigen(i) = lhs[i];
    }

    auto dof = rhs.size();
    auto N = SparseMatrix<double>(dof,dof);
    auto M = SparseMatrix<double>(dof,dof);

    N.setFromTriplets(triplet.begin(), triplet.end());
    M.setFromTriplets(triplet_m.begin(), triplet_m.end());
    auto ieSystem = N + 10000.0 * M;

    SimplicialLLT<SparseMatrix<double>> solver;
    PardisoLLT<SparseMatrix<double>> pardisoSolver;

    auto t0 = std::chrono::steady_clock::now();

    solver.analyzePattern(ieSystem);
    solver.factorize(ieSystem);
    std::cout << "SimplicialLLT factorize time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count())/ 1000.0 << " ms " << std::endl;
    t0 = std::chrono::steady_clock::now();
    VectorXd c = solver.solve(rhs_eigen);
    std::cout << "SimplicialLLT solve time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds >(std::chrono::steady_clock::now() - t0).count())/ 1000.0  << " ms " << std::endl;
    auto err_s = ieSystem * c - rhs_eigen;
    auto err_simplicial = err_s.norm();
    std::cout << "SimplicialLLT error: " << err_simplicial << std::endl;
    std::cout << std::endl;

    t0 = std::chrono::steady_clock::now();
    pardisoSolver.analyzePattern(ieSystem);
    pardisoSolver.factorize(ieSystem);
    std::cout << "PardisoLLT factorize time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count())/ 1000.0  << " ms " << std::endl;

    t0 = std::chrono::steady_clock::now();
    VectorXd c2 = pardisoSolver.solve(rhs_eigen);
    std::cout << "PardisoLLT solve time: " << static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count())/ 1000.0
    << " ms " << std::endl;

    auto err_pardiso = (ieSystem * c2 - rhs_eigen).norm();
    std::cout << "PardisoLLT error: " << err_pardiso << std::endl;
    return true;

}

int main(){
    Eigen::initParallel();
    auto res = test_solve();
    return 0;
}