#pragma once



#include "Eigen/Core"     
#include "MenticsCommon.h"

namespace MenticsGame {

extern std::function<double()> nextDouble;

template <int SIZE>
using Evector = Eigen::Matrix<double, SIZE, 1>;
using vect3 = Eigen::Vector3d;
using vect4 = Eigen::Vector4d;
using vect8 = Evector<8>;
using mat3x8 = Eigen::Matrix<double, 3, 8>;

typedef double(*vfunc2)(const std::vector<double> &x, std::vector<double> &grad, void *data);
typedef void(*mfunc2)(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data);

typedef double(*calcError)(const std::vector<double> &x, void *data);


inline bool isSimilar(double x1, double x2, double eps) {
	const double sum = x1 + x2;
	if (sum < eps) {
		return true;
	}
	else {
		return abs(x1 - x2) / sum < eps;
	}
}

inline vect3 randomVector(double scale) {
	vect3 v(vect3(static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / scale)), static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / scale)), static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / scale))));
		
		return v;
}

void testGrad(std::string name, int m, mfunc2 f, std::vector<double> at, double dx, double eps, void* data);
void testGrad(std::string name, vfunc2 f, std::vector<double> &at, double dx, double eps, void* data);


}