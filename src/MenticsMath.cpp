#include "stdafx.h"

#include "MenticsMath.h"

namespace MenticsGame {

std::default_random_engine generator;
std::uniform_real_distribution<double> distribution(-1, 1);
std::function<double()> nextDouble = std::bind(distribution, generator);

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
	return vect3(nextDouble() * scale, nextDouble() * scale, nextDouble() * scale);
}

void testGrad(std::string name, int m, mfunc2 f, std::vector<double> at, double dx, double eps, void* data) {
	std::vector<double> empty(m*at.size());
	std::vector<double> around(at);
	std::vector<double> gradAt(m*at.size());
	std::vector<double> valueAt(m);
	f(m, valueAt.data(), 8, around.data(), gradAt.data(), data);
	for (int i = 0; i < at.size(); i++) {
		memcpy(around.data(), at.data(), sizeof(double) * 8);
		around[i] = around[i] - dx / 2.0;
		std::vector<double> valueBack(m);
		f(m, valueBack.data(), 8, around.data(), empty.data(), data);
		around[i] = around[i] + dx;
		std::vector<double> valueForward(m);
		f(m, valueForward.data(), 8, around.data(), empty.data(), data);
		std::vector<double> gradShouldBe(m);
		for (int j = 0; j < m; j++) {
			gradShouldBe[j] = (valueForward[j] - valueBack[j]) / dx;
			if (!isSimilar(gradAt[j * 8 + i], gradShouldBe[j], eps)) {
				printf("Compare failed for %s grad at [%d] was %g and calculated to be %g for constraint %d\n", name.c_str(), i, gradAt[j * 8 + i], gradShouldBe[j], j);
			}
		}
	}
}
void testGrad(std::string name, vfunc2 f, std::vector<double> &at, double dx, double eps, void* data) {
	std::vector<double> empty(8);
	std::vector<double> around(at);
	std::vector<double> gradAt(at.size());
	const double valueAt = f(around, gradAt, data);
	for (int i = 0; i < at.size(); i++) {
		memcpy(around.data(), at.data(), sizeof(double) * 8);
		around[i] = around[i] - dx / 2.0;
		const double valueBack = f(around, empty, data);
		around[i] = around[i] + dx;
		const double valueForward = f(around, empty, data);
		const double gradShouldBe = (valueForward - valueBack) / dx;
		if (!isSimilar(gradAt[i], gradShouldBe, eps)) {
			printf("Compare failed for %s grad at[%d] %g was %g and calculated to be %g\n", name.c_str(), i, at[i], gradAt[i], gradShouldBe);
		}
	}
}

}