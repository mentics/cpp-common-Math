#pragma once





#ifndef EIGEN_FORWARDDECLARATIONS_H
namespace Eigen {

#define EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Size, SizeSuffix)   \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Size> Matrix##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, 1>    Vector##SizeSuffix##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, 1, Size>    RowVector##SizeSuffix##TypeSuffix;

#define EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, Size)         \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Size, Dynamic> Matrix##Size##X##TypeSuffix;  \
/** \ingroup matrixtypedefs */                                    \
typedef Matrix<Type, Dynamic, Size> Matrix##X##Size##TypeSuffix;

#define EIGEN_MAKE_TYPEDEFS_ALL_SIZES(Type, TypeSuffix) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 2, 2) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 3, 3) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, 4, 4) \
EIGEN_MAKE_TYPEDEFS(Type, TypeSuffix, Dynamic, X) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 2) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 3) \
EIGEN_MAKE_FIXED_TYPEDEFS(Type, TypeSuffix, 4)

	EIGEN_MAKE_TYPEDEFS_ALL_SIZES(int, i)
		EIGEN_MAKE_TYPEDEFS_ALL_SIZES(float, f)
		EIGEN_MAKE_TYPEDEFS_ALL_SIZES(double, d)
	template<typename _Scalar, int _Rows, int _Cols,
		int _Options = 0,
		int _MaxRows = _Rows,
		int _MaxCols = _Cols
	> class Matrix;
	class Vector2d;
	class Vector3d;
	class Vector4d;

}
#endif

namespace MenticsGame {

extern std::function<double()> nextDouble;

template <int SIZE>
using vector = Eigen::Matrix<double, SIZE, 1>;
using vect3 = Eigen::Vector3d;
using vect4 = Eigen::Vector4d;
using vect8 = vector<8>;
using mat3x8 = Eigen::Matrix<double, 3, 8>;

typedef double(*vfunc2)(const std::vector<double> &x, std::vector<double> &grad, void *data);
typedef void(*mfunc2)(unsigned m, double *result, unsigned n, const double* x, double* grad, void* f_data);

typedef double(*calcError)(const std::vector<double> &x, void *data);


bool isSimilar(double x1, double x2, double eps);
vect3 randomVector(double scale);

void testGrad(std::string name, int m, mfunc2 f, std::vector<double> at, double dx, double eps, void* data);
void testGrad(std::string name, vfunc2 f, std::vector<double> &at, double dx, double eps, void* data);

}