#include "stdafx-mathtest.h"
#include "CppUnitTest.h"

#include "MenticsCommon.h"
#include "MenticsCommonTest.h"
#include "MenticsMath.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

const int NUMPARAMS = 2;
const double CLOSE_ENOUGH = 1e-3;

namespace MenticsGame {
namespace MathTest2 {
int funcCalls = 0;
double bestOptVal = 100000000;
std::vector<double> bestX(NUMPARAMS);
const double k = 8;
const double m = 0.5;
const double TIMES = 10;

struct ProblemData {
    double p0;
    double v0;
    double t0;
    double t1;
    int times;
};

double diffEqError(unsigned n, const double* x, double* grad, void* data);
double checkError(double const* x, ProblemData& data);

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

TEST_CLASS(UnitTest2) {

    const std::string name = "PhysicsTest";

    TEST_CLASS_INITIALIZE(BeforeClass) {
        setupLog();
    }

    double solve(std::vector<double>& x, ProblemData& data) {
        nlopt::opt opt(nlopt::LD_SLSQP, NUMPARAMS);
        opt.set_min_objective(diffEqError, (void*)(&data));
        std::vector<double> lowerBound(NUMPARAMS);
        std::vector<double> upperBound(NUMPARAMS);
        for (int i = 0; i < NUMPARAMS; i++) {
            lowerBound[i] = -10;
            upperBound[i] = 10;
        }
        opt.set_lower_bounds(lowerBound);
        opt.set_upper_bounds(upperBound);

        opt.set_xtol_rel(1e-3);
        opt.set_maxeval(100);

        int iterations = 0;
        bool found = false;
        while (!found && iterations <= 100) {
            iterations++;
            for (int j = 0; j < NUMPARAMS; j++) {
                x[j] = MenticsGame::nextDouble();
            }

            double minf;
            try {
                const nlopt::result result = opt.optimize(x, minf);
                double optVal = opt.last_optimum_value();
                if (optVal < bestOptVal) {
                    bestOptVal = optVal;
                    bestX = x;
                }
                mlog->debug("iteration {0} {1} {3} \n", iterations, " funcCalls=", funcCalls);
                if (result < 0) {
                    mlog->warn("nlopt failed!");
                } else {
                    const double error = checkError(x.data(), data);
                    if (error > CLOSE_ENOUGH) {
                        mlog->warn("Checked error failure: {0} \n", result);
                        continue;
                    } else {
                        //printf("Found minimum at %g,%g after calls func: %d constraints: %d\n", x[6], x[7], funcCalls, constraintCalls);
                        // Success, stop looping
                        found = true;
                    }
                }
            } catch (const std::exception& e) {
                mlog->debug("iteration {0} {1} {2} {3} {5}) \n", iterations, " funcCalls=", funcCalls, " (nlopt exception: ", e.what());
                const double error = checkError(x.data(), data);
                if (error > CLOSE_ENOUGH) {
                } else {
                    mlog->warn("Exception but constraints satisfied\n");
                    found = true;
                }
                continue;
            }
        }
        return found ? iterations : -1;
    }

    struct CandidateFunction {
        std::vector<double> coeffs;

        CandidateFunction(std::vector<double> const& coeffs) : coeffs(coeffs) {}

        double at(double t) const {
            double degreeT = 1;
            double sum = 0;
            for (int i = 0; i < coeffs.size(); i++) {
                sum += coeffs[i] * degreeT;
                degreeT *= t;
            }
            return sum;
        }

        double dat(double t) const {
            double degreeT = 1;
            double mult = 1;
            double sum = 0;
            for (int i = 1; i < coeffs.size(); i++) {
                sum += mult * coeffs[i] * degreeT;
                degreeT *= t;
                mult++;
            }
            return sum;
        }

        double ddat(double t) const {
            double degreeT = 1;
            double mult = 2;
            double sum = 0;
            for (int i = 2; i < coeffs.size(); i++) {
                sum += mult * coeffs[i] * degreeT;
                degreeT *= t;
                mult *= mult + 1;
            }
            return sum;
        }
    };

public:

    TEST_METHOD(DiffEqError2) {
        ProblemData data{static_cast<int>(1.0), static_cast<int>(0.0), static_cast<int>(0.0), static_cast<int>(1.0), static_cast<int>(TIMES)};
        std::vector<double> x(NUMPARAMS);

        bool found = solve(x, data);
        x = bestX;

        std::vector<double> res{1.0, 0.0, x[0], x[1]};
        CandidateFunction result(res);
        if (bestOptVal < 100) {
            mlog->debug(bestOptVal);
        }
        Assert::IsTrue(result.ddat(0) < 0);
    }
};

double at(ProblemData* data, double const* x, double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    return data->p0 + data->v0 * t + x[0] * t2 + x[1] * t3;
}

double dat(ProblemData* data, double const* x, double t) {
    double t2 = t * t;
    return data->v0 + 2.0*x[0] * t + 3.0*x[1] * t2;
}

double ddat(ProblemData* data, double const* x, double t) {
    return 2.0*x[0] + 6.0*x[1] * t;
}

void gradP(Eigen::Vector2d& grad, double t) {
    double t2 = t * t;
    double t3 = t2 * t;
    grad[0] = t2;
    grad[1] = t3;
}

void gradPpp(Eigen::Vector2d& grad, double t) {
    grad[0] = 1;
    grad[1] = t;
}

double diffEqError(unsigned n, const double* x, double* grad, void* vdata) {
    funcCalls++;
    ProblemData* data = (ProblemData*)vdata;

    std::vector<double> times(data->times);
    times[0] = data->t0;
    times[data->times - 1] = data->t1;
    double interval = (data->t1 - data->t0) / (double)data->times;
    for (int i = 0; i < data->times - 2; i++) {
        times[i + 1] = data->t0 + interval * (i + 1);
    }
    std::vector<double> errors(data->times);
    std::vector<int> signs(data->times);

    double sumError = 0;
    for (int i = 0; i < data->times; i++) {
        double t = times[i];
        double atxt = at(data, x, t);
        double ddatxt = ddat(data, x, t);
        double inner = -k * atxt - (m * ddatxt);
        errors[i] = abs(inner);
        signs[i] = sgn(inner);
        sumError += errors[i];
    }

    if (grad != NULL) {
        Eigen::Vector2d fullGrad{0.0, 0.0};
        Eigen::Vector2d tmpGradP;
        Eigen::Vector2d tmpGradPpp;
        for (int i = 0; i < data->times; i++) {
            gradP(tmpGradP, times[i]);
            gradPpp(tmpGradPpp, times[i]);
            fullGrad += signs[i] * (-k * tmpGradP - m * tmpGradPpp);
        }

        grad[0] = fullGrad[0];
        grad[1] = fullGrad[1];
    }

    return sumError;
}

double checkError(double const* x, ProblemData& dd) {
    ProblemData* data = &dd;
    std::vector<double> times(data->times);
    times[0] = data->t0;
    times[data->times - 1] = data->t1;
    double interval = (data->t1 - data->t0) / (double)data->times;
    for (int i = 0; i < data->times - 2; i++) {
        times[i] = data->t0 + interval * (i + 1);
    }
    std::vector<double> errors(data->times);
    std::vector<int> signs(data->times);

    double sumError = 0;
    for (int i = 0; i < data->times; i++) {
        double t = times[i];
        double atxt = at(data, x, t);
        double ddatxt = ddat(data, x, t);
        double inner = -k * atxt - (m * ddatxt);
        errors[i] = abs(inner);
        signs[i] = sgn(inner);
        sumError += errors[i];
    }

    return sumError;
}
}
}
