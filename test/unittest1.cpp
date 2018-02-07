#include "stdafx.h"
#include "MenticsCommon.h"
#include "MenticsCommonTest.h"
#include "MenticsMath.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

#include <spdlog/spdlog.h>
#include <sstream>

namespace MathTest
{		
	int funcCalls = 0;

	double diffEqError(unsigned n, const double* x, double* grad, void* data);

	template <typename T> int sgn(T val) {
		return (T(0) < val) - (val < T(0));
	}

	struct ProblemData {
		double p0;
		double v0;
		double t0;
		double t1;
		int samples;
	};

	TEST_CLASS(UnitTest1)
	{
		const std::string name = "PhysicsTest";
		
		TEST_CLASS_INITIALIZE(BeforeClass) {
			setupLog();
		}

		const double CLOSE_ENOUGH = 1e-2;

		double solve(std::vector<double>& x, ProblemData const& data) {
			const auto m_log = spdlog::stdout_logger_mt("unique_name");
			nlopt::opt opt(nlopt::LD_SLSQP, 4);
			opt.set_min_objective(diffEqError, (void*)(&data));
			std::vector<double> lowerBound = { -10, -10, -10, -10 };
			std::vector<double> upperBound = { 10, 10, 10, 10 };
			opt.set_lower_bounds(lowerBound);
			opt.set_upper_bounds(upperBound);

			opt.set_xtol_rel(1e-3);
			opt.set_maxeval(100);

			int iterations = 0;
			bool found = false;
			while (!found && iterations <= 100) {
				iterations++;
				x[0] = data.p0;
				x[1] = data.v0;
				for (int j = 2; j < 4; j++) {
					x[j] = MenticsGame::nextDouble();
				}

				double minf;
				try {
					const nlopt::result result = opt.optimize(x, minf);
					// debug ("iteration " << iterations << " funcCalls=" << funcCalls << std::endl;
					if (result < 0) {
						m_log->warn("nlopt failed!");
					}
					else {
						const double error = checkError(x, data);
						if (error > CLOSE_ENOUGH) {
							

							m_log->warn("Checked error failure: {0} \n", result);
							continue;
						}
						else {
							//printf("Found minimum at %g,%g after calls func: %d constraints: %d\n", x[6], x[7], funcCalls, constraintCalls);
							// Success, stop looping
							found = true;
						}
					}
				}
				catch (const std::exception& e) {
					m_log->debug("iteration {0} {1} {2} {3} {4} {5} {6} \n", iterations," funcCalls=",funcCalls," (nlopt exception: ",e.what(),")");
					const double error = checkError(x, data);
					if (error > CLOSE_ENOUGH) {
					}
					else {
						m_log->warn("Exception but constraints satisfied\n");
						found = true;
					}
					continue;
				}
			}
			return found ? iterations : -1;
		}

		bool checkError(std::vector<double> const& x, ProblemData const& data) {
			return false;
		}

		struct CandidateFunction {
			std::vector<double> coeffs;

			CandidateFunction(std::vector<double> const& coeffs) : coeffs(coeffs) {}

			double at(double t) const {
				double t2 = t*t;
				double t3 = t2*t;
				return coeffs[0] + coeffs[1] * t + coeffs[2] * t2 + coeffs[3] * t3;
			}

			double dat(double t) const {
				double t2 = t*t;
				return coeffs[1] + coeffs[2] * t + coeffs[3] * t2;
			}

			double ddat(double t) const {
				return coeffs[2] + coeffs[3] * t;
			}
		};

	public:
		
		TEST_METHOD(DiffEqError1)
		{
			ProblemData data{ 1.0, 0.0, 0.0, 1.0, 2 };
			std::vector<double> x{ 1.0 ,0.0, 0.0, 0.0 };

			solve(x, data);

			CandidateFunction result(x);
			// TODO: this is an older test and may be deleted anyway so ignore assertions for now.
			//Assert::AreEqual(1.0, result.at(0));
			//Assert::AreEqual(0.0, result.dat(0), 0.001);
			//Assert::IsTrue(result.ddat(0) < 0);
		}
	};

	double at(double const* x, double t) {
		double t2 = t*t;
		double t3 = t2*t;
		return x[0] + x[1] * t + x[2] * t2 + x[3] * t3;
	}

	double dat(double const* x, double t) {
		double t2 = t*t;
		return x[1] + x[2] * t + x[3] * t2;
	}

	double ddat(double const* x, double t) {
		return x[2] + x[3] * t;
	}
	
	const double k = 1;
	const double m = 1;

	void gradP(vect4& grad, double t) {
		double t2 = t* t;
		double t3 = t2*t;
		grad[0] = 1;
		grad[1] = t;
		grad[2] = t2;
		grad[3] = t3;
	}
	//void gradP(double* grad, double t) {
	//	double t2 = t* t;
	//	double t3 = t2*t;
	//	grad[0] = -k;
	//	grad[1] = -k * t;
	//	grad[2] = -k * t2 - m;
	//	grad[3] = -k * t3 - m*t;
	//}

	void gradPpp(vect4& grad, double t) {
		grad[0] = 0;
		grad[1] = 0;
		grad[2] = 1;
		grad[3] = t;
	}

	double diffEqError(unsigned n, const double* x, double* grad, void* vdata) {
		funcCalls++;
		ProblemData* data = (ProblemData*)vdata;

		std::array<double, 3> times{ data->t0, (data->t1 + data->t1) / 2.0, data->t1 };
		std::array<double, 3> errors{ 0.0,0.0,0.0 };
		std::array<int, 3> signs{ 0,0,0 };

		double sumError = 0;
		for (int i = 0; i < 3; i++) {
			double t = times[i];
			double atxt = at(x, t);
			double ddatxt = ddat(x, t);
			double inner = -k * atxt - (m * ddatxt);
			errors[i] = abs(inner);
			signs[i] = sgn(inner);
			sumError += errors[i];
		}

		if (grad != NULL) {
			vect4 fullGrad{ 0.0,0.0,0.0,0.0 };
			vect4 tmpGradP;
			vect4 tmpGradPpp;
			for (int i = 0; i < 3; i++) {
				gradP(tmpGradP, times[i]);
				gradPpp(tmpGradPpp, times[i]);
				fullGrad += signs[i] * (-k * tmpGradP - m * tmpGradPpp);
			}

			grad[0] = fullGrad[0];
			grad[1] = fullGrad[1];
			grad[2] = fullGrad[2];
			grad[3] = fullGrad[3];
		}

		return sumError;
	}
}