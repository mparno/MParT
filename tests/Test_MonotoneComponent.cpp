#include <catch2/catch_all.hpp>

#include "MParT/MonotoneComponent.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing monotone component integrand", "[MonotoneIntegrand]") {
    
    const double testTol = 1e-7;

    unsigned int dim = 1;
    unsigned int maxDegree = 1; 
    FixedMultiIndexSet mset(dim, maxDegree); // Create a total order limited fixed multindex set

    // Make room for the cache
    std::vector<double> cache(dim*maxDegree);

    
    Kokkos::View<unsigned int*> startPos("Starting Positions", dim+1);
    for(unsigned int i=0; i<dim+1; ++i)
        startPos(i) = i;

    Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
    coeffs(0) = 1.0; // Constant term
    coeffs(1) = 1.0; // Linear term

    auto maxDegrees = mset.MaxDegrees();
    CachedMonotoneIntegrand<ProbabilistHermite, Exp> integrand(&cache[0], startPos, maxDegrees, mset, coeffs);
    
    CHECK(integrand(0.0) == Approx(exp(1)).epsilon(testTol));
    CHECK(integrand(0.5) == Approx(exp(1)).epsilon(testTol));
    CHECK(integrand(-0.5) == Approx(exp(1)).epsilon(testTol));
}


TEST_CASE( "Testing monotone component evaluation in 1d", "[MonotoneComponent1d]" ) {

    const double testTol = 1e-7;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;
    double lb = -2.0;
    double ub = 2.0;

    Kokkos::View<double**> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;
    
    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - f(0) + int_0^x exp( d f(t) ) dt =  1.0 + int_0^x exp(1) dt = 1 + |x| * exp(1)
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        
        
        Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
        coeffs(0) = 1.0; // Constant term
        coeffs(1) = 1.0; // Linear term

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol);

        MonotoneComponent<ProbabilistHermite, Exp, AdaptiveSimpson> comp(mset, quad);

        Kokkos::View<double*> output = comp.Evaluate(evalPts, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(output(i) == Approx(1+exp(1)*std::abs(evalPts(0,i))).epsilon(testTol));
        }
    }


    /* Create and evaluate a quadratic map
       - Set coefficients so that f(x) = 1.0 + x + 0.5*x^2
       - df/dt = 1.0 + t
       - f(0) + int_0^x exp( df/dt ) dt =  1.0 + int_0^x exp(1+t) dt = 1+exp(1+x)
    */
    SECTION("Quadratic Map"){
        unsigned int maxDegree = 2; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        
        Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol);

        MonotoneComponent<ProbabilistHermite, Exp, AdaptiveSimpson> comp(mset, quad);

        Kokkos::View<double*> output = comp.Evaluate(evalPts, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(output(i) == Approx(1+exp(1)*(exp(evalPts(0,i))-1)).epsilon(testTol));
        }
    }
}