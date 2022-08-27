#ifndef MPART_ConditionalMapBase_H
#define MPART_ConditionalMapBase_H

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "MParT/ParameterizedFunctionBase.h"

#include <Eigen/Core>

namespace mpart {

    template<typename MemorySpace>
    class TriangularMap;

    /**
     @brief Provides an abstract base class for conditional transport maps where the input dimension might be larger than output dimension.
     @details
      This class provides an interface for functions \f$T:\mathbb{R}^{N+M}\rightarrow \mathbb{R}^M\f$, where $N\geq 0$. Let
      \f$x_1\in \mathbb{R}^N\f$ denote the first block of inputs and \f$x_2\in\mathbb{R}^M\f$ denote the second block of inputs.
      Let \f$r\in\mathbb{R}^M\f$ denote the map output, \f$r=T(x_2; x_1)\f$.  The conditional maps implemented by children of this
      class guarantee that for fixed \f$x_1\f$, the conditional mapping from \f$x_1 \rightarrow r\f$ is invertible and the
      Jacobian matrix with respect to \f$x_2\f$, \f$\nabla_{x_2} T\f$, is positive definite.
     */
    template<typename MemorySpace>
    class ConditionalMapBase : public ParameterizedFunctionBase<MemorySpace>{

        friend class TriangularMap<MemorySpace>;

    public:

        /**
         @brief Construct a new Conditional Map Base object
         @param inDim The dimension \f$N\f$ of the input to this map.
         @param outDim The dimension \f$M\f$ of the output from this map.
         */
        ConditionalMapBase(unsigned int inDim, unsigned int outDim, unsigned int nCoeffs) : ParameterizedFunctionBase<MemorySpace>(inDim, outDim, nCoeffs){};

        virtual ~ConditionalMapBase() = default;
        
        /** For Monotone parameterizations that are based on a non-monotone base function, this function will return the base function.  If the monotone parameterization is 
            not constructed from a non-monotone base, then this function will return a nullptr.
        */
        virtual std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> GetBaseFunction(){return nullptr;};

        /** @brief Computes the log determinant of the map Jacobian.
        For a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R}^M\f$ with \f$M\leq N\f$ and components \f$T_i(x_{1:N-M+i})\f$, this
        function computes the determinant of the Jacobian of \f$T\f$ with respect to \f$x_{N-M:N}\f$.  While the map is rectangular,
        the Jacobian with respect to these inputs will be square.  The fact that the map is lower triangular will then imply that
        the determinant is given by the product of diagonal derviatives
        \f[
            \det{\nabla_{x_{N-M:N}} T} = \prod_{i=1}^M \frac{\partial T_i}{\partial x_{N-M+i}}.
        \f]
        @param pts The points where we want to evaluate the log determinant.
        */
        template<typename AnyMemorySpace>
        Kokkos::View<double*, AnyMemorySpace> LogDeterminant(StridedMatrix<const double, AnyMemorySpace> const& pts);

        /** LogDeterminant function with conversion between default view layout and const strided matrix. */
        template<typename... AllTraits>
        Kokkos::View<double*, typename Kokkos::View<double**, AllTraits...>::memory_space> LogDeterminant(Kokkos::View<double**, AllTraits...> pts){StridedMatrix<const double, typename Kokkos::View<double**,AllTraits...>::memory_space> newpts(pts); return this->LogDeterminant(newpts);}

        /** LogDeterminant function with conversion from regular strided matrix to const strided matrix. */
        template<typename AnyMemorySpace>
        Kokkos::View<double*, AnyMemorySpace> LogDeterminant(StridedMatrix<double, AnyMemorySpace> const& pts){StridedMatrix<const double, AnyMemorySpace> newpts(pts); return this->LogDeterminant(newpts);}


        /** LogDeterminant function with conversion from Eigen to Kokkos (and possibly copy to/from device.) */
        Eigen::VectorXd LogDeterminant(Eigen::Ref<const Eigen::RowMatrixXd> const& pts);

        virtual void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                        StridedVector<double, MemorySpace>              output) = 0;


        /** Returns the value of \f$x_2\f$ given \f$x_1\f$ and \f$r\f$.   Note that the \f$x1\f$ view may contain more
            than \f$N\f$ rows, but only the first \f$N\f$ will be used in this function.
        */
        template<typename AnyMemorySpace>
        StridedMatrix<double, AnyMemorySpace> Inverse(StridedMatrix<const double, AnyMemorySpace> const& x1,
                                                      StridedMatrix<const double, AnyMemorySpace> const& r);

        /** Inverse function with conversion between general view type and const strided matrix. */
        template<typename ViewType1, typename ViewType2>
        StridedMatrix<double, typename ViewType1::memory_space> Inverse(ViewType1 x,  ViewType2 r){
            StridedMatrix<const double, typename ViewType1::memory_space> newx(x); 
            StridedMatrix<const double, typename ViewType2::memory_space> newr(r); 
            return this->Inverse(newx,newr);
        }

        /** Inverse function with conversion between eigen matrix and Kokkos view. */
        Eigen::RowMatrixXd Inverse(Eigen::Ref<const Eigen::RowMatrixXd> const& x1, 
                                   Eigen::Ref<const Eigen::RowMatrixXd> const& r);

        virtual void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                 StridedMatrix<const double, MemorySpace> const& r,
                                 StridedMatrix<double, MemorySpace>              output) = 0;
        /**
           @brief Computes the gradient of the log determinant with respect to the map coefficients.
           @details For a map \f$T(x; w) : \mathbb{R}^N \rightarrow \mathbb{R}^M\f$ parameterized by coefficients \f$w\in\mathbb{R}^K\f$,
           this function computes 
           \f[
            \nabla_w \det{\nabla_x T(x_i; w)},
          \f] 
           at multiple points \f$x_i\f$.
           @param pts A collection of points where we want to evaluate the gradient.  Each column corresponds to a point.
           @return A matrix containing the coefficient gradient at each input point.  The \f$i^{th}\f$ column  contains \f$\nabla_w \det{\nabla_x T(x_i; w)}\f$.
        */
        template<typename AnyMemorySpace>
        StridedMatrix<double, AnyMemorySpace> LogDeterminantCoeffGrad(StridedMatrix<const double, AnyMemorySpace> const& pts);

        /** Include conversion between general view type and Strided matrix. */
        template<typename ViewType>
        StridedMatrix<double, typename ViewType::memory_space> LogDeterminantCoeffGrad(ViewType pts){
            StridedMatrix<const double, typename ViewType::memory_space> newpts(pts); 
            return  this->LogDeterminantCoeffGrad(newpts);
        }

        /** Evaluation with additional conversion of Eigen matrix to Kokkos unmanaged view. */
        Eigen::RowMatrixXd LogDeterminantCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts);

        virtual void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                 StridedMatrix<double, MemorySpace>              output) = 0;


    }; // class ConditionalMapBase<MemorySpace>
}

#endif