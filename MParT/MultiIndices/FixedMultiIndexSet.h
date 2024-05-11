#ifndef MPART_FIXEDMULTIINDEXSET_H
#define MPART_FIXEDMULTIINDEXSET_H

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#if defined(MPART_HAS_CEREAL)
#include <cereal/access.hpp> // For load_and_construct
#include "MParT/Utilities/Serialization.h"
#endif // MPART_HAS_CEREAL

namespace mpart{

class MultiIndexSet;

template<typename MemorySpace=Kokkos::HostSpace>
class FixedMultiIndexSet
{
public:
    #if defined(MPART_HAS_CEREAL)
    friend class cereal::access;
    #endif // MPART_HAS_CEREAL

    /** @brief Construct a fixed multiindex set in dense form.

        All components of the multiindex are stored.  This requires more memory for high
        dimensional problems, but might be easier to work with for some families of
        basis functions.
    */
    FixedMultiIndexSet(unsigned int                             _dim,
                       Kokkos::View<unsigned int*, MemorySpace> _orders);

    /** @brief Construct a fixed multiindex set in compressed form.

        Only nonzero orders are stored in this representation.   For multivariate polynomials,
        the compressed representation can yield faster polynomial evaluations.  Note that 
        the internal order of `nzDims` is guaranteed to to be increasing for each multiindex.  The
        internal values of `nzDims` might therefore differ from `_nzDims` because they will be 
        sorted.
    */
    FixedMultiIndexSet(unsigned int                             _dim,
                       Kokkos::View<unsigned int*, MemorySpace> _nzStarts,
                       Kokkos::View<unsigned int*, MemorySpace> _nzDims,
                       Kokkos::View<unsigned int*, MemorySpace> _nzOrders);
    /*
    Constructs a total order limited multiindex set.
    */
    FixedMultiIndexSet(unsigned int dim_,
                       unsigned int maxOrder_,
                       unsigned int minOrder_=0);

    
    /** @brief Constructs a new FixedMultiIndex set containing the cartesian product of 
        this set and otherSet.

        Consider two multiindex sets: \f$\mathcal{M}_1 \subset \mathbb{N}^{D_1}\f$ containing multiindices
        of length \f$D_1\f$ and \f$\mathcal{M}_2 \subset \mathbb{N}^{D_2}\f$ containing multiindices of length 
        \f$D_2\f$.  The Cartesian product defined by this function is the subset of \f$\mathbb{N}^{D_1+D_2} given by
        \f[
            \mathcal{M}_1 \bigtimes \mathcal{M}_2 = \{[\alpha_1,\alpha_2] \mid \quad \forall \alpha_1\in\mathcal{M}_1,\,\alpha_2\in\mathcal{M}_2 \},
        \f]

        This function can be more efficient than MultiIndexSet::Cartesian function when no limiter is used.
    */
    FixedMultiIndexSet<MemorySpace> Cartesian(FixedMultiIndexSet<MemorySpace> const& otherSet) const;

    /** @brief Concatenates two sets with the same dimension assuming no duplicate terms exist.
        
        Returns a new multiindex set with a combination of all the terms in this set and all the 
        terms in the other set.  This is equivalent to the union of the sets if no duplicate terms 
        exist.  No checks for duplicates are performed in this function, so it is possible for the 
        resulting FixedMultiIndexSet to have the same term more than once. 
    */
    FixedMultiIndexSet<MemorySpace> Concatenate(FixedMultiIndexSet<MemorySpace> const& otherSet) const;

    // Returns the maximum degree in the dimension dim
    Kokkos::View<const unsigned int*, MemorySpace> MaxDegrees() const;

    // Returns the multiindex with a given linear index
    std::vector<unsigned int> IndexToMulti(unsigned int index) const;

    // Returns the indices of multiindices with nonzero elements in dimension length
    std::vector<unsigned int> NonzeroDiagonalEntries() const;

    // Returns the linear index of a given multiindex.  Returns -1 if not found.
    int MultiToIndex(std::vector<unsigned int> const& multi) const;

    MultiIndexSet Unfix() const;

    void Print() const;

    KOKKOS_INLINE_FUNCTION unsigned int Length() const{
        return dim;
    }

    KOKKOS_INLINE_FUNCTION unsigned int Size() const
    {
        if(this->isCompressed){
            return nzStarts.extent(0)-1;
        }else{
            return nzOrders.extent(0) / dim;
        }
    }

    /** @brief Copy this FixedMultiIndexSet to device memory.
        @return A fixed multiindexset with arrays that live in device memory.
    */
    template<typename OtherMemorySpace>
    FixedMultiIndexSet<OtherMemorySpace> ToDevice();

#if defined(MPART_HAS_CEREAL)

    template<class Archive>
    void save(Archive & ar) const
    {
        ar(dim, isCompressed);
        cereal::save<unsigned int>(ar, nzStarts);
        cereal::save<unsigned int>(ar, nzDims);
        cereal::save<unsigned int>(ar, nzOrders);
        cereal::save<unsigned int>(ar, maxDegrees);
    }

    template<class Archive>
    void load(Archive & ar)
    {
        ar(dim, isCompressed);
        cereal::load<unsigned int>(ar, nzStarts);
        cereal::load<unsigned int>(ar, nzDims);
        cereal::load<unsigned int>(ar, nzOrders);
        cereal::load<unsigned int>(ar, maxDegrees);
    }

    FixedMultiIndexSet(){};
#endif // MPART_HAS_CEREAL

    Kokkos::View<unsigned int*, MemorySpace> nzStarts;
    Kokkos::View<unsigned int*, MemorySpace> nzDims;
    Kokkos::View<unsigned int*, MemorySpace> nzOrders;
    Kokkos::View<unsigned int*, MemorySpace> maxDegrees; // The maximum multiindex value (i.e., degree) in each dimension

private:

    unsigned int dim;
    bool isCompressed;

    void SetupTerms();

    void CalculateMaxDegrees();

    // Computes the number of terms in the multiindexset as well as the total number of nonzero components
    std::pair<unsigned int, unsigned int> TotalOrderSize(unsigned int maxOrder, unsigned int currDim);

    /**
     * @brief
     *
     * @param maxOrder The maximum total order (sum of powers) to include in multiindex set.
     * @param minOrder The minimum total order (inclusive) to include in the the multiindex set.  Can be used to construct nonoverlapping sets.
     * @param currDim The "top" dimension currently being filled in recursive calls to this function.  Starts at 0.
     * @param currNz The current index into the nzDims and nzOrders arrays.
     */
    void FillTotalOrder(unsigned int maxOrder,
                        unsigned int minOrder,
                        Kokkos::View<unsigned int*, MemorySpace> workspace,
                        unsigned int currDim,
                        unsigned int &currTerm,
                        unsigned int &currNz);

    FixedMultiIndexSet(unsigned int                             _dim,
                       Kokkos::View<unsigned int*, MemorySpace> _nzStarts,
                       Kokkos::View<unsigned int*, MemorySpace> _nzDims,
                       Kokkos::View<unsigned int*, MemorySpace> _nzOrders,
                       Kokkos::View<unsigned int*, MemorySpace> _maxDegrees): dim(_dim),
                                                                              isCompressed(true),
                                                                              nzStarts(_nzStarts),
                                                                              nzDims(_nzDims),
                                                                              nzOrders(_nzOrders),
                                                                              maxDegrees(_maxDegrees) {}

}; // class MultiIndexSet

} // namespace mpart


#endif