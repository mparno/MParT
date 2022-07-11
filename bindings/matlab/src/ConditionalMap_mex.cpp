#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "MexMapOptionsConversions.h"
#include "MParT/MapOptions.h"
#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/TriangularMap.h"
#include <Eigen/Dense>
#include "mexplus_eigen.h"


using namespace mpart;
using namespace mexplus;
using MemorySpace = Kokkos::HostSpace;

class ConditionalMapMex {       // The class
public:             
  std::shared_ptr<ConditionalMapBase<MemorySpace>> map_ptr;

  ConditionalMapMex(FixedMultiIndexSet<MemorySpace> const& mset, 
                                                          MapOptions opts){
    map_ptr = MapFactory::CreateComponent<MemorySpace>(mset,opts);
  }

  ConditionalMapMex(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks){
    map_ptr = std::make_shared<TriangularMap<MemorySpace>>(blocks);
  }
  ConditionalMapMex(unsigned int inputDim, unsigned int outputDim, unsigned int totalOrder, MapOptions opts){
    map_ptr = MapFactory::CreateTriangular<MemorySpace>(inputDim,outputDim,totalOrder,opts);
  }
}; //end class

// Instance manager for ConditionalMap.
template class mexplus::Session<ConditionalMapMex>;

namespace {


MEX_DEFINE(newTriMap) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  std::vector<intptr_t> list_id = input.get<std::vector<intptr_t>>(0);
  unsigned int numBlocks = list_id.size();
  std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> blocks(numBlocks);
  for(unsigned int i=0;i<numBlocks;++i){
      const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(list_id.at(i)); 
      blocks.at(i) = condMap.map_ptr;
    }
  output.set(0, Session<ConditionalMapMex>::create(new ConditionalMapMex(blocks)));
}

MEX_DEFINE(newTotalTriMap) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 12);
  OutputArguments output(nlhs, plhs, 1);
  unsigned int inputDim = input.get<unsigned int>(0);
  unsigned int outputDim = input.get<unsigned int>(1);
  unsigned int totalOrder = input.get<unsigned int>(2);

  MapOptions opts = MapOptionsFromMatlab(input.get<std::string>(3),input.get<std::string>(4),
                                         input.get<std::string>(5),input.get<double>(6),
                                         input.get<double>(7),input.get<unsigned int>(8),
                                         input.get<unsigned int>(9),input.get<unsigned int>(10),
                                         input.get<bool>(11));
  output.set(0, Session<ConditionalMapMex>::create(new ConditionalMapMex(inputDim,outputDim,totalOrder,opts)));
}

MEX_DEFINE(newMap) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 10);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  MapOptions opts = MapOptionsFromMatlab(input.get<std::string>(1),input.get<std::string>(2),
                                         input.get<std::string>(3),input.get<double>(4),
                                         input.get<double>(5),input.get<unsigned int>(6),
                                         input.get<unsigned int>(7),input.get<unsigned int>(8),
                                         input.get<bool>(9));
  output.set(0, Session<ConditionalMapMex>::create(new ConditionalMapMex(mset.Fix(),opts)));
}

// Defines MEX API for delete.
MEX_DEFINE(deleteMap) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<ConditionalMapMex>::destroy(input.get(0));
}

MEX_DEFINE(SetCoeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto coeffs = input.get<Eigen::MatrixXd>(1);
  condMap.map_ptr->SetCoeffs(coeffs.col(0));
}

MEX_DEFINE(Coeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto coeffs = KokkosToVec(condMap.map_ptr->Coeffs());
  output.set(0,coeffs);
}

MEX_DEFINE(numCoeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto numcoeffs = condMap.map_ptr->numCoeffs;
  output.set(0,numcoeffs);
}

// MEX_DEFINE(EvaluateImpl) (int nlhs, mxArray* plhs[],
//                  int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 3);
//   OutputArguments output(nlhs, plhs, 0);
//   ConditionalMapMex& condMap = *Session<ConditionalMapMex>::get(input.get(0));
//   Kokkos::View<const double**, Kokkos::HostSpace> pts = input.get<Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>>(1);
//   Kokkos::View<double**, Kokkos::HostSpace> result = input.get<Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace>>(2);
//   condMap.map_ptr->EvaluateImpl(pts,result);
//   // std::cout << evals.transpose();
//   // std::cout << '\n';
//   //output.set(0,evals);
// }

MEX_DEFINE(Evaluate) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  auto pts = input.get<Eigen::MatrixXd>(1);
  Eigen::MatrixXd evals = condMap.map_ptr->Evaluate(pts);
  output.set(0,evals);
}

MEX_DEFINE(LogDeterminant) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  Eigen::MatrixXd pts = input.get<Eigen::MatrixXd>(1);
  Eigen::MatrixXd logDet = condMap.map_ptr->LogDeterminant(pts);
  output.set(0,logDet);
}

MEX_DEFINE(Inverse) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  Eigen::MatrixXd x1 = input.get<Eigen::MatrixXd>(1);
  Eigen::MatrixXd r = input.get<Eigen::MatrixXd>(2);
  Eigen::MatrixXd inv = condMap.map_ptr->Inverse(x1,r);
  output.set(0,inv);
}

MEX_DEFINE(CoeffGrad) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  Eigen::MatrixXd pts = input.get<Eigen::MatrixXd>(1);
  Eigen::MatrixXd sens = input.get<Eigen::MatrixXd>(2);
  Eigen::MatrixXd out = condMap.map_ptr->CoeffGrad(pts,sens);
  output.set(0,out);
}

MEX_DEFINE(LogDeterminantCoeffGrad) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const ConditionalMapMex& condMap = Session<ConditionalMapMex>::getConst(input.get(0));
  Eigen::MatrixXd pts = input.get<Eigen::MatrixXd>(1);
  Eigen::MatrixXd out = condMap.map_ptr->LogDeterminantCoeffGrad(pts);
  output.set(0,out);
}

// MultiIndexSet
MEX_DEFINE(newMutliIndexSetEigen) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  Eigen::MatrixXd multis = input.get<Eigen::MatrixXd>(0);
  output.set(0, Session<MultiIndexSet>::create(new MultiIndexSet(multis.cast<int>())));
}

// Defines MEX API for delete.
MEX_DEFINE(deleteMultiIndexSet) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<MultiIndexSet>::destroy(input.get(0));
}

// Defines MEX API for delete.
MEX_DEFINE(MaxOrders) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.MaxOrders());
}

} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.