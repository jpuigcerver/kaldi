// fb/test-common.cc

#include "fb/test-common.h"

namespace kaldi {
namespace unittest {

bool CheckTokenTable(
    const std::vector<TokenMap>& table, const double* ref, int nt, int ns,
    double tol) {
  if (table.size() != nt) {
    KALDI_ERR << "Number of elements (" << table.size()
              << ") does not match reference (" << nt << ")";
    return false;
  }
  for (int t = 0; t < nt; ++t) {
    for (int s = 0; s < ns; ++s) {
      const TokenMap::const_iterator tok = table[t].find(s);
      const double tv = tok != table[t].end() ?
          tok->second.cost : -kaldi::kLogZeroDouble;
      const double rv = ref[t * ns + s];
      if ((rv > tol && !kaldi::ApproxEqual(tv, rv, tol)) ||
          (rv < tol && fabs(tv - rv) > tol)) {
        KALDI_ERR << "Element t=" << t << ",s=" << s << " in table (" << tv
                  << ") does not match reference (" << rv << ")";
        return false;
      }
    }
  }
  return true;
}

bool CheckLabelPosteriors(
    const std::vector<LabelMap>& table, const double* ref, int nt, int ns,
    double tol) {
  if (table.size() != nt) {
    KALDI_ERR << "Number of elements (" << table.size()
              << ") does not match reference (" << nt << ")";
    return false;
  }
  for (int t = 0; t < nt; ++t) {
    for (int s = 0; s < ns; ++s) {
      const LabelMap::const_iterator lab = table[t].find(s);
      const double tv = lab != table[t].end() ?
          lab->second : kaldi::kLogZeroDouble;
      const double rv = ref[t * ns + s];
      if ((rv > tol && !kaldi::ApproxEqual(tv, rv, tol)) ||
          (rv < tol && fabs(tv - rv) > tol)) {
        KALDI_ERR << "Element t=" << t << ",s=" << s << " in table (" << tv
                  << ") does not match reference (" << rv << ")";
        return false;
      }
    }
  }
  return true;
}


void CreateWFST_DummyState(fst::VectorFst<fst::StdArc>* fst, bool final) {
  fst->DeleteStates();
  fst->AddState();
  fst->SetStart(0);
  if (final) {
    fst->SetFinal(0, -log(1.0));
  }
}

// An arbitrary WFST with two input/output symbols + epsilon transitions.
// The WFST contains non-epsilon loops and it is non-stochastic.
void CreateWFST_Arbitrary(fst::VectorFst<fst::StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();  // State 0
  fst->AddState();  // State 1
  fst->AddState();  // State 2
  fst->AddState();  // State 3

  fst->AddArc(0, fst::StdArc(1, 1, -log(1.0), 0));
  fst->AddArc(0, fst::StdArc(2, 2, -log(1.0), 0));
  fst->AddArc(0, fst::StdArc(0, 0, -log(1.0), 1));
  fst->AddArc(0, fst::StdArc(1, 1, -log(1.0), 1));
  fst->AddArc(0, fst::StdArc(2, 2, -log(1.0), 1));
  fst->AddArc(0, fst::StdArc(0, 0, -log(0.5), 2));

  fst->AddArc(1, fst::StdArc(2, 2, -log(0.5), 1));
  fst->AddArc(1, fst::StdArc(0, 0, -log(0.5), 2));
  fst->AddArc(1, fst::StdArc(1, 1, -log(1.0), 2));
  fst->AddArc(1, fst::StdArc(0, 0, -log(0.5), 3));

  fst->AddArc(2, fst::StdArc(0, 0, -log(1.0), 3));
  fst->AddArc(2, fst::StdArc(1, 1, -log(1.0), 3));
  fst->AddArc(2, fst::StdArc(2, 2, -log(1.0), 3));

  fst->AddArc(3, fst::StdArc(1, 1, -log(0.5), 1));
  fst->AddArc(3, fst::StdArc(2, 2, -log(0.5), 1));

  fst->SetStart(0);
  fst->SetFinal(1, -log(1.0));
  fst->SetFinal(3, -log(0.5));
}

// A WFST with an epsilon loop which is k-closed for the log-semiring
void CreateWFST_EpsilonLoop(fst::VectorFst<fst::StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();  // State 0
  fst->AddState();  // State 1

  fst->AddArc(0, fst::StdArc(0, 0, -log(0.5), 1));
  fst->AddArc(0, fst::StdArc(1, 1, -log(0.5), 0));
  fst->AddArc(0, fst::StdArc(2, 2, -log(0.5), 0));
  fst->AddArc(1, fst::StdArc(0, 0, -log(0.5), 1));

  fst->SetStart(0);
  fst->SetFinal(1, -log(1.0));
}

void CreateWFST_EpsilonBucle(fst::VectorFst<fst::StdArc>* fst) {
  fst->DeleteStates();

  fst->AddState();  // State 0
  fst->AddState();  // State 1

  fst->AddArc(0, fst::StdArc(0, 0, -log(0.5), 1));
  fst->AddArc(0, fst::StdArc(1, 1, -log(0.5), 0));
  fst->AddArc(0, fst::StdArc(2, 2, -log(0.5), 0));
  fst->AddArc(1, fst::StdArc(0, 0, -log(0.5), 0));

  fst->SetStart(0);
  fst->SetFinal(1, -log(1.0));
}

void CreateObservation_Empty(DummyDecodable* decodable) {
  decodable->Init(2, 0, std::vector<double>());
}

void CreateObservation_Arbitrary(DummyDecodable* decodable) {
  std::vector<double> observation(2 * 4);
  // t = 1
  observation[0] = log(0.5);
  observation[1] = log(0.5);
  // t = 2
  observation[2] = log(0.0);
  observation[3] = log(1.0);
  // t = 3
  observation[4] = log(3.0);
  observation[5] = log(0.0);
  // t = 4
  observation[6] = log(1.3);
  observation[7] = log(0.7);
  decodable->Init(2, 4, observation);
}

}  // namespace unittest
}  // namespace kaldi
