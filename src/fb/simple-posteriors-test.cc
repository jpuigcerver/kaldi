#undef NDEBUG   // Tests cannot run if NDEBUG flag is defined

#include "fb/simple-backward.h"
#include "fb/simple-forward.h"
#include "fb/test-common.h"
#include "util/kaldi-io.h"

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::SimpleForward forward(fst, 1E20, 1E-9);
  kaldi::SimpleBackward backward(fst, 1E20, 1E-9);
  vector<kaldi::LabelMap> pst;

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_EpsilonBucle(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), backward.TotalCost(), FB_EQ_EPS);
  kaldi::ComputeLabelsPosterior(
      fst, forward.GetTable(), backward.GetTable(), &dec, &pst);
  const double pst_ref_1[] = {
    log(0.0), log(0.5), log(0.5),
    log(0.0), log(0.0), log(1.0),
    log(0.0), log(1.0), log(0.0),
    log(0.0), log(0.65), log(0.35)
  };
  kaldi::unittest::CheckLabelPosteriors(pst, pst_ref_1, 4, 3, FB_EQ_EPS);


  std::cout << "Test OK.\n";
  return 0;
}
