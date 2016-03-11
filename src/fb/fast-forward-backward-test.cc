#undef NDEBUG   // Tests cannot run if NDEBUG flag is defined

#include "fb/fast-forward-backward.h"
#include "fb/test-common.h"
#include "util/kaldi-io.h"

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> vfst;
  fst::ForwardBackwardFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::FastForwardBackward fb(fst, 1E20, 1E20, 1E-9);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_EpsilonBucle(&vfst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  fst = vfst;

  KALDI_ASSERT(fb.ForwardBackward(&dec));

  const vector<kaldi::LabelMap>& pst = fb.LabelPosteriors();
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
