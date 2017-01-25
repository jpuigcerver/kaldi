#undef NDEBUG   // Tests cannot run if NDEBUG flag is defined

#include "fb/simple-backward.h"
#include "fb/test-common.h"
#include "util/kaldi-io.h"

#define EQ_EPS 1E-5

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::SimpleBackward backward(fst, 1E9, 1E-9);

  // Dummy WFST with a single non-final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, false);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec) == false);
  kaldi::AssertEqual(backward.TotalCost(), -kaldi::kLogZeroDouble, FB_EQ_EPS);

  // Dummy WFST with a single final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), 0.0, FB_EQ_EPS);

  // Dummy WFST with a single final state. Non-empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec) == false);
  kaldi::AssertEqual(backward.TotalCost(), -kaldi::kLogZeroDouble, FB_EQ_EPS);

  // Arbitrary WFST. Empty input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(1.75), FB_EQ_EPS);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(145.9125), FB_EQ_EPS);
  const double backward_table_1[] = {
    -log(145.9125), -log(22.3078), -log(15.70313), -log(8.446875),
    -log(98.85935), -log(16.89375), -log(12.01875), -log(7.25625),
    -log(61.44375), -log(14.5125), -log(9.2625), -log(4.7625),
    -log(10.925), -log(3.175), -log(2.5), -log(1.5),
    -log(1.75), -log(1.5), -log(0.5), -log(0.5),
  };
  KALDI_ASSERT(kaldi::unittest::CheckTokenTable(
      backward.GetTable(), backward_table_1, 5, 4, FB_EQ_EPS));


  /////////////////////////////////////////
  // A WFST with an epsilon loop.
  /////////////////////////////////////////

  // Empty input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  // kaldi::AssertEqual will fail, since log(1.0) is exactly 0.0, but
  // the total cost is something like 2E-7. kaldi::AssertEqual checks
  // relative error, and it will fail in this case, since one term is 0.0
  KALDI_ASSERT(fabs(backward.TotalCost() - log(1.0)) < FB_EQ_EPS);

  // Arbitrary input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(0.375), FB_EQ_EPS);
  const double backward_table_2[] = {
    -log(0.375), -log(0.0),
    -log(0.75), -log(0.0),
    -log(1.5), -log(0.0),
    -log(1.0), -log(0.0),
    -log(1.0), -log(2.0)
  };
  KALDI_ASSERT(kaldi::unittest::CheckTokenTable(
      backward.GetTable(), backward_table_2, 5, 2, FB_EQ_EPS));


  /////////////////////////////////////////
  // A WFST with an epsilon bucle.
  /////////////////////////////////////////

  // Empty input sequence
  kaldi::unittest::CreateWFST_EpsilonBucle(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(4.0 / 6.0), FB_EQ_EPS);

  // Arbitrary input sequence
  kaldi::unittest::CreateWFST_EpsilonBucle(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(backward.Backward(&dec));
  kaldi::AssertEqual(backward.TotalCost(), -log(64.0 / 81.0), FB_EQ_EPS);
  const double backward_table_3[] = {
    -log(64.0/81.0), -log(32.0/81.0),
    -log(32.0/27.0), -log(16.0/27.0),
    -log(16.0/9.0), -log(8.0/9.0),
    -log(8.0/9.0), -log(4.0/9.0),
    -log(2.0/3.0), -log(4.0/3.0)
  };
  KALDI_ASSERT(kaldi::unittest::CheckTokenTable(
      backward.GetTable(), backward_table_3, 5, 2, FB_EQ_EPS));

  std::cout << "Test OK.\n";
  return 0;
}
