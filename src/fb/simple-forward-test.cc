#undef NDEBUG   // Tests cannot run if NDEBUG flag is defined

#include "fb/simple-forward.h"
#include "fb/test-common.h"
#include "util/kaldi-io.h"

int main(int argc, char** argv) {
  fst::VectorFst<fst::StdArc> fst;
  kaldi::unittest::DummyDecodable dec;
  kaldi::SimpleForward forward(fst, 1E9, 1E-9);

  // Dummy WFST with a single non-final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, false);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -kaldi::kLogZeroDouble, FB_EQ_EPS);

  // Dummy WFST with a single final state. Empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), 0.0, FB_EQ_EPS);

  // Dummy WFST with a single final state. Non-empty input sequence.
  kaldi::unittest::CreateWFST_DummyState(&fst, true);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec) == false);
  kaldi::AssertEqual(forward.TotalCost(), -kaldi::kLogZeroDouble, FB_EQ_EPS);

  // Arbitrary WFST. Empty input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(1.75), FB_EQ_EPS);

  // Arbitrary WFST. Arbitrary input sequence.
  kaldi::unittest::CreateWFST_Arbitrary(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(145.9125), FB_EQ_EPS);
  const double forward_table_1[] = {
    -log(1), -log(1), -log(1), -log(1.5),
    -log(1), -log(3), -log(2.5), -log(5),
    -log(1), -log(6), -log(3.5), -log(9),
    -log(3), -log(19.5), -log(29.25), -log(49.5),
    -log(6), -log(68.325), -log(62.5125), -log(155.175)
  };
  KALDI_ASSERT(kaldi::unittest::CheckTokenTable(
      forward.GetTable(), forward_table_1, 5, 4, FB_EQ_EPS));


  /////////////////////////////////////////
  // A WFST with an epsilon loop.
  /////////////////////////////////////////

  // Empty input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  // kaldi::AssertEqual will fail, since log(1.0) is exactly 0.0, but
  // the total cost is something like 2E-7. kaldi::AssertEqual checks
  // relative error, and it will fail in this case, since one term is 0.0
  KALDI_ASSERT(fabs(forward.TotalCost() - log(1.0)) < FB_EQ_EPS);

  // Arbitrary input sequence
  kaldi::unittest::CreateWFST_EpsilonLoop(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(0.375), FB_EQ_EPS);
  const double forward_table_2[] = {
    -log(1.0), -log(1.0),
    -log(0.5), -log(0.5),
    -log(0.25), -log(0.25),
    -log(0.375), -log(0.375),
    -log(0.375), -log(0.375)
  };
  KALDI_ASSERT(kaldi::unittest::CheckTokenTable(
      forward.GetTable(), forward_table_2, 5, 2, FB_EQ_EPS));


  /////////////////////////////////////////
  // A WFST with an epsilon bucle.
  /////////////////////////////////////////

  // Empty input sequence
  kaldi::unittest::CreateWFST_EpsilonBucle(&fst);
  kaldi::unittest::CreateObservation_Empty(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(4.0 / 6.0), FB_EQ_EPS);

  // Arbitrary input sequence
  kaldi::unittest::CreateWFST_EpsilonBucle(&fst);
  kaldi::unittest::CreateObservation_Arbitrary(&dec);
  KALDI_ASSERT(forward.Forward(&dec));
  kaldi::AssertEqual(forward.TotalCost(), -log(64.0 / 81.0), FB_EQ_EPS);
  const double forward_table_3[] = {
    -log(4.0/3.0), -log(2.0/3.0),
    -log(8.0/9.0), -log(4.0/9.0),
    -log(16.0/27.0), -log(8.0/27.0),
    -log(32.0/27.0), -log(16.0/27.0),
    -log(128.0/81.0), -log(64.0/81.0)
  };
  KALDI_ASSERT(kaldi::unittest::CheckTokenTable(
      forward.GetTable(), forward_table_3, 5, 2, FB_EQ_EPS));

  std::cout << "Test OK.\n";
  return 0;
}
