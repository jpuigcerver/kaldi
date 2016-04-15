#undef NDEBUG  // Tests cannot run if NDEBUG flag is defined
#include "decoder/lattice-faster-interplm-decoder.h"
#include "util/kaldi-io.h"
#include "fst/script/print.h"

namespace kaldi {
namespace unittest {

// A decodable interface to run the tests
class DummyDecodable : public DecodableInterface {
 private:
  int32 num_states_;
  int32 num_frames_;
  std::vector<double> observations_;

 public:
  DummyDecodable() : DecodableInterface(), num_states_(0), num_frames_(-1) { }

  void Init(int32 num_states, int32 num_frames,
            const std::vector<double>& observations) {
    KALDI_ASSERT(observations.size() == num_states * num_frames);
    num_states_ = num_states;
    num_frames_ = num_frames;
    observations_ = observations;
  }

  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    KALDI_ASSERT(frame >= 0 && frame < NumFramesReady());
    KALDI_ASSERT(state_index > 0 && state_index <= NumIndices());
    return observations_[frame * num_states_ + state_index - 1];
  }

  virtual int32 NumFramesReady() const { return num_frames_; }

  virtual int32 NumIndices() const { return num_states_; }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }
};

void CreateSimpleDeterministicFsts(
    fst::VectorFst<fst::StdArc>* hcl,
    fst::VectorFst<fst::StdArc>* lm1,
    fst::VectorFst<fst::StdArc>* lm2) {

  hcl->AddState();
  hcl->AddState();
  hcl->AddArc(0, fst::StdArc(1, 1, 0.0, 0));
  hcl->AddArc(0, fst::StdArc(2, 2, 0.0, 0));
  hcl->SetFinal(0, 0.0);
  hcl->SetStart(0);

  lm1->AddState();
  lm1->AddState();
  lm1->AddState();
  lm1->AddArc(0, fst::StdArc(1, 1, -log(0.4), 1));
  lm1->AddArc(0, fst::StdArc(2, 2, -log(0.6), 1));
  lm1->AddArc(1, fst::StdArc(1, 1, -log(0.2), 2));
  lm1->AddArc(1, fst::StdArc(2, 2, -log(0.4), 2));
  lm1->SetFinal(1, -log(0.4));
  lm1->SetFinal(2, 0.0);
  lm1->SetStart(0);
  fst::ArcSort(lm1, fst::ILabelCompare<fst::StdArc>());

  lm2->AddState();
  lm2->AddState();
  lm2->AddArc(0, fst::StdArc(1, 1, -log(1.0), 1));
  lm2->AddArc(1, fst::StdArc(2, 2, -log(0.3), 1));
  lm2->SetFinal(1, -log(0.7));
  lm2->SetStart(0);
  fst::ArcSort(lm2, fst::ILabelCompare<fst::StdArc>());
}

void TestDeterministicLM() {
  DummyDecodable decodable;
  fst::VectorFst<fst::StdArc> hcl, lm1, lm2;
  LatticeFasterInterpLmDecoderConfig config;
  config.alpha = 0.7;
  config.beam = 1000;
  CompactLattice lat;
  // Decoding with deterministic input {1}: (0, 0) -> (1, 1)
  {
    decodable.Init(2, 1, {log(1.0), log(0.0)});
    CreateSimpleDeterministicFsts(&hcl, &lm1, &lm2);
    LatticeFasterInterpLmDecoder decoder(config, hcl, lm1, lm2);
    KALDI_ASSERT(decoder.Decode(&decodable));
    KALDI_ASSERT(decoder.GetLattice(&lat, true));

    CompactLattice expected_lat;
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.SetStart(0);
    expected_lat.AddArc(0,
                        CompactLatticeArc(1, 1, CompactLatticeWeight(
                            LatticeWeight(-log(0.58),0), {1}), 1));
    expected_lat.SetFinal(1, CompactLatticeWeight(
        LatticeWeight(-log(.55517241379310344827), 0), {}));
    KALDI_ASSERT(fst::Equivalent(lat, expected_lat));
  }
  // Decoding with deterministic input {2}: (0, 0) -> (1, -)
  {
    decodable.Init(2, 1, {log(0.0), log(1.0)});
    CreateSimpleDeterministicFsts(&hcl, &lm1, &lm2);
    LatticeFasterInterpLmDecoder decoder(config, hcl, lm1, lm2);
    KALDI_ASSERT(decoder.Decode(&decodable));
    KALDI_ASSERT(decoder.GetLattice(&lat, true));

    CompactLattice expected_lat;
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.SetStart(0);
    expected_lat.AddArc(0,
                        CompactLatticeArc(2, 2, CompactLatticeWeight(
                            LatticeWeight(-log(0.7 * 0.6),0), {2}), 1));
    expected_lat.SetFinal(1, CompactLatticeWeight(
        LatticeWeight(-log(0.4), 0), {}));
    KALDI_ASSERT(fst::Equivalent(lat, expected_lat));
  }
  // Decoding with stochastic input (general case)
  {
    decodable.Init(2, 2, {log(0.6), log(0.4), log(0.3), log(0.7)});
    CreateSimpleDeterministicFsts(&hcl, &lm1, &lm2);
    LatticeFasterInterpLmDecoder decoder(config, hcl, lm1, lm2);
    KALDI_ASSERT(decoder.Decode(&decodable));
    KALDI_ASSERT(decoder.GetLattice(&lat, true));
    fst::script::PrintFst(lat, std::cerr);
    std::cerr << std::endl;
    CompactLattice expected_lat;
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.SetStart(0);
    expected_lat.AddArc(0,
                        CompactLatticeArc(1, 1, CompactLatticeWeight(
                            LatticeWeight(-log(0.58), -log(0.6)), {1}), 1));
    expected_lat.AddArc(0,
                        CompactLatticeArc(2, 2, CompactLatticeWeight(
                            LatticeWeight(-log(0.42), -log(0.4)), {2}), 2));
    expected_lat.AddArc(1,
                        CompactLatticeArc(1, 1, CompactLatticeWeight(
                            LatticeWeight(-log(0.056 / 0.58),
                                          -log(0.3)), {1}), 3));
    expected_lat.AddArc(1,
                        CompactLatticeArc(2, 2, CompactLatticeWeight(
                            LatticeWeight(-log(0.121 / 0.58),
                                          -log(0.7)), {2}), 4));
    expected_lat.AddArc(2,
                        CompactLatticeArc(1, 1, CompactLatticeWeight(
                            LatticeWeight(-log(0.084 / 0.42),
                                          -log(0.3)), {1}), 3));
    expected_lat.AddArc(2,
                        CompactLatticeArc(2, 2, CompactLatticeWeight(
                            LatticeWeight(-log(0.168 / 0.42),
                                          -log(0.7)), {2}), 3));
    expected_lat.SetFinal(3, CompactLatticeWeight(
        LatticeWeight(0.331534, 0.0), {}));
    expected_lat.SetFinal(4, CompactLatticeWeight(
        LatticeWeight(-0.217413, 0.0), {}));
    fst::script::PrintFst(expected_lat, std::cerr);
    std::cerr << std::endl;
  }
}

}  // namespace unittest
}  // namespace kaldi

int main(int argc, char** argv) {
  kaldi::unittest::TestDeterministicLM();
  std::cout << "Test OK.\n";
  return 0;
}
