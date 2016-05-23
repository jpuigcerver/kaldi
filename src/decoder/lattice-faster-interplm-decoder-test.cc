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

class LatticeFasterInterpLMDecoderTestInterface {
 public:
  virtual void operator()() const = 0;

 protected:
  virtual void Init_HCL(fst::VectorFst<fst::StdArc>* f) const = 0;
  virtual void Init_LM1(fst::VectorFst<fst::StdArc>* f) const = 0;
  virtual void Init_LM2(fst::VectorFst<fst::StdArc>* f) const = 0;
  virtual void Init_Decodable(DummyDecodable* d) const = 0;
  virtual void Init_ExpectedRawLattice(Lattice* f) const = 0;
  virtual void Init_ExpectedBestPath(Lattice* f) const = 0;
};

class LatticeFasterInterpLMDecoderTestBase :
      public LatticeFasterInterpLMDecoderTestInterface {
 public:
  LatticeFasterInterpLMDecoderTestBase(float alpha,
                                       float beam = 1000,
                                       float lattice_beam = 1000) {
    config_.alpha = alpha;
    config_.beam = beam;
    config_.lattice_beam = lattice_beam;
  }

  virtual void operator()() const {
    DummyDecodable decodable;
    fst::VectorFst<fst::StdArc> hcl;
    fst::VectorFst<fst::StdArc> lm1;
    fst::VectorFst<fst::StdArc> lm2;

    Init_Decodable(&decodable);
    Init_HCL(&hcl);
    Init_LM1(&lm1);
    fst::ArcSort(&lm1, fst::ILabelCompare<fst::StdArc>());
    Init_LM2(&lm2);
    fst::ArcSort(&lm2, fst::ILabelCompare<fst::StdArc>());

    LatticeFasterInterpLmDecoder decoder(config_, hcl, lm1, lm2);
    KALDI_ASSERT(decoder.Decode(&decodable));

    Lattice raw_lat, best_path;
    KALDI_ASSERT(decoder.GetRawLattice(&raw_lat, true));
    KALDI_ASSERT(decoder.GetBestPath(&best_path, true));

    Lattice expected_raw_lat, expected_best_path;
    Init_ExpectedRawLattice(&expected_raw_lat);
    Init_ExpectedBestPath(&expected_best_path);

    KALDI_ASSERT(fst::Equivalent(raw_lat, expected_raw_lat));
    KALDI_ASSERT(fst::Equivalent(best_path, expected_best_path));
  }

 private:
  LatticeFasterInterpLmDecoderConfig config_;
};

class SimpleDeterministicInputTest :
      public LatticeFasterInterpLMDecoderTestInterface {
 public:
  SimpleDeterministicInputTest(int32 num_labels,
                               const std::vector<int32>& input)
      : num_labels_(num_labels), input_(input) {}

 protected:
  virtual void Init_Decodable(DummyDecodable* d) const {
    std::vector<double> likelihoods(num_labels_ * input_.size(), log(0.0));
    for (size_t i = 0; i < input_.size(); i += num_labels_) {
      likelihoods[i + input_[i] - 1] = log(1.0);
    }
    d->Init(num_labels_, input_.size(), likelihoods);
  }

 private:
  const int32 num_labels_;
  const std::vector<int32> input_;
};

// Build a very simple and deterministic HCL wfst
class SimpleDeterministicHCLTest :
      public LatticeFasterInterpLMDecoderTestInterface {
 protected:
  virtual void Init_HCL(fst::VectorFst<fst::StdArc>* hcl) const {
    hcl->AddState();
    hcl->AddArc(0, fst::StdArc(1, 1, 0.0, 0));
    hcl->AddArc(0, fst::StdArc(2, 2, 0.0, 0));
    hcl->SetStart(0);
    hcl->SetFinal(0, 0.0);
  }
};

// Build a HCL wfst with an epsilon loop, which is equivalent to
// the wfst produced by SimpleDeterministicHCLTest
class SimpleWithEpsHCLTest :
      public LatticeFasterInterpLMDecoderTestInterface {
 protected:
  virtual void Init_HCL(fst::VectorFst<fst::StdArc>* hcl) const {
    hcl->AddState();
    hcl->AddState();
    hcl->AddArc(0, fst::StdArc(1, 1, 0.0, 1));
    hcl->AddArc(0, fst::StdArc(2, 2, 0.0, 1));
    hcl->AddArc(1, fst::StdArc(0, 0, 0.0, 0));
    hcl->AddArc(1, fst::StdArc(1, 1, -log(0.5), 1));
    hcl->AddArc(1, fst::StdArc(2, 2, -log(0.7), 1));
    hcl->SetStart(0);
    hcl->SetFinal(1, 0.0);
  }
};

class SimpleDeterministicLMTest :
      public LatticeFasterInterpLMDecoderTestInterface {
 protected:
  virtual void Init_LM1(fst::VectorFst<fst::StdArc>* lm1) const {
    lm1->AddState();
    lm1->AddState();
    lm1->AddState();
    lm1->AddArc(0, fst::StdArc(1, 1, -log(0.4), 1));
    lm1->AddArc(0, fst::StdArc(2, 2, -log(0.6), 1));
    lm1->AddArc(1, fst::StdArc(1, 1, -log(0.2), 2));
    lm1->AddArc(1, fst::StdArc(2, 2, -log(0.4), 2));
    lm1->SetStart(0);
    lm1->SetFinal(1, -log(0.4));
    lm1->SetFinal(2, 0.0);
  }

  virtual void Init_LM2(fst::VectorFst<fst::StdArc>* lm2) const {
      lm2->AddState();
      lm2->AddState();
      lm2->AddArc(0, fst::StdArc(1, 1, -log(1.0), 1));
      lm2->AddArc(1, fst::StdArc(2, 2, -log(0.3), 1));
      lm2->SetStart(0);
      lm2->SetFinal(1, -log(0.7));
  }
};

class DeterministicInput1_DeterministicHCL_DeterministicLM_Test :
      public LatticeFasterInterpLMDecoderTestBase,
      public SimpleDeterministicInputTest,
      public SimpleDeterministicHCLTest,
      public SimpleDeterministicLMTest {
 public:
  DeterministicInput1_DeterministicHCL_DeterministicLM_Test()
      : LatticeFasterInterpLMDecoderTestBase(0.7),
        SimpleDeterministicInputTest(2, {1}),
        SimpleDeterministicHCLTest(), SimpleDeterministicLMTest() {}

  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    f->AddState();
    f->AddState();
    f->AddArc(0, LatticeArc(1, 1, LatticeWeight(-log(0.58),0), 1));
    f->SetStart(0);
    f->SetFinal(1, LatticeWeight(-log(.55517241379310344827), 0));
  }
};

void CreateSimpleDeterministicFsts(
    fst::VectorFst<fst::StdArc>* hcl,
    fst::VectorFst<fst::StdArc>* lm1,
    fst::VectorFst<fst::StdArc>* lm2) {
  hcl->DeleteStates();
  lm1->DeleteStates();
  lm2->DeleteStates();

  hcl->AddState();
  hcl->AddArc(0, fst::StdArc(1, 1, 0.0, 0));
  hcl->AddArc(0, fst::StdArc(2, 2, 0.0, 0));
  hcl->SetStart(0);
  hcl->SetFinal(0, 0.0);
  fst::ArcSort(hcl, fst::OLabelCompare<fst::StdArc>());

  lm1->AddState();
  lm1->AddState();
  lm1->AddState();
  lm1->AddArc(0, fst::StdArc(1, 1, -log(0.4), 1));
  lm1->AddArc(0, fst::StdArc(2, 2, -log(0.6), 1));
  lm1->AddArc(1, fst::StdArc(1, 1, -log(0.2), 2));
  lm1->AddArc(1, fst::StdArc(2, 2, -log(0.4), 2));
  lm1->SetStart(0);
  lm1->SetFinal(1, -log(0.4));
  lm1->SetFinal(2, 0.0);
  fst::ArcSort(lm1, fst::ILabelCompare<fst::StdArc>());

  lm2->AddState();
  lm2->AddState();
  lm2->AddArc(0, fst::StdArc(1, 1, -log(1.0), 1));
  lm2->AddArc(1, fst::StdArc(2, 2, -log(0.3), 1));
  lm2->SetStart(0);
  lm2->SetFinal(1, -log(0.7));
  fst::ArcSort(lm2, fst::ILabelCompare<fst::StdArc>());
}

void CreateFsts_TestHCLWithEpsilon(
    fst::VectorFst<fst::StdArc>* hcl,
    fst::VectorFst<fst::StdArc>* lm1,
    fst::VectorFst<fst::StdArc>* lm2) {
  hcl->DeleteStates();
  lm1->DeleteStates();
  lm2->DeleteStates();

  hcl->AddState();
  hcl->AddState();
  hcl->AddArc(0, fst::StdArc(1, 1, 0.0, 1));
  hcl->AddArc(0, fst::StdArc(2, 2, 0.0, 1));
  hcl->AddArc(1, fst::StdArc(0, 0, 0.0, 0));
  hcl->AddArc(1, fst::StdArc(1, 1, -log(0.5), 1));
  hcl->AddArc(1, fst::StdArc(2, 2, -log(0.7), 1));
  hcl->SetStart(0);
  hcl->SetFinal(1, 0.0);
  fst::ArcSort(hcl, fst::OLabelCompare<fst::StdArc>());

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
    CompactLattice expected_lat;
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.AddState();
    expected_lat.SetStart(0);
    expected_lat.AddArc(0, CompactLatticeArc(1, 1, CompactLatticeWeight(
        LatticeWeight(-log(0.58), -log(0.6)), {1}), 1));
    expected_lat.AddArc(0, CompactLatticeArc(2, 2, CompactLatticeWeight(
        LatticeWeight(-log(0.42), -log(0.4)), {2}), 2));
    expected_lat.AddArc(1, CompactLatticeArc(2, 2, CompactLatticeWeight(
        LatticeWeight(-log(0.202 / 0.58), -log(0.7)), {2}), 3));
    expected_lat.AddArc(1, CompactLatticeArc(1, 1, CompactLatticeWeight(
        LatticeWeight(-log(0.056 / 0.58), -log(0.3)), {1}), 4));
    expected_lat.AddArc(2, CompactLatticeArc(1, 1, CompactLatticeWeight(
        LatticeWeight(-log(0.084 / 0.42), -log(0.3)), {1}), 4));
    expected_lat.AddArc(2, CompactLatticeArc(2, 2, CompactLatticeWeight(
        LatticeWeight(-log(0.168 / 0.42), -log(0.7)), {2}), 4));
    expected_lat.SetFinal(3, CompactLatticeWeight(
        LatticeWeight(-log(0.175 / 0.202), 0.0), {}));
    expected_lat.SetFinal(4, CompactLatticeWeight(
        LatticeWeight(0.0, 0.0), {}));
    KALDI_ASSERT(fst::RandEquivalent(lat, expected_lat, 100));
  }
}

}  // namespace unittest
}  // namespace kaldi

int main(int argc, char** argv) {
  using namespace kaldi::unittest;
  //unittest::TestDeterministicLM();
  DeterministicInput1_DeterministicHCL_DeterministicLM_Test t;
  std::cout << "Test OK.\n";
  return 0;
}
