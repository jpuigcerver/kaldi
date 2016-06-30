#undef NDEBUG  // Tests cannot run if NDEBUG flag is defined
#include "decoder/lattice-faster-interplm-decoder.h"
#include "util/kaldi-io.h"
#include "fst/script/print.h"

#define CHECK_EQUIV_FST(FST1, FST2, MSG, NPATHS) {                      \
    if (!fst::RandEquivalent((FST1), (FST2), (NPATHS))) {               \
      std::cerr << (MSG) << std::endl;                                  \
      std::cerr << "Expected FST:" << std::endl;                        \
      std::cerr << "Start state = " << (FST1).Start() << std::endl;     \
      fst::script::PrintFst((FST1), std::cerr);                         \
      std::cerr << "Actual FST:" << std::endl;                          \
      std::cerr << "Start state = " << (FST2).Start() << std::endl;     \
      fst::script::PrintFst((FST2), std::cerr);                         \
      exit(1);                                                          \
    }                                                                   \
  }

DECLARE_int32(v);

namespace kaldi {
namespace unittest {

void CreateLatticeFromPaths(const std::vector< std::vector<int> >& paths,
                            const std::vector<BaseFloat>& lm_costs,
                            const std::vector<BaseFloat>& ac_costs,
                            Lattice* f) {
  KALDI_ASSERT(paths.size() == lm_costs.size());
  KALDI_ASSERT(paths.size() == ac_costs.size());
  f->DeleteStates();
  f->SetStart(f->AddState());
  for (size_t p = 0; p < paths.size(); ++p) {
    Lattice::StateId j = 0;
    for (size_t i = 0; i < paths[p].size(); ++i) {
      const Lattice::StateId k = f->AddState();
      f->AddArc(j, LatticeArc(paths[p][i], paths[p][i], LatticeWeight(0.0, 0.0), k));
      j = k;
    }
    f->SetFinal(j, LatticeWeight(lm_costs[p], ac_costs[p]));
  }
  fst::Connect(f);
}

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


// Initialize a deterministic DummyDecodable interface
class DeterministicDecodableInitializer {
 public:
  DeterministicDecodableInitializer(int32 num_labels,
                                    const std::vector<int32>& input)
      : num_labels_(num_labels), input_(input) {}

  void operator()(DummyDecodable* d) const {
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


class StochasticDecodableInitializer {
 public:
  StochasticDecodableInitializer(int32 num_labels,
                                 const std::vector<double>& input)
      : num_labels_(num_labels), input_(input) {
    KALDI_ASSERT(input_.size() % num_labels_ == 0);
  }

  void operator()(DummyDecodable* d) const {
    d->Init(num_labels_, input_.size() / num_labels_, input_);
  }

 private:
  const int32 num_labels_;
  const std::vector<double> input_;
};


// Initialize a very simple and deterministic HCL
class DeterministicHCLInitializer {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* hcl) const {
    hcl->DeleteStates();
    hcl->AddState();
    hcl->AddArc(0, fst::StdArc(1, 1, 0.0, 0));
    hcl->AddArc(0, fst::StdArc(2, 2, 0.0, 0));
    hcl->SetStart(0);
    hcl->SetFinal(0, 0.0);
  }
};


// Initialize a HCL wfst with an epsilon loop, which is equivalent to the
// WFST produced by DeterministicHCLInitializer.
class EpsilonLoopHCLInitializer {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* hcl) const {
    hcl->DeleteStates();
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


// Initialize a deterministic LM.
// Language: (a, b, aa, ab, ba, bb)
class DeterministicLMInitializer1 {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* lm) const {
    lm->DeleteStates();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddArc(0, fst::StdArc(1, 1, -log(0.4), 1));
    lm->AddArc(0, fst::StdArc(2, 2, -log(0.6), 1));
    lm->AddArc(1, fst::StdArc(1, 1, -log(0.2), 2));
    lm->AddArc(1, fst::StdArc(2, 2, -log(0.4), 2));
    lm->SetStart(0);
    lm->SetFinal(1, -log(0.4));
    lm->SetFinal(2, 0.0);
  }
};

// Initialize a deterministic LM.
// Language: ab*
class DeterministicLMInitializer2 {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* lm) const {
    lm->DeleteStates();
    lm->AddState();
    lm->AddState();
    lm->AddArc(0, fst::StdArc(1, 1, -log(1.0), 1));
    lm->AddArc(1, fst::StdArc(2, 2, -log(0.3), 1));
    lm->SetStart(0);
    lm->SetFinal(1, -log(0.7));
  }
};

// Initialize a non-deterministic LM, equivalent to DeterministicLMInitializer1,
// but represented as a non-deterministic WFST.
class NondeterministicLMInitializer1 {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* lm) const {
    lm->DeleteStates();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddArc(0, fst::StdArc(1, 1, -log(0.4 * 0.6), 1));
    lm->AddArc(0, fst::StdArc(2, 2, -log(0.3 * 0.6), 1));
    lm->AddArc(0, fst::StdArc(2, 2, -log(0.3 * 0.6), 1));
    lm->AddArc(0, fst::StdArc(1, 1, -log(0.4 * 0.4), 3));
    lm->AddArc(0, fst::StdArc(2, 2, -log(0.6 * 0.4), 3));
    lm->AddArc(1, fst::StdArc(1, 1, -log(0.1), 2));
    lm->AddArc(1, fst::StdArc(1, 1, -log(0.1), 2));
    lm->AddArc(1, fst::StdArc(2, 2, -log(0.2), 2));
    lm->SetStart(0);
    lm->SetFinal(2, -log(1.0 / 0.6));
    lm->SetFinal(3, 0.0);
  }
};

// Equivalent to the Deterministic WFST 1, but introduces an epsilon arc
class NondeterministicLMInitializer2 {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* lm) const {
    lm->DeleteStates();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddArc(0, fst::StdArc(1, 1, -log(0.4), 1));
    lm->AddArc(0, fst::StdArc(2, 2, -log(0.6), 1));
    lm->AddArc(1, fst::StdArc(0, 0, -log(0.4), 2));
    lm->AddArc(1, fst::StdArc(1, 1, -log(0.2), 2));
    lm->AddArc(1, fst::StdArc(2, 2, -log(0.4), 2));
    lm->SetStart(0);
    lm->SetFinal(2, 0.0);
  }
};

// Equivalent to the Deterministic WFST 2, but introduces epsilon arcs
class NondeterministicLMInitializer3 {
 public:
  void operator()(fst::VectorFst<fst::StdArc>* lm) const {
    lm->DeleteStates();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddState();
    lm->AddArc(0, fst::StdArc(0, 0, -log(0.3), 2));
    lm->AddArc(0, fst::StdArc(1, 1, -log(0.7), 1));
    lm->AddArc(1, fst::StdArc(0, 0, 0.0, 3));
    lm->AddArc(2, fst::StdArc(1, 1, 0.0, 3));
    lm->AddArc(3, fst::StdArc(0, 0, -log(0.7), 4));
    lm->AddArc(3, fst::StdArc(2, 2, -log(0.3), 3));
    lm->SetStart(0);
    lm->SetFinal(4, 0.0);
  }
};


// Base class for tests
template <typename D_I, typename HCL_I, typename LM1_I, typename LM2_I>
class LatticeFasterInterpLMDecoderTestBase {
 public:
  typedef D_I DecodableInitializer;
  typedef HCL_I HCLInitializer;
  typedef LM1_I LM1_Initializer;
  typedef LM2_I LM2_Initializer;

  LatticeFasterInterpLMDecoderTestBase(
      float alpha, float beam, float lattice_beam,
      const DecodableInitializer& di, const HCLInitializer& hcli,
      const LM1_Initializer& lm1i, const LM2_Initializer& lm2i) :
      di_(di), hcli_(hcli), lm1i_(lm1i), lm2i_(lm2i) {
    config_.alpha = alpha;
    config_.beam = beam;
    config_.lattice_beam = lattice_beam;
    config_.determinize_lattice = false;
    config_.det_opts.phone_determinize = false;
    config_.det_opts.word_determinize = false;
  }

  virtual void operator()(const std::string& test_name) const {
    DummyDecodable decodable;
    fst::VectorFst<fst::StdArc> hcl;
    fst::VectorFst<fst::StdArc> lm1;
    fst::VectorFst<fst::StdArc> lm2;

    di_(&decodable);
    hcli_(&hcl);
    lm1i_(&lm1);
    lm2i_(&lm2);

    fst::ArcSort(&lm1, fst::ILabelCompare<fst::StdArc>());
    fst::ArcSort(&lm2, fst::ILabelCompare<fst::StdArc>());

    LatticeFasterInterpLmDecoder decoder(config_, hcl, lm1, lm2);
    KALDI_ASSERT(decoder.Decode(&decodable));

    Lattice raw_lat, best_path;
    KALDI_ASSERT(decoder.GetRawLattice(&raw_lat, true));
    KALDI_ASSERT(decoder.GetBestPath(&best_path, true));

    Lattice expected_raw_lat, expected_best_path;
    Init_ExpectedRawLattice(&expected_raw_lat);
    Init_ExpectedBestPath(&expected_best_path);

    // Check equivalence in Lattice semiring.
    // Enough when lattices are determinized in the tropical/lattice-semiring,
    // which is the usual case.
    CHECK_EQUIV_FST(
        expected_best_path, best_path,
        test_name + ": Failed best path equivalence test in lattice-semiring!", 10);
    CHECK_EQUIV_FST(
        expected_raw_lat, raw_lat,
        test_name + ": Failed raw lattice equivalence test in lattice-semiring!", 20);
std::cerr << "Expected:" << std::endl;
fst::script::PrintFst(expected_raw_lat, std::cerr);
std::cerr << "Actual:" << std::endl;
fst::script::PrintFst(raw_lat, std::cerr);
    // Check equivalence in Log semiring.
    //
    {
      fst::VectorFst<fst::LogArc> expected_log_raw_lat, expected_log_best_path;
      fst::VectorFst<fst::LogArc> log_raw_lat, log_best_path;
      ConvertLattice(expected_raw_lat, &expected_log_raw_lat);
      ConvertLattice(expected_best_path, &expected_log_best_path);
      ConvertLattice(raw_lat, &log_raw_lat);
      ConvertLattice(best_path, &log_best_path);
      CHECK_EQUIV_FST(
          expected_log_best_path, log_best_path,
          test_name + ": Failed best path equivalence test in log-semiring!", 10);
      CHECK_EQUIV_FST(
          expected_log_raw_lat, log_raw_lat,
          test_name + ": Failed raw lattice equivalence test in log-semiring!", 20);
    }
  }

 protected:
  virtual void Init_ExpectedBestPath(Lattice* f) const = 0;
  virtual void Init_ExpectedRawLattice(Lattice* f) const = 0;

 private:
  LatticeFasterInterpLmDecoderConfig config_;
  DecodableInitializer di_;
  HCLInitializer hcli_;
  LM1_Initializer lm1i_;
  LM2_Initializer lm2i_;
};


// Decoding with deterministic input {1}: (0, 0) -> (1, 1)
// All the other FSTs are also deterministic
class DeterministicInput_DeterministicHCL_DeterministicLM_Test :
      public LatticeFasterInterpLMDecoderTestBase<
  DeterministicDecodableInitializer, DeterministicHCLInitializer,
  DeterministicLMInitializer1, DeterministicLMInitializer2> {
 public:
  DeterministicInput_DeterministicHCL_DeterministicLM_Test()
      : LatticeFasterInterpLMDecoderTestBase(
            0.7, 1000, 1000,
            DeterministicDecodableInitializer(2, {1}),
            DeterministicHCLInitializer(),
            DeterministicLMInitializer1(),
            DeterministicLMInitializer2()) {}

 protected:
  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    f->AddState();
    f->AddState();
    f->AddArc(0, LatticeArc(1, 1, LatticeWeight(-log(0.58),0), 1));
    f->SetStart(0);
    f->SetFinal(1, LatticeWeight(-log(.55517241379310344827), 0));
  }

  virtual void Init_ExpectedBestPath(Lattice* f) const {
    Init_ExpectedRawLattice(f);
  }
};

// Decoding with deterministic input {1}: (0, 0) -> (1, 1)
// HCL has an epsilon loop, which makes it equivalent to the
// deterministic case.
class DeterministicInput_EpsLoopHCL_DeterministicLM_Test1 :
      public LatticeFasterInterpLMDecoderTestBase<
  DeterministicDecodableInitializer, EpsilonLoopHCLInitializer,
  DeterministicLMInitializer1, DeterministicLMInitializer2> {
 public:
  DeterministicInput_EpsLoopHCL_DeterministicLM_Test1()
      : LatticeFasterInterpLMDecoderTestBase(
            0.7, 1000, 1000,
            DeterministicDecodableInitializer(2, {1}),
            EpsilonLoopHCLInitializer(),
            DeterministicLMInitializer1(),
            DeterministicLMInitializer2()) {}

 protected:
  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    f->DeleteStates();
    f->AddState();
    f->AddState();
    f->AddArc(0, LatticeArc(1, 1, LatticeWeight(-log(0.58),0), 1));
    f->SetStart(0);
    f->SetFinal(1, LatticeWeight(-log(.55517241379310344827), 0));
  }

  virtual void Init_ExpectedBestPath(Lattice* f) const {
    f->DeleteStates();
    Init_ExpectedRawLattice(f);
  }
};

// Decoding with deterministic input {2}: (0, 0) -> (1, -)
// HCL has an epsilon loop, which makes it equivalent to the
// deterministic case.
class DeterministicInput_EpsLoopHCL_DeterministicLM_Test2 :
      public LatticeFasterInterpLMDecoderTestBase<
  DeterministicDecodableInitializer, EpsilonLoopHCLInitializer,
  DeterministicLMInitializer1, DeterministicLMInitializer2> {
 public:
  DeterministicInput_EpsLoopHCL_DeterministicLM_Test2()
      : LatticeFasterInterpLMDecoderTestBase(
            0.7, 1000, 1000,
            DeterministicDecodableInitializer(2, {2}),
            EpsilonLoopHCLInitializer(),
            DeterministicLMInitializer1(),
            DeterministicLMInitializer2()) {}

 protected:
  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    f->DeleteStates();
    f->AddState();
    f->AddState();
    f->SetStart(0);
    f->AddArc(0, LatticeArc(2, 2, LatticeWeight(-log(0.7 * 0.6), 0), 1));
    f->SetFinal(1, LatticeWeight(-log(0.4), 0));
  }

  virtual void Init_ExpectedBestPath(Lattice* f) const {
    f->DeleteStates();
    Init_ExpectedRawLattice(f);
  }
};

class StochasticInput_DeterministicHCL_DeterministicLM_Test :
      public LatticeFasterInterpLMDecoderTestBase<
  StochasticDecodableInitializer, DeterministicHCLInitializer,
  DeterministicLMInitializer1, DeterministicLMInitializer2> {
 public:
  StochasticInput_DeterministicHCL_DeterministicLM_Test()
      : LatticeFasterInterpLMDecoderTestBase(
            0.7, 1000, 1000,
            StochasticDecodableInitializer(
                2, {log(0.6), log(0.4), log(0.3), log(0.7)}),
            DeterministicHCLInitializer(),
            DeterministicLMInitializer1(),
            DeterministicLMInitializer2()) {}

 protected:
  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    f->DeleteStates();
    f->AddState();
    f->AddState();
    f->AddState();
    f->AddState();
    f->AddState();
    f->SetStart(0);
    f->AddArc(0, LatticeArc(1, 1, LatticeWeight(-log(0.58), -log(0.6)), 1));
    f->AddArc(0, LatticeArc(2, 2, LatticeWeight(-log(0.42), -log(0.4)), 2));
    f->AddArc(1, LatticeArc(2, 2, LatticeWeight(-log(0.202 / 0.58), -log(0.7)), 3));
    f->AddArc(1, LatticeArc(1, 1, LatticeWeight(-log(0.056 / 0.58), -log(0.3)), 4));
    f->AddArc(2, LatticeArc(1, 1, LatticeWeight(-log(0.084 / 0.42), -log(0.3)), 4));
    f->AddArc(2, LatticeArc(2, 2, LatticeWeight(-log(0.168 / 0.42), -log(0.7)), 4));
    f->SetFinal(3, LatticeWeight(-log(0.175 / 0.202), 0.0));
    f->SetFinal(4, LatticeWeight(0.0, 0.0));
  }

  virtual void Init_ExpectedBestPath(Lattice* f) const {
    f->DeleteStates();
    f->AddState();
    f->AddState();
    f->AddState();
    f->SetStart(0);
    f->AddArc(0, LatticeArc(1, 1, LatticeWeight(-log(0.58), -log(0.6)), 1));
    f->AddArc(1, LatticeArc(2, 2, LatticeWeight(-log(0.202 / 0.58), -log(0.7)), 2));
    f->SetFinal(2, LatticeWeight(-log(0.175 / 0.202), 0.0));
  }
};

class StochasticInput_DeterministicHCL_NondeterministicLM_Test :
      public LatticeFasterInterpLMDecoderTestBase<
  StochasticDecodableInitializer, DeterministicHCLInitializer,
  NondeterministicLMInitializer1, DeterministicLMInitializer2> {
 public:
  StochasticInput_DeterministicHCL_NondeterministicLM_Test()
      : LatticeFasterInterpLMDecoderTestBase(
            0.7, 1000, 1000,
            StochasticDecodableInitializer(
                2, {log(0.6), log(0.4), log(0.3), log(0.7)}),
            DeterministicHCLInitializer(),
            NondeterministicLMInitializer1(),
            DeterministicLMInitializer2()) {}

 protected:
  virtual void Init_ExpectedBestPath(Lattice* f) const {
    f->DeleteStates();
    f->AddState();
    f->AddState();
    f->AddState();
    f->SetStart(0);
    f->AddArc(0, LatticeArc(1, 1, LatticeWeight(-log(0.468), -log(0.6)), 1));
    f->AddArc(1, LatticeArc(2, 2, LatticeWeight(-log(0.1236 / 0.468), -log(0.7)), 2));
    f->SetFinal(2, LatticeWeight(-log(0.119 / 0.1236), 0.0));
  }

  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    CreateLatticeFromPaths(
        {
          // (1, 1)
          std::vector<int>{1, 1},
          // (1, 2)
          std::vector<int>{1, 2},
          std::vector<int>{1, 2},
          // (2, 1)
          std::vector<int>{2, 1},
          // (2, 2)
          std::vector<int>{2, 2}
        },
        // LM costs
        {
          // (1, 1)
          -log(0.7 * 0.24 * 0.2 / 0.6),
          // (1, 2)
          -log(0.7 * 0.24 * 0.2 / 0.6 + 0.3 * 1 * 0.3 * 0.7),
          -log(0.3 * 1.0 * 0.3 * 0.7),
          // (2, 1)
          -log(0.7 * 0.36 * 0.2 / 0.6),
          // (2, 2)
          -log(0.7 * 0.36 * 0.2 / 0.6)
        },
        // Acoustic costs
        {
          // (1, 1)
          -log(0.6 * 0.3),
          // (1, 2)
          -log(0.6 * 0.7),
          -log(0.6 * 0.7),
          // (2, 1)
          -log(0.4 * 0.3),
          // (2, 2)
          -log(0.4 * 0.7)
        }, f);
  }
};

class StochasticInput_DeterministicHCL_NondeterministicLM_Test2 :
      public LatticeFasterInterpLMDecoderTestBase<
  StochasticDecodableInitializer, DeterministicHCLInitializer,
  NondeterministicLMInitializer2, NondeterministicLMInitializer3> {
 public:
  StochasticInput_DeterministicHCL_NondeterministicLM_Test2()
      : LatticeFasterInterpLMDecoderTestBase(
            0.7, 1000, 1000,
            StochasticDecodableInitializer(
                2, {log(0.6), log(0.4), log(0.3), log(0.7)}),
            DeterministicHCLInitializer(),
            NondeterministicLMInitializer2(),
            NondeterministicLMInitializer3()) {}

 protected:
  virtual void Init_ExpectedBestPath(Lattice* f) const {
    CreateLatticeFromPaths(
        { std::vector<int>{1, 0, 2, 0} },
        {-log(0.7 * 0.4 * 0.4 + 0.3 * 0.7 * 1 * 0.3 * 0.7)},
        {-log(0.6 * 0.7)}, f);
  }

  virtual void Init_ExpectedRawLattice(Lattice* f) const {
    CreateLatticeFromPaths(
        {
          // (1, 1)
          std::vector<int>{1, 1},
          std::vector<int>{0, 1, 1},
          std::vector<int>{1, 0, 1},
          // (1, 2)
          std::vector<int>{0, 1, 2, 0},
          std::vector<int>{1, 0, 2, 0},
          // (2, 1)
          std::vector<int>{0, 2, 1},
          std::vector<int>{2, 1},
          // (2, 2)
          std::vector<int>{0, 2, 2},
          std::vector<int>{2, 2}
        },
        // LM costs
        {
          // (1, 1)
          // Path only through LM1, many paths due to non-determinism
          -log(0.7 * 0.4 * 0.2),
          -log(0.7 * 0.4 * 0.2),
          -log(0.7 * 0.4 * 0.2),
          // (1, 2)
          // (1 path through LM1, 2 paths through LM2)
          -log(0.7 * 0.4 * 0.4 + 0.3 * 0.3 * 1 * 0.3 * 0.7),
          -log(0.7 * 0.4 * 0.4 + 0.3 * 0.7 * 1 * 0.3 * 0.7),
          // (2, 1)
          // Path only through LM1, duplicated due to non-determinism in LM2
          -log(0.7 * 0.6 * 0.2),
          -log(0.7 * 0.6 * 0.2),
          // (2, 2)
          -log(0.7 * 0.6 * 0.4),
          -log(0.7 * 0.6 * 0.4)
        },
        // Acoustic costs
        {
          // (1, 1)
          -log(0.6 * 0.3),
          -log(0.6 * 0.3),
          -log(0.6 * 0.3),
          // (1, 2)
          -log(0.6 * 0.7),
          -log(0.6 * 0.7),
          // (2, 1)
          -log(0.4 * 0.3),
          -log(0.4 * 0.3),
          // (2, 2)
          -log(0.4 * 0.7),
          -log(0.4 * 0.7)
        }, f);
  }
};

}  // namespace unittest
}  // namespace kaldi

int main(int argc, char** argv) {
  using namespace kaldi::unittest;
  FLAGS_v = 3;
  /*DeterministicInput_DeterministicHCL_DeterministicLM_Test()("Test 1");
  DeterministicInput_EpsLoopHCL_DeterministicLM_Test1()("Test 2");
  DeterministicInput_EpsLoopHCL_DeterministicLM_Test2()("Test 3");
  StochasticInput_DeterministicHCL_DeterministicLM_Test()("Test 4"); */
  StochasticInput_DeterministicHCL_NondeterministicLM_Test()("Test 5");
  StochasticInput_DeterministicHCL_NondeterministicLM_Test2()("Test 6");
  std::cout << "Test OK." << std::endl;
  return 0;
}
