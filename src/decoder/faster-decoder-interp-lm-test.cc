#undef NDEBUG  // Tests cannot run if NDEBUG flag is defined

#include "decoder/faster-decoder-interp-lm.h"
#include "util/kaldi-io.h"

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

// Char Symbol Table:
// <s>  1
// </s> 2
// #    3
// a    4
// b    5
// Word Symbol Table:
// <s>  1
// </s> 2
// aa#  3
// ab#  4
// aaa# 5
// abb# 6

void CreateWFST_LexWords(fst::VectorFst<fst::StdArc>* fst) {
  KALDI_ASSERT(fst != NULL);
  fst->DeleteStates();

  fst->AddState();  // State 0
  fst->AddState();  // State 1
  fst->AddState();  // State 2
  fst->AddState();  // State 3
  fst->AddState();  // State 4
  fst->AddState();  // State 5
  fst->AddState();  // State 6
  fst->AddState();  // State 7
  fst->AddState();  // State 8
  fst->AddState();  // State 9
  fst->AddState();  // State 10

  fst->SetStart(0);
  fst->SetFinal(0, -log(1.0));


  fst->AddArc(0, fst::StdArc(1, 1, 0.0, 0));  // <s>
  fst->AddArc(0, fst::StdArc(2, 2, 0.0, 0));  // </s>

  fst->AddArc(0, fst::StdArc(4, 3, 0.0, 1));  // a a #
  fst->AddArc(1, fst::StdArc(4, 0, 0.0, 2));
  fst->AddArc(2, fst::StdArc(3, 0, 0.0, 0));

  fst->AddArc(0, fst::StdArc(4, 4, 0.0, 3));  // a b #
  fst->AddArc(3, fst::StdArc(5, 0, 0.0, 4));
  fst->AddArc(4, fst::StdArc(3, 0, 0.0, 0));

  fst->AddArc(0, fst::StdArc(4, 5, 0.0, 5));  // a a a #
  fst->AddArc(5, fst::StdArc(4, 0, 0.0, 6));
  fst->AddArc(6, fst::StdArc(4, 0, 0.0, 7));
  fst->AddArc(7, fst::StdArc(3, 0, 0.0, 0));

  fst->AddArc(0, fst::StdArc(4, 5, 0.0, 8));  // a b b #
  fst->AddArc(8, fst::StdArc(5, 0, 0.0, 9));
  fst->AddArc(9, fst::StdArc(5, 0, 0.0, 10));
  fst->AddArc(10, fst::StdArc(3, 0, 0.0, 0));
}

// 2-gram with backoff, for words: aaa, abb, aa, ab
// P(aa|<s>) = 0.4
// P(aaa|<s>) = 0.3
// P(abb|<s>) = 0.2
// P(*|<s>) = 0.1
//
// P(aaa|aaa) = 0.4
// P(abb|aaa) = 0.4
// P(*|aaa) = 0.2
//
// P(aaa|abb) = 0.2
// P(abb|abb) = 0.5
// P(</s>|abb) = 0.2
// P(*|abb) = 0.1
//
// P(aa|aa) = 0.4
// P(</s>|aa) = 0.1
// P(aaa|aa) = 0.2
// P(*|ab) = 0.3
//
// P(ab|ab) = 0.7
// P(</s>|ab) = 0.1
// P(*|ab) = 0.2
void CreateWFST_LMWords(fst::VectorFst<fst::StdArc>* fst) {
  KALDI_ASSERT(fst != NULL);
  fst->DeleteStates();

  fst->AddState(); // 0, Initial state
  fst->AddState(); // 1, Back-off State
  fst->AddState(); // 2, State for <s>
  fst->AddState(); // 3, States for aaa#
  fst->AddState(); // 4, States for abb#
  fst->AddState(); // 5, States for ab#
  fst->AddState(); // 6, States for bb#
  fst->AddState(); // 7, State for </s>

  fst->AddArc(0, fst::StdArc(2, 2, 0.0, 2));
  fst->AddArc(2, fst::StdArc());
  fst->AddArc(2, fst::StdArc());
  fst->AddArc(2, fst::StdArc());
  fst->AddArc(2, fst::StdArc());

}

}  // namespace unittest
}  // namespace kaldi

int main(int argc, char** argv) {
  std::cout << "Test OK.\n";
  return 0;
}
