// fb/test-common.h

#ifndef KALDI_FB_TEST_COMMON_H_
#define KALDI_FB_TEST_COMMON_H_
#include "itf/decodable-itf.h"
#include "fb/simple-common.h"

#include <vector>

namespace kaldi {
namespace unittest {

/// Relative tolerance in floating point operations
#define FB_EQ_EPS 1E-6

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

/// Check if the elements of a TokenMap table are the ones expected. If some
/// element fails, additional information will be shown.
bool CheckTokenTable(
    const std::vector<TokenMap>& table, const double* ref, int nt, int ns,
    double tol = FB_EQ_EPS);
/// Check that all elements in the label posteriors table are the ones expected
/// If some element fails the test, additional information will be shown.
bool CheckLabelPosteriors(
    const std::vector<LabelMap>& table, const double* ref, int nt, int ns,
    double tol = FB_EQ_EPS);


/// A WFST with a single state and no transitions
void CreateWFST_DummyState(fst::VectorFst<fst::StdArc>* fst, bool final);
/// An arbitrary WFST with two input/output symbols + epsilon transitions.
/// The WFST contains non-epsilon self-loops nor bucles and it is
/// non-stochastic.
void CreateWFST_Arbitrary(fst::VectorFst<fst::StdArc>* fst);
/// A WFST with two states and two input/output symbols. One state if the start
/// and emitting state, the other one is the final and has only an epsilon
/// self-loop.
void CreateWFST_EpsilonLoop(fst::VectorFst<fst::StdArc>* fst);
/// Same as before, but the non-emitting state has a epsilon transition to
/// the start/emitting state.
void CreateWFST_EpsilonBucle(fst::VectorFst<fst::StdArc>* fst);


void CreateObservation_Empty(DummyDecodable* decodable);
void CreateObservation_Arbitrary(DummyDecodable* decodable);

}  // namespace unittest
}  // namespace kaldi

#endif  // KALDI_FB_TEST_COMMON_H_
