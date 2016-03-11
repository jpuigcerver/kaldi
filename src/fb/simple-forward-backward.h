// fb/simple-forward-backward.h

// Copyright 2015  Joan Puigcerver

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_FB_SIMPLE_FORWARD_BACKWARD_H_
#define KALDI_FB_SIMPLE_FORWARD_BACKWARD_H_

#include "fb/simple-common.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "util/stl-utils.h"

namespace kaldi {

class SimpleForwardBackward {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef fst::Fst<StdArc> Fst;
  typedef fst::ArcIterator<Fst> ArcIterator;
  typedef fst::StateIterator<Fst> StateIterator;

  SimpleForwardBackward(
      const Fst &fst, double beam_bkw, double beam_fwd, double delta);
  ~SimpleForwardBackward();

  bool ForwardBackward(DecodableInterface *decodable);

  double LogLikelihood() const { return likelihood_; }
  int32 NumFramesDecoded() const { return num_frames_decoded_; }
  const vector<LabelMap>& LabelPosteriors() const { return label_posteriors_; }

 private:
  void Initialize(DecodableInterface* decodable);
  void InitForward();
  void InitBackward();

  bool Backward(DecodableInterface* decodable);
  bool Forward(DecodableInterface* decodable);

  void BackwardProcessEmitting(DecodableInterface* decodable);
  void BackwardProcessNonemitting();
  void ForwardProcessEmitting(DecodableInterface* decodable);
  void ForwardProcessNonemitting();

  void CheckEpsilonArcs();

  const Fst& fst_;
  vector<LabelMap> label_posteriors_;
  vector<TokenMap> backward_;
  TokenMap* prev_forward_;
  TokenMap* curr_forward_;
  int32 num_frames_decoded_;
  double beam_bkw_;
  double beam_fwd_;
  double delta_;
  double likelihood_;

  /// Store here whether the given arc has input/output epsilon arcs
  /// (first element is for input arcs, second for output arcs)
  unordered_map<StateId, pair<bool, bool> > state_with_epsilon_arcs_;
  /// Store here whether there is any epsilon arc in the WFST at all
  bool wfst_with_epsilon_arcs_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleForwardBackward);
};

}  // namespace kaldi


#endif  // KALDI_FB_SIMPLE_FORWARD_BACKWARD_H_
