// fb/simple-forward.h

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

#ifndef KALDI_FB_SIMPLE_FORWARD_H_
#define KALDI_FB_SIMPLE_FORWARD_H_

#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "util/stl-utils.h"
#include "fb/simple-common.h"

namespace kaldi {

class SimpleForward {
 public:
  typedef fst::StdArc StdArc;
  typedef StdArc::Label Label;
  typedef StdArc::StateId StateId;
  typedef fst::Fst<StdArc> Fst;
  typedef fst::ArcIterator<Fst> ArcIterator;
  typedef fst::StateIterator<Fst> StateIterator;

  SimpleForward(const Fst &fst, double beam, double delta) :
      fst_(fst), beam_(beam), delta_(delta) { }

  ~SimpleForward();

  /// Decode this utterance.
  /// Returns true if any tokens reached the end of the file (regardless of
  /// whether they are in a final state); query ReachedFinal() after Decode()
  /// to see whether we reached a final state.
  bool Forward(DecodableInterface *decodable);

  /// TotalCost() returns the total cost of reaching any of the final states
  /// from the initial state.
  /// WARNING: This may be different to the likelihood of the observation, when
  /// the WFST contains epsilon transitions!
  double TotalCost() const;

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

  /// Returns the forward table of the in the input labels of the WFST
  /// This typically is the forward table of the transition-ids
  const vector<TokenMap>& GetTable() const {
    return forward_;
  }

 private:
  void InitForward();

  // ProcessEmitting decodes the frame num_frames_decoded_ of the
  // decodable object, then increments num_frames_decoded_.
  void ProcessEmitting(DecodableInterface *decodable);
  void ProcessNonemitting();

  void CheckEpsilonArcs();

  vector<TokenMap> forward_;
  vector<double> scale_factor_;
  const Fst &fst_;
  double beam_;
  double delta_;
  int32 num_frames_decoded_;

  unordered_map<StateId, bool> state_with_epsilon_arc_;
  bool wfst_with_epsilon_arc_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(SimpleForward);
};


} // end namespace kaldi.


#endif
