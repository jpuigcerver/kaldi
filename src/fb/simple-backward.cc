// fb/simple-backward.cc

// Copyright 2015 Joan Puigcerver

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

#include "fb/simple-backward.h"
#include "fb/queue-set.h"
#include <algorithm>

namespace kaldi {


SimpleBackward::~SimpleBackward() {
  backward_.clear();
}


void SimpleBackward::InitBackward(DecodableInterface *decodable) {
  // clean up from last time:
  backward_.clear();
  // initialize decoding:
  backward_.push_back(TokenMap());
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    if (fst_.Final(siter.Value()) != -kaldi::kLogZeroDouble) {
      backward_.back().insert(make_pair(
          siter.Value(), fst_.Final(siter.Value()).Value()));
    }
  }
  // We need this asserts because the backward pass cannot be done in a
  // sequence processed online, that is, we need the whole sequence to be
  // available from the beginning
  KALDI_ASSERT(decodable->NumFramesReady() >= 0);
  KALDI_ASSERT(decodable->IsLastFrame(decodable->NumFramesReady() - 1));
  num_frames_decoded_ = 0;
  CheckEpsilonArcs();
  ProcessNonemitting();
}


bool SimpleBackward::Backward(DecodableInterface *decodable) {
  InitBackward(decodable);
  while(num_frames_decoded_ < decodable->NumFramesReady()) {
    backward_.push_back(TokenMap());
    ProcessEmitting(decodable);
    ProcessNonemitting();
    PruneToks(beam_, &backward_.back());
  }
  // Reverse backward matrix [T, T-1, ..., 1] -> [1, ..., T-1, T]
  std::reverse(backward_.begin(), backward_.end());
  return (!backward_.front().empty());
}


void SimpleBackward::ProcessEmitting(DecodableInterface *decodable) {
  // Processes emitting arcs for one frame.  Propagates from prev_toks_ to
  // curr_toks_.
  const TokenMap& prev_toks_ = backward_[backward_.size() - 2];
  TokenMap& curr_toks_ = backward_.back();
  const int32 frame = decodable->NumFramesReady() - num_frames_decoded_ - 1;
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    const StateId state = siter.Value();
    for (ArcIterator aiter(fst_, state); !aiter.Done(); aiter.Next()) {
      const StdArc& arc = aiter.Value();
      TokenMap::const_iterator ptok = prev_toks_.find(arc.nextstate);
      if (arc.ilabel == 0 || ptok == prev_toks_.end())
        continue;
      const double acoustic_cost =
          -decodable->LogLikelihood(frame, arc.ilabel);
      Token& ctok = curr_toks_.insert(make_pair(
          state, Token(-kaldi::kLogZeroDouble))).first->second;
      ctok.UpdateEmitting(
          ptok->second.cost, arc.weight.Value(), acoustic_cost);
    }
  }
  num_frames_decoded_++;
}


void SimpleBackward::ProcessNonemitting() {
  // Processes nonemitting arcs for one frame.  Propagates within
  // curr_toks_.
  if (!wfst_with_epsilon_arc_) return;
  TokenMap& curr_toks_ = backward_.back();
  QueueSet<StateId> queue_set;
  for (TokenMap::iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    if (!state_with_epsilon_arc_[tok->first]) continue;
    queue_set.push(tok->first);
    tok->second.last_cost = tok->second.cost;
  }

  while (!queue_set.empty()) {
    const StateId state = queue_set.front();
    queue_set.pop();

    Token& ptok = curr_toks_.find(state)->second;
    const double last_cost = ptok.last_cost;
    ptok.last_cost = -kaldi::kLogZeroDouble;

    for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
      for (ArcIterator aiter(fst_, siter.Value()); !aiter.Done();
           aiter.Next()) {
        const StdArc& arc = aiter.Value();
        if (arc.ilabel != 0 || arc.nextstate != state) continue;
        Token& ctok = curr_toks_.insert(make_pair(
            siter.Value(), Token(-kaldi::kLogZeroDouble))).first->second;
        if (ctok.UpdateNonEmitting(last_cost, arc.weight.Value(), delta_)) {
          queue_set.push(siter.Value());
        }
      }
    }
  }
}


double SimpleBackward::TotalCost() const {
  double total_cost = -kaldi::kLogZeroDouble;
  if (fst_.Start() != fst::kNoStateId) {
    TokenMap::const_iterator tok = backward_.front().find(fst_.Start());
    total_cost = tok == backward_.front().end() ?
        -kaldi::kLogZeroDouble : tok->second.cost;
  }
  return total_cost;
}


void SimpleBackward::CheckEpsilonArcs() {
  wfst_with_epsilon_arc_ = false;
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    state_with_epsilon_arc_[siter.Value()] = false;
  }

  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    for (ArcIterator aiter(fst_, siter.Value()); !aiter.Done(); aiter.Next()) {
      if (aiter.Value().ilabel == 0) {
        state_with_epsilon_arc_[aiter.Value().nextstate] = true;
        wfst_with_epsilon_arc_ = true;
        break;
      }
    }
  }
}


} // end namespace kaldi.
