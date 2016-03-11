// fb/simple-forward-backward.cc

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

#include "fb/simple-forward-backward.h"
#include "fb/queue-set.h"
#include <algorithm>

namespace kaldi {

SimpleForwardBackward::SimpleForwardBackward(
    const Fst &fst, double beam_bkw, double beam_fwd, double delta) :
    fst_(fst), prev_forward_(new TokenMap), curr_forward_(new TokenMap),
    num_frames_decoded_(0), beam_bkw_(beam_bkw), beam_fwd_(beam_fwd),
    delta_(delta), likelihood_(kaldi::kLogZeroDouble),
    wfst_with_epsilon_arcs_(false) {
}

SimpleForwardBackward::~SimpleForwardBackward() {
  delete prev_forward_;
  delete curr_forward_;
}

void SimpleForwardBackward::CheckEpsilonArcs() {
  wfst_with_epsilon_arcs_ = false;
  state_with_epsilon_arcs_.clear();
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    state_with_epsilon_arcs_[siter.Value()] = make_pair(false, false);
  }
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    const StateId i = siter.Value();
    for (ArcIterator aiter(fst_, i); !aiter.Done(); aiter.Next()) {
      const StateId j = aiter.Value().nextstate;
      const Label label = aiter.Value().ilabel;
      if (label == 0) {
        state_with_epsilon_arcs_[j].first  = true;
        state_with_epsilon_arcs_[i].second = true;
        wfst_with_epsilon_arcs_ = true;
      }
    }
  }
}

void SimpleForwardBackward::Initialize(DecodableInterface* decodable) {
  CheckEpsilonArcs();
  backward_.clear();
  prev_forward_->clear();
  curr_forward_->clear();
  // We need this asserts because the backward pass cannot be done in a
  // sequence processed online, that is, we need the whole sequence to be
  // available from the beginning
  KALDI_ASSERT(decodable->NumFramesReady() >= 0);
  KALDI_ASSERT(decodable->IsLastFrame(decodable->NumFramesReady() - 1));
}

bool SimpleForwardBackward::ForwardBackward(DecodableInterface *decodable) {
  Initialize(decodable);
  if (!Backward(decodable)){
    KALDI_WARN << "No tokens survived the end of the backward pass!";
    return false;
  }
  if (!Forward(decodable)) {
    KALDI_WARN << "No tokens survived the end of the forward pass!";
    return false;
  }
  return true;
}

///////////////////////////////////////////////////////////////////////////
/// BACKWARD PASS METHODS ...
///////////////////////////////////////////////////////////////////////////

bool SimpleForwardBackward::Backward(DecodableInterface *decodable) {
  // Initialize backward trellis with all final states
  backward_.push_back(TokenMap());
  for (StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    if (fst_.Final(siter.Value()) != -kaldi::kLogZeroDouble) {
      backward_.back().insert(make_pair(
          siter.Value(), fst_.Final(siter.Value()).Value()));
    }
  }
  // Process nonemitting arcs to reach all states with epsilon-transition
  // to any of the final states
  num_frames_decoded_ = 0;
  BackwardProcessNonemitting();
  // Process all input frames
  while(num_frames_decoded_ < decodable->NumFramesReady()) {
    PruneToks(beam_bkw_, &backward_.back());
    backward_.push_back(TokenMap());
    BackwardProcessEmitting(decodable);
    BackwardProcessNonemitting();
  }
  // Reverse backward matrix [T, T-1, ..., 1] -> [1, ..., T-1, T]
  std::reverse(backward_.begin(), backward_.end());
  return (!backward_.front().empty());
}

void SimpleForwardBackward::BackwardProcessEmitting(
    DecodableInterface* decodable) {
  // Processes emitting arcs for one frame. Propagates from prev_toks_ to
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

void SimpleForwardBackward::BackwardProcessNonemitting() {
  // If the wfst has only emitting arcs, we are done
  if (!wfst_with_epsilon_arcs_) return;

  // Push current active tokens (with epsilon input arcs) to the queue
  TokenMap& curr_toks_ = backward_.back();
  QueueSet<StateId> queue_set;
  for (TokenMap::iterator tok = curr_toks_.begin();
       tok != curr_toks_.end(); ++tok) {
    if (state_with_epsilon_arcs_[tok->first].first) {
      queue_set.push(tok->first);
      tok->second.last_cost = tok->second.cost;
    }
  }

  // Process all states in the queue
  while (!queue_set.empty()) {
    const StateId state = queue_set.front();
    queue_set.pop();

    Token& ptok = curr_toks_.find(state)->second;
    const double last_cost = ptok.last_cost;
    ptok.last_cost = -kaldi::kLogZeroDouble;

    // Traverse all input arcs (with epsilon labels) to the current state
    // TODO(joapuipe): This is currently very inneficient, because we have to
    // traverse all the graph, in order to check which states arrive to the
    // considered state
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


///////////////////////////////////////////////////////////////////////////
/// FORWARD PASS METHODS ...
///////////////////////////////////////////////////////////////////////////

bool SimpleForwardBackward::Forward(DecodableInterface* decodable) {
  // Initialize forward trellis with the initial state
  StateId start_state = fst_.Start();
  if (start_state != fst::kNoStateId) {
    curr_forward_->insert(make_pair(start_state, 0.0));
  }
  num_frames_decoded_ = 0;
  ForwardProcessNonemitting();

  // Compute total likelihood using forward_[0] and backward_[0].
  // Note: Remember that we have backward_[0] because we did the full
  // backward pass before.
  likelihood_ = ComputeLikelihood(*curr_forward_, backward_[0]);

  while(!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    // Prune tokens from the previous timestep
    PruneToksForwardBackward(
        -likelihood_, beam_fwd_, curr_forward_,
        &backward_[num_frames_decoded_]);
    // Compute label posteriors at time t.
    label_posteriors_.push_back(LabelMap());
    ComputeLabelsPosteriorAtTimeT(
        fst_, decodable, num_frames_decoded_, *curr_forward_,
        backward_[num_frames_decoded_ + 1], &label_posteriors_.back());
    // Swap current & previous tokens, and clean current tokens
    std::swap(prev_forward_, curr_forward_);
    curr_forward_->clear();
    ForwardProcessEmitting(decodable);
    ForwardProcessNonemitting();
  }
  return (!curr_forward_->empty());
}

void SimpleForwardBackward::ForwardProcessEmitting(
    DecodableInterface* decodable) {
  const int32 frame = num_frames_decoded_;
  ++num_frames_decoded_;
  for (TokenMap::const_iterator ptok = prev_forward_->begin();
       ptok != prev_forward_->end(); ++ptok) {
    const StateId state = ptok->first;
    for (ArcIterator aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const StdArc& arc = aiter.Value();
      // propagate only emitting symbols, and arcs to those states that
      // ware active in the backward pass in the corresponding frame
      // [i.e. alpha(s, t) * beta(s, t) > 0]
      if (arc.ilabel &&
          backward_[num_frames_decoded_].count(arc.nextstate)) {
        const double acoustic_cost =
            -decodable->LogLikelihood(frame, arc.ilabel);
        Token& ctok = curr_forward_->insert(make_pair(
            arc.nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
        ctok.UpdateEmitting(
            ptok->second.cost, arc.weight.Value(), acoustic_cost);
      }
    }
  }
}

void SimpleForwardBackward::ForwardProcessNonemitting() {
  // If the wfst has only emitting arcs, we are done
  if (!wfst_with_epsilon_arcs_) return;

  QueueSet<StateId> queue_set;
  for (TokenMap::iterator tok = curr_forward_->begin();
       tok != curr_forward_->end(); ++tok) {
    if (!state_with_epsilon_arcs_[tok->first].second) continue;
    queue_set.push(tok->first);
    tok->second.last_cost = tok->second.cost;
  }

  while (!queue_set.empty()) {
    const StateId state = queue_set.front();
    queue_set.pop();

    Token& ptok = curr_forward_->find(state)->second;
    const double last_cost = ptok.last_cost;
    ptok.last_cost = -kaldi::kLogZeroDouble;

    for (ArcIterator aiter(fst_, state); !aiter.Done();
         aiter.Next()) {
      const StdArc& arc = aiter.Value();
      if (arc.ilabel &&
          backward_[num_frames_decoded_].count(arc.nextstate)) {
        Token& ctok = curr_forward_->insert(make_pair(
            arc.nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
        if (ctok.UpdateNonEmitting(last_cost, arc.weight.Value(), delta_)) {
          queue_set.push(arc.nextstate);
        }
      }
    }
  }
}


}  // namespace kaldi
