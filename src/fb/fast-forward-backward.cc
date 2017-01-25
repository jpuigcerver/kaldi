// fb/fast-forward-backward.cc

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

#include "fb/fast-forward-backward.h"
#include "fb/queue-set.h"
#include <algorithm>

namespace kaldi {

FastForwardBackward::FastForwardBackward(
    const Fst& fst, double beam_fwd, double beam_bkw, double delta) :
    fst_(fst), prev_forward_(new TokenMap), curr_forward_(new TokenMap),
    num_frames_decoded_(0), beam_bkw_(beam_bkw), beam_fwd_(beam_fwd),
    delta_(delta), likelihood_(kaldi::kLogZeroDouble) {
}

FastForwardBackward::~FastForwardBackward() {
  delete prev_forward_;
  delete curr_forward_;
}

void FastForwardBackward::Initialize(DecodableInterface* decodable) {
  backward_.clear();
  prev_forward_->clear();
  curr_forward_->clear();
  // We need this asserts because the backward pass cannot be done in a
  // sequence processed online, that is, we need the whole sequence to be
  // available from the beginning
  KALDI_ASSERT(decodable->NumFramesReady() >= 0);
  KALDI_ASSERT(decodable->IsLastFrame(decodable->NumFramesReady() - 1));
}

bool FastForwardBackward::ForwardBackward(DecodableInterface *decodable) {
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

bool FastForwardBackward::Backward(DecodableInterface *decodable) {
  // Initialize backward trellis with the final states
  backward_.push_back(TokenMap());
  for(StateIterator siter(fst_); !siter.Done(); siter.Next()) {
    const FBArc::Weight& final = fst_.Final(siter.Value());
    if (final != FBArc::Weight::Zero()) {
      backward_[0].insert(make_pair(siter.Value(), final.Value()));
    }
  }
  // Process nonemitting arcs to reach all states with epsilon-transition
  // to any of the final states
  ProcessNonemitting<true>(&backward_[0]);
  PruneToks(beam_bkw_, &backward_[0]);
  // Process all input frames ...
  for (num_frames_decoded_ = 0;
       num_frames_decoded_ < decodable->NumFramesReady();
       ++num_frames_decoded_) {
    backward_.push_back(TokenMap());
    ProcessEmitting<true>(
        decodable, decodable->NumFramesReady() - num_frames_decoded_ - 1,
        backward_[num_frames_decoded_],  &backward_.back());
    ProcessNonemitting<true>(&backward_.back());
    PruneToks(beam_bkw_, &backward_.back());
  }
  // Reverse backward matrix [T, T-1, ..., 1] -> [1, ..., T-1, T]
  std::reverse(backward_.begin(), backward_.end());
  return (!backward_.front().empty());
}

///////////////////////////////////////////////////////////////////////////
/// FORWARD PASS METHODS ...
///////////////////////////////////////////////////////////////////////////

bool FastForwardBackward::Forward(DecodableInterface* decodable) {
  // Initialize forward trellis with the initial state
  StateId start_state = fst_.Start();
  if (start_state != fst::kNoStateId) {
    curr_forward_->insert(make_pair(start_state, 0.0));
  }
  // Process nonemitting arcs to reach all states with epsilon-transition
  // from the start state
  ProcessNonemitting<false>(curr_forward_);
  // Compute total likelihood using forward_[0] and backward_[0].
  // Note: Remember that we have backward_[0] because we did the full
  // backward pass before.
  likelihood_ = ComputeLikelihood(*curr_forward_, backward_[0]);
  PruneToksForwardBackward(
      -likelihood_, beam_fwd_, curr_forward_, &backward_[0]);
  // Process all input frames ...
  for (num_frames_decoded_ = 0;
       num_frames_decoded_ < decodable->NumFramesReady();
       ++num_frames_decoded_) {
    // Compute label posteriors at time t.
    label_posteriors_.push_back(LabelMap());
    ComputeLabelsPosteriorAtTimeT(
        fst_, decodable, num_frames_decoded_, *curr_forward_,
        backward_[num_frames_decoded_ + 1], &label_posteriors_.back());
    // Swap current & previous tokens, and clean current tokens
    std::swap(prev_forward_, curr_forward_);
    curr_forward_->clear();
    ProcessEmitting<false>(
        decodable, num_frames_decoded_, *prev_forward_, curr_forward_);
    ProcessNonemitting<false>(curr_forward_);
    PruneToksForwardBackward(
        -likelihood_, beam_fwd_, curr_forward_,
        &backward_[num_frames_decoded_]);
  }
  return (!curr_forward_->empty());
}

///////////////////////////////////////////////////////////////////////////
/// GENERIC FORWARD/BACKWARD PASS METHODS ...
///////////////////////////////////////////////////////////////////////////

template <bool backward>
void FastForwardBackward::ProcessEmitting(
    DecodableInterface* decodable, int32_t frame,
    const TokenMap& prev_toks, TokenMap* curr_toks) {
  for (TokenMap::const_iterator ptok = prev_toks.begin();
       ptok != prev_toks.end(); ++ptok) {
    const StateId state = ptok->first;
    for (ArcIterator aiter(fst_, state, backward); !aiter.Done();
         aiter.Next()) {
      const FBArc& arc = aiter.Value();
      const StateId nextstate = backward ? arc.prevstate : arc.nextstate;
      // propagate only emitting symbols, and arcs to those states that
      // were active in the backward pass in the corresponding frame
      // [i.e. alpha(s, t) * beta(s, t) > 0]
      if (arc.ilabel) {
        const double acoustic_cost =
            -decodable->LogLikelihood(frame, arc.ilabel);
        Token& ctok = curr_toks->insert(make_pair(
            nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
        ctok.UpdateEmitting(
            ptok->second.cost, arc.weight.Value(), acoustic_cost);
      }
    }
  }
}

template <bool backward>
void FastForwardBackward::ProcessNonemitting(TokenMap* curr_toks) {
  // If the wfst has no input epsilons, we are done
  if (!fst_.Properties(fst::kIEpsilons, false)) return;

  QueueSet<StateId> queue_set;
  for (TokenMap::iterator tok = curr_toks->begin(); tok != curr_toks->end();
       ++tok) {
    if (fst_.NumInputEpsilons<backward>(tok->first)) {
      queue_set.push(tok->first);
      tok->second.last_cost = tok->second.cost;
    }
  }

  while (!queue_set.empty()) {
    const StateId state = queue_set.front();
    queue_set.pop();

    Token& ptok = curr_toks->find(state)->second;
    const double last_cost = ptok.last_cost;
    ptok.last_cost = -kaldi::kLogZeroDouble;

    for (ArcIterator aiter(fst_, state, backward); !aiter.Done();
         aiter.Next()) {
      const FBArc& arc = aiter.Value();
      const StateId nextstate = backward ? arc.prevstate : arc.nextstate;
      if (!arc.ilabel) {
        Token& ctok = curr_toks->insert(make_pair(
            nextstate, Token(-kaldi::kLogZeroDouble))).first->second;
        if (ctok.UpdateNonEmitting(last_cost, arc.weight.Value(), delta_)) {
          queue_set.push(nextstate);
        }
      }
    }
  }
}


}  // namespace kaldi
