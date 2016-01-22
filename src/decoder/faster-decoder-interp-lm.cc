// decoder/faster-decoder-interp-lm.cc

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

#include "decoder/faster-decoder-interp-lm.h"

namespace kaldi {

FasterDecoderInterpLm::FasterDecoderInterpLm(
    const fst::Fst<fst::StdArc> &hcl,
    fst::DeterministicOnDemandFst<fst::StdArc> &lm1,
    fst::DeterministicOnDemandFst<fst::StdArc> &lm2,
    const FasterDecoderInterpLmOptions &config) :
    hcl_(hcl), lm1_(lm1), lm2_(lm2), config_(config), num_frames_decoded_(-1) {
  KALDI_ASSERT(config_.hash_ratio >= 1.0);  // less doesn't make much sense.
  KALDI_ASSERT(config_.max_active > 1);
  KALDI_ASSERT(config_.min_active >= 0 && config_.min_active < config_.max_active);
  KALDI_ASSERT(config_.interp_weight >= 0.0 && config_.interp_weight <= 1.0);
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
}


void FasterDecoderInterpLm::InitDecoding() {
  // clean up from last time:
  ClearToks(toks_.Clear());
  KALDI_ASSERT(hcl_.Start() != fst::kNoStateId);
  KALDI_ASSERT(lm1_.Start() != fst::kNoStateId);
  KALDI_ASSERT(lm2_.Start() != fst::kNoStateId);
  StateTriplet start_state(hcl_.Start(), lm1_.Start(), lm2_.Start());
  const BaseFloat cost_lm1 = -log(config_.interp_weight);
  const BaseFloat cost_lm2 = -log(1.0 - config_.interp_weight);
  toks_.Insert(start_state, new Token(Arc(0, 0, Weight::One(), hcl_.Start()),
                                      Arc(0, 0, cost_lm1, lm1_.Start()),
                                      Arc(0, 0, cost_lm2, lm1_.Start()), NULL));
  ProcessNonemitting(std::numeric_limits<float>::max());
  num_frames_decoded_ = 0;
}


void FasterDecoderInterpLm::Decode(DecodableInterface *decodable) {
  InitDecoding();
  while (!decodable->IsLastFrame(num_frames_decoded_ - 1)) {
    const double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff);
  }
}

void FasterDecoderInterpLm::AdvanceDecoding(DecodableInterface *decodable,
                                            int32 max_num_frames) {
  KALDI_ASSERT(num_frames_decoded_ >= 0 &&
               "You must call InitDecoding() before AdvanceDecoding()");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= num_frames_decoded_);
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     num_frames_decoded_ + max_num_frames);
  while (num_frames_decoded_ < target_frames_decoded) {
    // note: ProcessEmitting() increments num_frames_decoded_
    const double weight_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(weight_cutoff);
  }
}


bool FasterDecoderInterpLm::ReachedFinal() {
  for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
    StateTriplet state = e->key;
    if (e->val->TotalCost() != std::numeric_limits<double>::infinity() &&
        hcl_.Final(state[0]) != Weight::Zero() &&
        FinalLmCost(state[1], state[2]) != std::numeric_limits<double>::infinity())
      return true;
  }
  return false;
}

bool FasterDecoderInterpLm::GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                                        bool use_final_probs) {
  // GetBestPath gets the decoding output.  If "use_final_probs" is true
  // AND we reached a final state, it limits itself to final states;
  // otherwise it gets the most likely token not taking into
  // account final-probs.  fst_out will be empty (Start() == kNoStateId) if
  // nothing was available.  It returns true if it got output (thus, fst_out
  // will be nonempty).
  fst_out->DeleteStates();
  Token *best_tok = NULL;
  const bool is_final = ReachedFinal();
  if (!is_final) {
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail)
      if (best_tok == NULL || *best_tok < *(e->val) )
        best_tok = e->val;
  } else {
    const double infinity =  std::numeric_limits<double>::infinity();
    double best_cost = infinity;
    for (const Elem *e = toks_.GetList(); e != NULL; e = e->tail) {
      const StateTriplet state = e->key;
      const double this_cost = e->val->TotalCost() + hcl_.Final(state[0]).Value() +
          FinalLmCost(state[1], state[2]);
      if (this_cost < best_cost && this_cost != infinity) {
        best_cost = this_cost;
        best_tok = e->val;
      }
    }
  }
  if (best_tok == NULL) return false;  // No output.

  std::vector<LatticeArc> arcs_reverse;  // arcs in reverse order.

  for (Token *tok = best_tok; tok != NULL; tok = tok->prev_) {
    const double tot_cost = tok->TotalCost() - (tok->prev_ ? tok->prev_->TotalCost() : 0.0);
    const double ac_cost = tok->cost_[3] - (tok->prev_ ? tok->prev_->cost_[3] : 0.0);
    const double graph_cost = tot_cost - ac_cost;
    LatticeArc l_arc(tok->ilabel_,
                     tok->olabel_,
                     LatticeWeight(graph_cost, ac_cost),
                     tok->state_[0]);
    arcs_reverse.push_back(l_arc);
  }
  KALDI_ASSERT(arcs_reverse.back().nextstate == hcl_.Start());
  arcs_reverse.pop_back();  // that was a "fake" token... gives no info.

  StateId cur_state = fst_out->AddState();
  fst_out->SetStart(cur_state);
  for (ssize_t i = static_cast<ssize_t>(arcs_reverse.size())-1; i >= 0; i--) {
    LatticeArc arc = arcs_reverse[i];
    arc.nextstate = fst_out->AddState();
    fst_out->AddArc(cur_state, arc);
    cur_state = arc.nextstate;
  }
  if (is_final && use_final_probs) {
    const BaseFloat final_weight = hcl_.Final(best_tok->state_[0]).Value() +
        FinalLmCost(best_tok->state_[1], best_tok->state_[2]);
    fst_out->SetFinal(cur_state, LatticeWeight(final_weight, 0.0));
  } else {
    fst_out->SetFinal(cur_state, LatticeWeight::One());
  }
  RemoveEpsLocal(fst_out);
  return true;
}


// Gets the weight cutoff.  Also counts the active tokens.
double FasterDecoderInterpLm::GetCutoff(Elem *list_head, size_t *tok_count,
                                        BaseFloat *adaptive_beam, Elem **best_elem) {
  double best_cost = std::numeric_limits<double>::infinity();
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      const double w = e->val->TotalCost();
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_cost + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      double w = e->val->TotalCost();
      tmp_array_.push_back(w);
      if (w < best_cost) {
        best_cost = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    double beam_cutoff = best_cost + config_.beam,
        min_active_cutoff = std::numeric_limits<double>::infinity(),
        max_active_cutoff = std::numeric_limits<double>::infinity();

    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_cost + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_cost;
      else {
        std::nth_element(tmp_array_.begin(),
                         tmp_array_.begin() + config_.min_active,
                         tmp_array_.size() > static_cast<size_t>(config_.max_active) ?
                         tmp_array_.begin() + config_.max_active :
                         tmp_array_.end());
        min_active_cutoff = tmp_array_[config_.min_active];
      }
    }
    if (min_active_cutoff > beam_cutoff) { // min_active is looser than beam.
      if (adaptive_beam)
        *adaptive_beam = min_active_cutoff - best_cost + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

void FasterDecoderInterpLm::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

// ProcessEmitting returns the likelihood cutoff used.
double FasterDecoderInterpLm::ProcessEmitting(DecodableInterface *decodable) {
  int32 frame = num_frames_decoded_;
  Elem *last_toks = toks_.Clear();
  size_t tok_cnt;
  BaseFloat adaptive_beam;
  Elem *best_elem = NULL;
  double weight_cutoff = GetCutoff(last_toks, &tok_cnt,
                                   &adaptive_beam, &best_elem);
  KALDI_VLOG(3) << tok_cnt << " tokens active.";
  PossiblyResizeHash(tok_cnt);  // This makes sure the hash is always big enough.

  // This is the cutoff we use after adding in the log-likes (i.e.
  // for the next frame).  This is a bound on the cutoff we will use
  // on the next frame.
  double next_weight_cutoff = std::numeric_limits<double>::infinity();

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.
  if (best_elem) {
    StateTriplet state = best_elem->key;
    Token *tok = best_elem->val;
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(hcl_, state[0]);
         !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel != 0) {  // we'd propagate..
        const BaseFloat ac_cost = - decodable->LogLikelihood(frame, arc.ilabel);
        // Find arcs through the LMs
        Arc arc_lm1, arc_lm2;
        FindLmArcs(tok->state_[1], tok->state_[2], arc.olabel, &arc_lm1, &arc_lm2);
        // We have to linearly interpolate the weights between the two LMs and add that
        // cost to the token only when a output symbol is emitted in the HCL FST.
        const double new_weight =
            (tok->cost_[0] + arc.weight.Value() + ac_cost)                 // HCL cost
            - kaldi::LogAdd(- tok->cost_[1] - arc_lm1.weight.Value(),      // Int. LM cost
                            - tok->cost_[2] - arc_lm2.weight.Value());

        if (new_weight + adaptive_beam < next_weight_cutoff)
          next_weight_cutoff = new_weight + adaptive_beam;
      }
    }
  }

  // int32 n = 0, np = 0;

  // the tokens are now owned here, in last_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call TokenDelete
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = last_toks, *e_tail; e != NULL; e = e_tail) {  // loop this way
    // n++;
    // because we delete "e" as we go.
    StateTriplet state = e->key;
    Token *tok = e->val;
    if (tok->TotalCost() < weight_cutoff) {  // not pruned.
      // np++;
      for (fst::ArcIterator<fst::Fst<Arc> > aiter(hcl_, state[0]);
           !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        if (arc.ilabel != 0) {  // propagate..
          BaseFloat ac_cost =  - decodable->LogLikelihood(frame, arc.ilabel);
          // Find arcs through the LMs
          Arc arc_lm1, arc_lm2;
          FindLmArcs(tok->state_[1], tok->state_[2], arc.olabel, &arc_lm1, &arc_lm2);
          // We have to linearly interpolate the weights between the two LMs and add that
          // cost to the token only when a output symbol is emitted in the HCL FST.
          const double new_weight =
              (tok->cost_[0] + arc.weight.Value() + ac_cost)                 // HCL cost
              - kaldi::LogAdd(- tok->cost_[1] - arc_lm1.weight.Value(),      // Int. LM cost
                              - tok->cost_[2] - arc_lm2.weight.Value());

          if (new_weight < next_weight_cutoff) {  // not pruned..
            Token *new_tok = new Token(arc, arc_lm1, arc_lm2, ac_cost, tok);
            Elem *e_found = toks_.Find(new_tok->state_);
            if (new_weight + adaptive_beam < next_weight_cutoff)
              next_weight_cutoff = new_weight + adaptive_beam;
            if (e_found == NULL) {
              toks_.Insert(new_tok->state_, new_tok);
            } else {
              if ( *(e_found->val) < *new_tok ) {
                Token::TokenDelete(e_found->val);
                e_found->val = new_tok;
              } else {
                Token::TokenDelete(new_tok);
              }
            }
          }
        }
      }
    }
    e_tail = e->tail;
    Token::TokenDelete(e->val);
    toks_.Delete(e);
  }
  num_frames_decoded_++;
  return next_weight_cutoff;
}

// TODO: first time we go through this, could avoid using the queue.
void FasterDecoderInterpLm::ProcessNonemitting(double cutoff) {
  // Processes nonemitting arcs for one frame.
  KALDI_ASSERT(queue_.empty());
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
    queue_.push_back(e->key);
  while (!queue_.empty()) {
    StateTriplet state = queue_.back();
    queue_.pop_back();
    Token *tok = toks_.Find(state)->val;  // would segfault if state not
    // in toks_ but this can't happen.
    if (tok->TotalCost() > cutoff) { // Don't bother processing successors.
      continue;
    }
    KALDI_ASSERT(tok != NULL && state == tok->state_);
    for (fst::ArcIterator<fst::Fst<Arc> > aiter(hcl_, state[0]); !aiter.Done();
         aiter.Next()) {
      const Arc &arc = aiter.Value();
      if (arc.ilabel == 0) {  // propagate nonemitting only...
        Arc arc_lm1, arc_lm2;
        FindLmArcs(tok->state_[1], tok->state_[2], arc.olabel, &arc_lm1, &arc_lm2);
        Token *new_tok = new Token(arc, arc_lm1, arc_lm2, tok);
        if (new_tok->TotalCost() > cutoff) {  // prune
          Token::TokenDelete(new_tok);
        } else {
          Elem *e_found = toks_.Find(new_tok->state_);
          if (e_found == NULL) {
            toks_.Insert(new_tok->state_, new_tok);
            queue_.push_back(new_tok->state_);
          } else {
            if ( *(e_found->val) < *new_tok ) {
              Token::TokenDelete(e_found->val);
              e_found->val = new_tok;
              queue_.push_back(new_tok->state_);
            } else {
              Token::TokenDelete(new_tok);
            }
          }
        }
      }
    }
  }
}

void FasterDecoderInterpLm::ClearToks(Elem *list) {
  for (Elem *e = list, *e_tail; e != NULL; e = e_tail) {
    Token::TokenDelete(e->val);
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

void FasterDecoderInterpLm::FindLmArcs(StateId state_lm1, StateId state_lm2, Label label,
                                       Arc *arc_lm1, Arc *arc_lm2) {
  KALDI_ASSERT(arc_lm1 != NULL & arc_lm2 != NULL);
  if (label == 0) {
    // If no symbol is emitted, we do not change the state in the LMs, which is just
    // like having a loop arc in the current state of each LM with zero cost.
    arc_lm1->ilabel = arc_lm1->olabel = arc_lm2->ilabel = arc_lm2->olabel = 0;
    arc_lm1->weight = arc_lm2->weight = Weight::One();
    arc_lm1->nextstate = state_lm1;
    arc_lm2->nextstate = state_lm2;
  } else {
    // Try to go through an arc labeled with `label', from the current state in both LM.
    // If no arc is found with `label', then go to a terminal state with infinity cost for
    // that LM.
    if (state_lm1 == fst::kNoStateId || !lm1_.GetArc(state_lm1, label, arc_lm1)) {
      arc_lm1->ilabel = arc_lm1->olabel = label;
      arc_lm1->weight = Weight::Zero();
      arc_lm1->nextstate = fst::kNoStateId;
    }
    if (state_lm2 == fst::kNoStateId || !lm2_.GetArc(state_lm2, label, arc_lm2)) {
      arc_lm2->ilabel = arc_lm2->olabel = label;
      arc_lm2->weight = Weight::Zero();
      arc_lm2->nextstate = fst::kNoStateId;
    }
  }
}

BaseFloat FasterDecoderInterpLm::FinalLmCost(StateId state_lm1, StateId state_lm2) const {
  const Weight w1 = state_lm1 == fst::kNoStateId ? Weight::Zero() : lm1_.Final(state_lm1);
  const Weight w2 = state_lm2 == fst::kNoStateId ? Weight::Zero() : lm2_.Final(state_lm2);
  return -kaldi::LogAdd(-w1.Value(), -w2.Value());
}

} // end namespace kaldi.
