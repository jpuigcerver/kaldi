// decoder/lattice-faster-decoder.cc

// Copyright 2009-2012  Microsoft Corporation  Mirko Hannemann
//           2013-2014  Johns Hopkins University (Author: Daniel Povey)
//                2014  Guoguo Chen

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

// Note on svn: this file is "upstream" from lattice-faster-online-decoder.cc, and
// changes in this file should be merged into lattice-faster-online-decoder.cc,
// after committing the changes to this file, using the command
// svn merge ^/sandbox/online/src/decoder/lattice-faster-decoder.cc lattice-faster-online-decoder.cc

#include "decoder/lattice-faster-interplm-decoder.h"
#include "lat/lattice-functions.h"

namespace kaldi {

// instantiate this class once for each thing you have to decode.
LatticeFasterInterpLmDecoder::LatticeFasterInterpLmDecoder(
    const LatticeFasterInterpLmDecoderConfig &config,
    const Fst& hcl, const Fst& lm1, const Fst& lm2):
    hcl_(hcl), lm1_(lm1), lm2_(lm2),
    matcher1_(lm1_, fst::MATCH_INPUT), matcher2_(lm2_, fst::MATCH_INPUT),
    delete_fst_(false), config_(config), num_toks_(0) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  KALDI_ASSERT(lm1_.Properties(fst::kILabelSorted, true) == fst::kILabelSorted);
  KALDI_ASSERT(lm2_.Properties(fst::kILabelSorted, true) == fst::kILabelSorted);
}

LatticeFasterInterpLmDecoder::LatticeFasterInterpLmDecoder(
    const LatticeFasterInterpLmDecoderConfig &config,
    const Fst* hcl, const Fst* lm1, const Fst* lm2) :
    hcl_(*hcl), lm1_(*lm1), lm2_(*lm2),
    matcher1_(lm1_, fst::MATCH_INPUT), matcher2_(lm2_, fst::MATCH_INPUT),
    delete_fst_(true), config_(config), num_toks_(0) {
  config.Check();
  toks_.SetSize(1000);  // just so on the first frame we do something reasonable.
  KALDI_ASSERT(lm1_.Properties(fst::kILabelSorted, true) == fst::kILabelSorted);
  KALDI_ASSERT(lm2_.Properties(fst::kILabelSorted, true) == fst::kILabelSorted);
}


LatticeFasterInterpLmDecoder::~LatticeFasterInterpLmDecoder() {
  DeleteElems(toks_.Clear());
  ClearActiveTokens();
  if (delete_fst_) {
    delete &(hcl_);
    delete &(lm1_);
    delete &(lm2_);
  }
}

void LatticeFasterInterpLmDecoder::InitDecoding() {
  // clean up from last time:
  DeleteElems(toks_.Clear());
  cost_offsets_.clear();
  ClearActiveTokens();
  warned_ = false;
  num_toks_ = 0;
  decoding_finalized_ = false;
  final_costs_.clear();
  KALDI_ASSERT(hcl_.Start() != fst::kNoStateId);
  KALDI_ASSERT(lm1_.Start() != fst::kNoStateId);
  KALDI_ASSERT(lm2_.Start() != fst::kNoStateId);
  StateTriplet start_state(hcl_.Start(), lm1_.Start(), lm2_.Start());
  active_toks_.resize(1);

  Token *start_tok = new Token(0.0, 0.0,
                               -log(config_.alpha), -log(1.0 - config_.alpha),
                               0.0, NULL, NULL);
  active_toks_[0].toks = start_tok;
  toks_.Insert(start_state, start_tok);
  num_toks_++;
  ProcessNonemitting(config_.beam);
}

// Returns true if any kind of traceback is available (not necessarily from
// a final state).  It should only very rarely return false; this indicates
// an unusual search error.
bool LatticeFasterInterpLmDecoder::Decode(DecodableInterface *decodable) {
  InitDecoding();

  // We use 1-based indexing for frames in this decoder (if you view it in
  // terms of features), but note that the decodable object uses zero-based
  // numbering, which we have to correct for when we call it.
  while (!decodable->IsLastFrame(NumFramesDecoded() - 1)) {
    if (NumFramesDecoded() % config_.prune_interval == 0)
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    const BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
  FinalizeDecoding();

  // Returns true if we have any kind of traceback available (not necessarily
  // to the end state; query ReachedFinal() for that).
  return !active_toks_.empty() && active_toks_.back().toks != NULL;
}


// Outputs an FST corresponding to the single best path through the lattice.
bool LatticeFasterInterpLmDecoder::GetBestPath(Lattice *olat,
                                               bool use_final_probs) const {
  Lattice raw_lat;
  GetRawLattice(&raw_lat, use_final_probs);
  ShortestPath(raw_lat, olat);
  return (olat->NumStates() != 0);
}

// Outputs an FST corresponding to the raw, state-level
// tracebacks.
bool LatticeFasterInterpLmDecoder::GetRawLattice(Lattice *ofst,
                                                 bool use_final_probs) const {
  typedef LatticeArc Arc;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;
  typedef Arc::Label Label;

  // Note: you can't use the old interface (Decode()) if you want to
  // get the lattice with use_final_probs = false.  You'd have to do
  // InitDecoding() and then AdvanceDecoding().
  if (decoding_finalized_ && !use_final_probs)
    KALDI_ERR << "You cannot call FinalizeDecoding() and then call "
              << "GetRawLattice() with use_final_probs == false";

  unordered_map<Token*, BaseFloat> final_costs_local;

  const unordered_map<Token*, BaseFloat> &final_costs =
      (decoding_finalized_ ? final_costs_ : final_costs_local);
  if (!decoding_finalized_ && use_final_probs)
    ComputeFinalCosts(&final_costs_local, NULL, NULL);

  ofst->DeleteStates();
  // num-frames plus one (since frames are one-based, and we have
  // an extra frame for the start-state).
  int32 num_frames = active_toks_.size() - 1;
  KALDI_ASSERT(num_frames > 0);
  const int32 bucket_count = num_toks_/2 + 3;
  unordered_map<Token*, StateId> tok_map(bucket_count);
  // First create all states.
  std::vector<Token*> token_list;
  for (int32 f = 0; f <= num_frames; f++) {
    if (active_toks_[f].toks == NULL) {
      KALDI_WARN << "GetRawLattice: no tokens active on frame " << f
                 << ": not producing lattice.\n";
      return false;
    }
    TopSortTokens(active_toks_[f].toks, &token_list);
    for (size_t i = 0; i < token_list.size(); i++)
      if (token_list[i] != NULL)
        tok_map[token_list[i]] = ofst->AddState();
  }
  // The next statement sets the start state of the output FST.  Because we
  // topologically sorted the tokens, state zero must be the start-state.
  ofst->SetStart(0);

  KALDI_VLOG(4) << "init:" << num_toks_/2 + 3 << " buckets:"
                << tok_map.bucket_count() << " load:" << tok_map.load_factor()
                << " max:" << tok_map.max_load_factor();
  // Now create all arcs.
  for (int32 f = 0; f <= num_frames; f++) {
    for (Token *tok = active_toks_[f].toks; tok != NULL; tok = tok->next) {
      StateId cur_state = tok_map[tok];
      for (ForwardLink *l = tok->links; l != NULL; l = l->next) {
        auto iter = tok_map.find(l->next_tok);
        StateId nextstate = iter->second;
        KALDI_ASSERT(iter != tok_map.end());
        BaseFloat cost_offset = 0.0;
        if (l->ilabel != 0) {  // emitting..
          KALDI_ASSERT(f >= 0 && f < cost_offsets_.size());
          cost_offset = cost_offsets_[f];
        }
        // LM cost of ForwardLink =
        // Acumulated cost of the LM through the ForwardLink -
        // Acumulated cost of the LM through the source of the ForwardLink (tok)
        const BaseFloat lm_cost =
            (-kaldi::LogAdd(-(tok->lm1_cost + l->lm1_cost),
                            -(tok->lm2_cost + l->lm2_cost))) - tok->LMCost();
        // Graph cost = HCL cost of ForwardLink + LM cost of ForwardLink
        Arc arc(l->ilabel, l->olabel,
                LatticeWeight(l->hcl_cost + lm_cost, l->aco_cost - cost_offset),
                nextstate);
        ofst->AddArc(cur_state, arc);
      }
      if (f == num_frames) {
        if (use_final_probs && !final_costs.empty()) {
          unordered_map<Token*, BaseFloat>::const_iterator iter =
              final_costs.find(tok);
          if (iter != final_costs.end())
            ofst->SetFinal(cur_state, LatticeWeight(iter->second, 0));
        } else {
          ofst->SetFinal(cur_state, LatticeWeight::One());
        }
      }
    }
  }
  return (ofst->NumStates() > 0);
}


// This function is now deprecated, since now we do determinization from outside
// the LatticeFasterInterpLmDecoder class.  Outputs an FST corresponding to the
// lattice-determinized lattice (one path per word sequence).
bool LatticeFasterInterpLmDecoder::GetLattice(CompactLattice *ofst,
                                              bool use_final_probs) const {
  Lattice raw_fst;
  GetRawLattice(&raw_fst, use_final_probs);
  Invert(&raw_fst);  // make it so word labels are on the input.
  // (in phase where we get backward-costs).
  fst::ILabelCompare<LatticeArc> ilabel_comp;
  ArcSort(&raw_fst, ilabel_comp);  // sort on ilabel; makes
  // lattice-determinization more efficient.

  fst::DeterminizeLatticePrunedOptions lat_opts;
  lat_opts.max_mem = config_.det_opts.max_mem;

  DeterminizeLatticePruned(raw_fst, config_.lattice_beam, ofst, lat_opts);
  raw_fst.DeleteStates();  // Free memory-- raw_fst no longer needed.
  Connect(ofst);  // Remove unreachable states... there might be
  // a small number of these, in some cases.
  // Note: if something went wrong and the raw lattice was empty,
  // we should still get to this point in the code without warnings or failures.
  return (ofst->NumStates() != 0);
}

void LatticeFasterInterpLmDecoder::PossiblyResizeHash(size_t num_toks) {
  size_t new_sz = static_cast<size_t>(static_cast<BaseFloat>(num_toks)
                                      * config_.hash_ratio);
  if (new_sz > toks_.Size()) {
    toks_.SetSize(new_sz);
  }
}

// FindOrAddToken either locates a token in hash of toks_,
// or if necessary inserts a new, empty token (i.e. with no forward links)
// for the current frame.  [note: it's inserted if necessary into hash toks_
// and also into the singly linked list of tokens active on this frame
// (whose head is at active_toks_[frame]).
LatticeFasterInterpLmDecoder::Token*
LatticeFasterInterpLmDecoder::FindOrAddToken(
    StateTriplet state, int32 frame_plus_one, BaseFloat tot_cost,
    BaseFloat hcl_cost, BaseFloat lm1_cost, BaseFloat lm2_cost, bool *changed) {
  // Returns the Token pointer.  Sets "changed" (if non-NULL) to true
  // if the token was newly created or the cost changed.
  KALDI_ASSERT(frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  Elem *e_found = toks_.Find(state);
  if (e_found == NULL) {  // no such token presently.
    const BaseFloat extra_cost = 0.0;
    // tokens on the currently final frame have zero extra_cost
    // as any of them could end up
    // on the winning path.
    Token *new_tok = new Token (tot_cost, hcl_cost, lm1_cost, lm2_cost,
                                extra_cost, NULL, toks);
    // NULL: no forward links yet
    toks = new_tok;
    num_toks_++;
    toks_.Insert(state, new_tok);
    if (changed) *changed = true;
    return new_tok;
  } else {
    Token *tok = e_found->val;  // There is an existing Token for this state.
    if (tok->tot_cost > tot_cost) {  // replace old token
      tok->tot_cost = tot_cost;
      tok->hcl_cost = hcl_cost;
      tok->lm1_cost = lm1_cost;
      tok->lm2_cost = lm2_cost;
      // we don't allocate a new token, the old stays linked in active_toks_
      // we only replace the tot_cost
      // in the current frame, there are no forward links (and no extra_cost)
      // only in ProcessNonemitting we have to delete forward links
      // in case we visit a state for the second time
      // those forward links, that lead to this replaced token before:
      // they remain and will hopefully be pruned later (PruneForwardLinks...)
      if (changed) *changed = true;
    } else {
      if (changed) *changed = false;
    }
    return tok;
  }
}

// prunes outgoing links for all tokens in active_toks_[frame]
// it's called by PruneActiveTokens all links, that have
// link_extra_cost > lattice_beam are pruned
void LatticeFasterInterpLmDecoder::PruneForwardLinks(
    int32 frame_plus_one, bool *extra_costs_changed, bool *links_pruned,
    BaseFloat delta) {
  // delta is the amount by which the extra_costs must change
  // If delta is larger,  we'll tend to go back less far
  //    toward the beginning of the file.
  // extra_costs_changed is set to true if extra_cost was changed for any token
  // links_pruned is set to true if any link in any token was pruned

  *extra_costs_changed = false;
  *links_pruned = false;
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  if (active_toks_[frame_plus_one].toks == NULL) {  // empty list; should not happen.
    if (!warned_) {
      KALDI_WARN << "No tokens alive [doing pruning].. warning first "
          "time only for each utterance\n";
      warned_ = true;
    }
  }

  // We have to iterate until there is no more change, because the links
  // are not guaranteed to be in topological order.
  bool changed = true;  // difference new minus old extra cost >= delta ?
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost for tok.
      BaseFloat tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // tok_extra_cost is the best (min) of link_extra_cost of outgoing links
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        const BaseFloat link_cost =
            tok->AcousticCost() + link->aco_cost +             // Acoustic
            tok->hcl_cost + link->hcl_cost -                   // HCL
            kaldi::LogAdd(-(tok->lm1_cost + link->lm1_cost),   // Interp LM
                          -(tok->lm2_cost + link->lm2_cost));
        // difference in brackets is >= 0
        // link_exta_cost is the difference in score between the best paths
        // through link source state and through link destination state
        BaseFloat link_extra_cost =
            next_tok->extra_cost + (link_cost - next_tok->tot_cost);
        KALDI_ASSERT(link_extra_cost == link_extra_cost);  // check for NaN
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link;  // advance link but leave prev_link the same.
          *links_pruned = true;
        } else {   // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) {  // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;  // move to next link
          link = link->next;
        }
      }  // for all outgoing links
      if (fabs(tok_extra_cost - tok->extra_cost) > delta)
        changed = true;   // difference new minus old is bigger than delta
      tok->extra_cost = tok_extra_cost;
      // will be +infinity or <= lattice_beam_.
      // infinity indicates, that no forward link survived pruning
    }  // for all Token on active_toks_[frame]
    if (changed) *extra_costs_changed = true;

    // Note: it's theoretically possible that aggressive compiler
    // optimizations could cause an infinite loop here for small delta and
    // high-dynamic-range scores.
  } // while changed
}

// PruneForwardLinksFinal is a version of PruneForwardLinks that we call
// on the final frame.  If there are final tokens active, it uses
// the final-probs for pruning, otherwise it treats all tokens as final.
void LatticeFasterInterpLmDecoder::PruneForwardLinksFinal() {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame_plus_one = active_toks_.size() - 1;

  if (active_toks_[frame_plus_one].toks == NULL)  // empty list; should not happen.
    KALDI_WARN << "No tokens alive at end of file";

  typedef unordered_map<Token*, BaseFloat>::const_iterator IterType;
  ComputeFinalCosts(&final_costs_, &final_relative_cost_, &final_best_cost_);
  decoding_finalized_ = true;
  // We call DeleteElems() as a nicety, not because it's really necessary;
  // otherwise there would be a time, after calling PruneTokensForFrame() on the
  // final frame, when toks_.GetList() or toks_.Clear() would contain pointers
  // to nonexistent tokens.
  DeleteElems(toks_.Clear());

  // Now go through tokens on this frame, pruning forward links...  may have to
  // iterate a few times until there is no more change, because the list is not
  // in topological order.  This is a modified version of the code in
  // PruneForwardLinks, but here we also take account of the final-probs.
  bool changed = true;
  const BaseFloat delta = 1.0e-05;
  while (changed) {
    changed = false;
    for (Token *tok = active_toks_[frame_plus_one].toks;
         tok != NULL; tok = tok->next) {
      ForwardLink *link, *prev_link = NULL;
      // will recompute tok_extra_cost.  It has a term in it that corresponds
      // to the "final-prob", so instead of initializing tok_extra_cost to
      // infinity below we set it to the difference between the
      // (score+final_prob) of this token, and the best such (score+final_prob).
      BaseFloat final_cost;
      if (final_costs_.empty()) {
        final_cost = 0.0;
      } else {
        IterType iter = final_costs_.find(tok);
        if (iter != final_costs_.end())
          final_cost = iter->second;
        else
          final_cost = std::numeric_limits<BaseFloat>::infinity();
      }
      BaseFloat tok_extra_cost = tok->tot_cost + final_cost - final_best_cost_;
      // tok_extra_cost will be a "min" over either directly being final, or
      // being indirectly final through other links, and the loop below may
      // decrease its value:
      for (link = tok->links; link != NULL; ) {
        // See if we need to excise this link...
        Token *next_tok = link->next_tok;
        const BaseFloat link_cost =
            tok->AcousticCost() + link->aco_cost +             // Acoustic
            tok->hcl_cost + link->hcl_cost -                   // HCL
            kaldi::LogAdd(-(tok->lm1_cost + link->lm1_cost),   // Interp LM
                          -(tok->lm2_cost + link->lm2_cost));
        BaseFloat link_extra_cost =
            next_tok->extra_cost + (link_cost - next_tok->tot_cost);
        if (link_extra_cost > config_.lattice_beam) {  // excise link
          ForwardLink *next_link = link->next;
          if (prev_link != NULL) prev_link->next = next_link;
          else tok->links = next_link;
          delete link;
          link = next_link; // advance link but leave prev_link the same.
        } else { // keep the link and update the tok_extra_cost if needed.
          if (link_extra_cost < 0.0) { // this is just a precaution.
            if (link_extra_cost < -0.01)
              KALDI_WARN << "Negative extra_cost: " << link_extra_cost;
            link_extra_cost = 0.0;
          }
          if (link_extra_cost < tok_extra_cost)
            tok_extra_cost = link_extra_cost;
          prev_link = link;
          link = link->next;
        }
      }
      // prune away tokens worse than lattice_beam above best path.  This step
      // was not necessary in the non-final case because then, this case
      // showed up as having no forward links.  Here, the tok_extra_cost has
      // an extra component relating to the final-prob.
      if (tok_extra_cost > config_.lattice_beam)
        tok_extra_cost = std::numeric_limits<BaseFloat>::infinity();
      // to be pruned in PruneTokensForFrame

      if (!ApproxEqual(tok->extra_cost, tok_extra_cost, delta))
        changed = true;
      tok->extra_cost = tok_extra_cost;  // will be +inf or <= lattice_beam_.
      KALDI_ASSERT(tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()
                   || tok->extra_cost < config_.lattice_beam);
    }
  } // while changed
}

BaseFloat LatticeFasterInterpLmDecoder::FinalRelativeCost() const {
  if (!decoding_finalized_) {
    BaseFloat relative_cost;
    ComputeFinalCosts(NULL, &relative_cost, NULL);
    return relative_cost;
  } else {
    // we're not allowed to call that function if FinalizeDecoding() has
    // been called; return a cached value.
    return final_relative_cost_;
  }
}


// Prune away any tokens on this frame that have no forward links.
// [we don't do this in PruneForwardLinks because it would give us
// a problem with dangling pointers].
// It's called by PruneActiveTokens if any forward links have been pruned
void LatticeFasterInterpLmDecoder::PruneTokensForFrame(int32 frame_plus_one) {
  KALDI_ASSERT(frame_plus_one >= 0 && frame_plus_one < active_toks_.size());
  Token *&toks = active_toks_[frame_plus_one].toks;
  if (toks == NULL)
    KALDI_WARN << "No tokens alive [doing pruning]";
  Token *tok, *next_tok, *prev_tok = NULL;
  for (tok = toks; tok != NULL; tok = next_tok) {
    next_tok = tok->next;
    if (tok->extra_cost == std::numeric_limits<BaseFloat>::infinity()) {
      // token is unreachable from end of graph; (no forward links survived)
      // excise tok from list and delete tok.
      if (prev_tok != NULL) prev_tok->next = tok->next;
      else toks = tok->next;
      delete tok;
      num_toks_--;
    } else {  // fetch next Token
      prev_tok = tok;
    }
  }
}

// Go backwards through still-alive tokens, pruning them, starting not from
// the current frame (where we want to keep all tokens) but from the frame before
// that.  We go backwards through the frames and stop when we reach a point
// where the delta-costs are not changing (and the delta controls when we consider
// a cost to have "not changed").
void LatticeFasterInterpLmDecoder::PruneActiveTokens(BaseFloat delta) {
  int32 cur_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // The index "f" below represents a "frame plus one", i.e. you'd have to subtract
  // one to get the corresponding index for the decodable object.
  for (int32 f = cur_frame_plus_one - 1; f >= 0; f--) {
    // Reason why we need to prune forward links in this situation:
    // (1) we have never pruned them (new TokenList)
    // (2) we have not yet pruned the forward links to the next f,
    // after any of those tokens have changed their extra_cost.
    if (active_toks_[f].must_prune_forward_links) {
      bool extra_costs_changed = false, links_pruned = false;
      PruneForwardLinks(f, &extra_costs_changed, &links_pruned, delta);
      if (extra_costs_changed && f > 0) // any token has changed extra_cost
        active_toks_[f-1].must_prune_forward_links = true;
      if (links_pruned) // any link was pruned
        active_toks_[f].must_prune_tokens = true;
      active_toks_[f].must_prune_forward_links = false; // job done
    }
    if (f+1 < cur_frame_plus_one &&      // except for last f (no forward links)
        active_toks_[f+1].must_prune_tokens) {
      PruneTokensForFrame(f+1);
      active_toks_[f+1].must_prune_tokens = false;
    }
  }
  KALDI_VLOG(4) << "PruneActiveTokens: pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

void LatticeFasterInterpLmDecoder::ComputeFinalCosts(
    unordered_map<Token*, BaseFloat> *final_costs,
    BaseFloat *final_relative_cost, BaseFloat *final_best_cost) const {
  KALDI_ASSERT(!decoding_finalized_);
  if (final_costs != NULL)
    final_costs->clear();
  const Elem *final_toks = toks_.GetList();
  const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
  BaseFloat best_cost = infinity, best_cost_with_final = infinity;
  while (final_toks != NULL) {
    StateTriplet state = final_toks->key;
    Token *tok = final_toks->val;
    const Elem *next = final_toks->tail;
    // Initialize total cost with final and final cost to infinity, since
    // this is the cost for all non final states
    BaseFloat tot_cost_with_final = infinity, final_cost = infinity;
    // Is the state in the HCL model final?
    const bool is_final_hcl = hcl_.Final(state[0]) != Weight::Zero();
    // Is the state in the LM1 model final?
    const bool is_final_lm1 =
        (state[1] != fst::kNoStateId) && (lm1_.Final(state[1]) != Weight::Zero());
    // Is the state in the LM2 model final?
    const bool is_final_lm2 =
        (state[2] != fst::kNoStateId) && (lm2_.Final(state[2]) != Weight::Zero());
    // Is the state in the INTERPOLATED LM final?
    // NOTES:
    // If the states in both LMs are final, then obviously yes.
    // If there is only a path through one of the LM (the state of the other
    // is kNoStateId), then the state if final too.
    // However, if the two states are valid, but only one is final, then the
    // state cannot be final, otherwise we would be duplicating some paths.
    // The score of the 1-best path would not change, but the lattice would.
    const bool is_final_lm_interp =
        (is_final_lm1 && is_final_lm2) ||
        (is_final_lm1 && state[2] == fst::kNoStateId) ||
        (is_final_lm2 && state[1] == fst::kNoStateId);
    if (is_final_hcl && is_final_lm_interp) {
      // Acumulated cost through HCL, including final
      const BaseFloat tot_cost_nolm_with_final =
          (tok->tot_cost - tok->LMCost()) + hcl_.Final(state[0]).Value();
      // Acumulated cost through LM1, including final
      const BaseFloat lm1_cost_with_final =
          state[1] == fst::kNoStateId ? infinity :
          tok->lm1_cost + lm1_.Final(state[1]).Value();
      // Acumulated cost through LM2, including final
      const BaseFloat lm2_cost_with_final =
          state[2] == fst::kNoStateId ? infinity :
          tok->lm2_cost + lm2_.Final(state[2]).Value();
      // Total acumulated cost, including final
      tot_cost_with_final = tot_cost_nolm_with_final
          - kaldi::LogAdd(-lm1_cost_with_final, -lm2_cost_with_final);
      // Final weight for the token
      final_cost = tot_cost_with_final - tok->tot_cost;
    }
    best_cost_with_final = std::min(tot_cost_with_final, best_cost_with_final);
    if (final_costs != NULL && final_cost != infinity)
      (*final_costs)[tok] = final_cost;
    best_cost = std::min(tok->tot_cost, best_cost);
    final_toks = next;
  }
  if (final_relative_cost != NULL) {
    if (best_cost == infinity && best_cost_with_final == infinity) {
      // Likely this will only happen if there are no tokens surviving.
      // This seems the least bad way to handle it.
      *final_relative_cost = infinity;
    } else {
      *final_relative_cost = best_cost_with_final - best_cost;
    }
  }
  if (final_best_cost != NULL) {
    if (best_cost_with_final != infinity) { // final-state exists.
      *final_best_cost = best_cost_with_final;
    } else { // no final-state exists.
      *final_best_cost = best_cost;
    }
  }
}

void LatticeFasterInterpLmDecoder::AdvanceDecoding(
    DecodableInterface *decodable, int32 max_num_frames) {
  KALDI_ASSERT(!active_toks_.empty() && !decoding_finalized_ &&
               "You must call InitDecoding() before AdvanceDecoding");
  int32 num_frames_ready = decodable->NumFramesReady();
  // num_frames_ready must be >= num_frames_decoded, or else
  // the number of frames ready must have decreased (which doesn't
  // make sense) or the decodable object changed between calls
  // (which isn't allowed).
  KALDI_ASSERT(num_frames_ready >= NumFramesDecoded());
  int32 target_frames_decoded = num_frames_ready;
  if (max_num_frames >= 0)
    target_frames_decoded = std::min(target_frames_decoded,
                                     NumFramesDecoded() + max_num_frames);
  while (NumFramesDecoded() < target_frames_decoded) {
    if (NumFramesDecoded() % config_.prune_interval == 0) {
      PruneActiveTokens(config_.lattice_beam * config_.prune_scale);
    }
    BaseFloat cost_cutoff = ProcessEmitting(decodable);
    ProcessNonemitting(cost_cutoff);
  }
}

// FinalizeDecoding() is a version of PruneActiveTokens that we call
// (optionally) on the final frame.  Takes into account the final-prob of
// tokens.  This function used to be called PruneActiveTokensFinal().
void LatticeFasterInterpLmDecoder::FinalizeDecoding() {
  int32 final_frame_plus_one = NumFramesDecoded();
  int32 num_toks_begin = num_toks_;
  // PruneForwardLinksFinal() prunes final frame (with final-probs), and
  // sets decoding_finalized_.
  PruneForwardLinksFinal();
  for (int32 f = final_frame_plus_one - 1; f >= 0; f--) {
    bool b1, b2; // values not used.
    BaseFloat dontcare = 0.0; // delta of zero means we must always update
    PruneForwardLinks(f, &b1, &b2, dontcare);
    PruneTokensForFrame(f + 1);
  }
  PruneTokensForFrame(0);
  KALDI_VLOG(4) << "pruned tokens from " << num_toks_begin
                << " to " << num_toks_;
}

/// Gets the weight cutoff.  Also counts the active tokens.
BaseFloat LatticeFasterInterpLmDecoder::GetCutoff(
    Elem *list_head, size_t *tok_count, BaseFloat *adaptive_beam,
    Elem **best_elem) {
  BaseFloat best_weight = std::numeric_limits<BaseFloat>::infinity();
  // positive == high cost == bad.
  size_t count = 0;
  if (config_.max_active == std::numeric_limits<int32>::max() &&
      config_.min_active == 0) {
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = static_cast<BaseFloat>(e->val->tot_cost);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;
    if (adaptive_beam != NULL) *adaptive_beam = config_.beam;
    return best_weight + config_.beam;
  } else {
    tmp_array_.clear();
    for (Elem *e = list_head; e != NULL; e = e->tail, count++) {
      BaseFloat w = e->val->tot_cost;
      tmp_array_.push_back(w);
      if (w < best_weight) {
        best_weight = w;
        if (best_elem) *best_elem = e;
      }
    }
    if (tok_count != NULL) *tok_count = count;

    BaseFloat beam_cutoff = best_weight + config_.beam,
        min_active_cutoff = std::numeric_limits<BaseFloat>::infinity(),
        max_active_cutoff = std::numeric_limits<BaseFloat>::infinity();

    KALDI_VLOG(6) << "Number of tokens active on frame " << NumFramesDecoded()
                  << " is " << tmp_array_.size();

    if (tmp_array_.size() > static_cast<size_t>(config_.max_active)) {
      std::nth_element(tmp_array_.begin(),
                       tmp_array_.begin() + config_.max_active,
                       tmp_array_.end());
      max_active_cutoff = tmp_array_[config_.max_active];
    }
    if (max_active_cutoff < beam_cutoff) { // max_active is tighter than beam.
      if (adaptive_beam)
        *adaptive_beam = max_active_cutoff - best_weight + config_.beam_delta;
      return max_active_cutoff;
    }
    if (tmp_array_.size() > static_cast<size_t>(config_.min_active)) {
      if (config_.min_active == 0) min_active_cutoff = best_weight;
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
        *adaptive_beam = min_active_cutoff - best_weight + config_.beam_delta;
      return min_active_cutoff;
    } else {
      *adaptive_beam = config_.beam;
      return beam_cutoff;
    }
  }
}

BaseFloat LatticeFasterInterpLmDecoder::ProcessEmitting(
    DecodableInterface *decodable) {
  KALDI_ASSERT(active_toks_.size() > 0);
  const int32 frame = active_toks_.size() - 1; // frame is the frame-index
                                               // (zero-based) used to get likelihoods
                                               // from the decodable object.
  active_toks_.resize(active_toks_.size() + 1);

  Elem *final_toks = toks_.Clear();
  Elem *best_elem = NULL;
  BaseFloat adaptive_beam;
  size_t tok_cnt;
  const BaseFloat cur_cutoff = GetCutoff(final_toks, &tok_cnt, &adaptive_beam,
                                         &best_elem);
  KALDI_VLOG(6) << "Adaptive beam on frame " << NumFramesDecoded() << " is "
                << adaptive_beam << ", cutoff is " << cur_cutoff;
  PossiblyResizeHash(tok_cnt);

  BaseFloat next_cutoff = std::numeric_limits<BaseFloat>::infinity();
  // pruning "online" before having seen all tokens

  // Used to keep probabilities in a good dynamic range.
  const BaseFloat cost_offset =
      best_elem ? -best_elem->val->tot_cost : 0.0;
  KALDI_VLOG(6) << "Cost offset on frame " << NumFramesDecoded() << " is "
                << cost_offset;

  // First process the best token to get a hopefully
  // reasonably tight bound on the next cutoff.  The only
  // products of the next block are "next_cutoff".
  if (best_elem) {
    const StateTriplet curr_state = best_elem->key;
    Token *tok = best_elem->val;
    for (fst::ArcIterator<fst::Fst<Arc> > hcl_aiter(hcl_, curr_state[0]);
         !hcl_aiter.Done(); hcl_aiter.Next()) {
      const Arc& hcl_arc = hcl_aiter.Value();
      if (hcl_arc.ilabel == 0) continue;
      const BaseFloat nlkh = -decodable->LogLikelihood(frame, hcl_arc.ilabel);
      if (hcl_arc.olabel == 0) {    // HCL arc has an epsilon output label
        // new_weight = offset + tok->tot_cost + hcl_arc + (-lkh)
        const BaseFloat new_weight = hcl_arc.weight.Value() + nlkh;
        if (new_weight + adaptive_beam < next_cutoff)
          next_cutoff = new_weight + adaptive_beam;
      } else {
        // nolm_cost = offset + tok->tot_cost - tok->LMCost() + hcl_arc + (-lkh)
        const BaseFloat nolm_cost =
            hcl_arc.weight.Value() + nlkh - tok->LMCost();
        FindILabelArcs(curr_state[1], hcl_arc.olabel, &matcher1_, &lm1_arcs_,
                       &lm1_unmatched_arcs_);
        FindILabelArcs(curr_state[2], hcl_arc.olabel, &matcher2_, &lm2_arcs_,
                       &lm2_unmatched_arcs_);
        // Use arcs with hcl_arc.olabel input label from both LMs
        if (!lm1_arcs_.empty() && !lm2_arcs_.empty()) {
          for (const Arc& lm1_arc : lm1_arcs_) {
            const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
            for (const Arc& lm2_arc : lm2_arcs_) {
              // To interpolate both arcs, they must have the same output
              // label. This will be true always in the traditional recipe
              // where LM1 and LM2 are acceptors, but this is added to be
              // safe otherwise.
              if (lm1_arc.olabel != lm2_arc.olabel) continue;
              const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
              const BaseFloat new_weight =
                  nolm_cost + (-kaldi::LogAdd(-lm1_cost, -lm2_cost));
              if (new_weight + adaptive_beam < next_cutoff)
                next_cutoff = new_weight + adaptive_beam;
              // These arcs matched, so, remove them from the unmatched sets.
              lm1_unmatched_arcs_.erase(lm1_arc);
              lm2_unmatched_arcs_.erase(lm2_arc);
            }
          }
        }
        // Process unmatched arcs from LM1
        for (const Arc& lm1_arc : lm1_unmatched_arcs_) {
          const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
          const BaseFloat new_weight = nolm_cost + lm1_cost;
          if (new_weight + adaptive_beam < next_cutoff)
            next_cutoff = new_weight + adaptive_beam;
        }
        // Process unmatched arcs from LM2
        for (const Arc& lm2_arc : lm2_unmatched_arcs_) {
          const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
          const BaseFloat new_weight = nolm_cost + lm2_cost;
          if (new_weight + adaptive_beam < next_cutoff)
            next_cutoff = new_weight + adaptive_beam;
        }
      }
    }
  }

  // Store the offset on the acoustic likelihoods that we're applying.
  // Could just do cost_offsets_.push_back(cost_offset), but we
  // do it this way as it's more robust to future code changes.
  cost_offsets_.resize(frame + 1, 0.0);
  cost_offsets_[frame] = cost_offset;

  KALDI_LOG << "next_cutoff = " << next_cutoff;

  // the tokens are now owned here, in final_toks, and the hash is empty.
  // 'owned' is a complex thing here; the point is we need to call DeleteElem
  // on each elem 'e' to let toks_ know we're done with them.
  for (Elem *e = final_toks, *e_tail = NULL; e != NULL; e = e_tail) {
    // loop this way because we delete "e" as we go.
    const StateTriplet curr_state = e->key;
    Token *tok = e->val;
    if (tok->tot_cost > cur_cutoff) continue;

    for (fst::ArcIterator<fst::Fst<Arc> > hcl_aiter(hcl_, curr_state[0]);
         !hcl_aiter.Done(); hcl_aiter.Next()) {
      const Arc &hcl_arc = hcl_aiter.Value();
      if (hcl_arc.ilabel == 0) continue;

      const BaseFloat nlkh =
          cost_offset - decodable->LogLikelihood(frame, hcl_arc.ilabel);
      const BaseFloat hcl_cost = tok->hcl_cost + hcl_arc.weight.Value();
      if (hcl_arc.olabel == 0) {     // HCL arc with epsilon output
        const BaseFloat tot_cost =
            tok->tot_cost + hcl_arc.weight.Value() + nlkh;
        AddEmittingToken(curr_state, tok,
                         hcl_arc.nextstate, curr_state[1], curr_state[2],
                         frame, hcl_arc.ilabel, 0,
                         tot_cost, hcl_cost, tok->lm1_cost, tok->lm2_cost,
                         nlkh, hcl_arc.weight.Value(), 0.0, 0.0,
                         adaptive_beam, &next_cutoff);
      } else {                      // HCL arc with non-epsilon output
        const BaseFloat nolm_cost =
            (tok->tot_cost - tok->LMCost()) + hcl_arc.weight.Value() + nlkh;
        FindILabelArcs(curr_state[1], hcl_arc.olabel, &matcher1_, &lm1_arcs_,
                       &lm1_unmatched_arcs_);
        FindILabelArcs(curr_state[2], hcl_arc.olabel, &matcher2_, &lm2_arcs_,
                       &lm2_unmatched_arcs_);
        if (!lm1_arcs_.empty() && !lm2_arcs_.empty()) {
          for (const Arc& lm1_arc : lm1_arcs_) {
            const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
            for (const Arc& lm2_arc : lm2_arcs_) {
              if (lm1_arc.olabel != lm2_arc.olabel) continue;
              lm1_unmatched_arcs_.erase(lm1_arc);
              lm2_unmatched_arcs_.erase(lm2_arc);
              const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
              const BaseFloat tot_cost =
                  nolm_cost + (-kaldi::LogAdd(-lm1_cost, -lm2_cost));
              AddEmittingToken(curr_state, tok,
                               hcl_arc.nextstate, lm1_arc.nextstate,
                               lm2_arc.nextstate,
                               frame, hcl_arc.ilabel, lm1_arc.olabel,
                               tot_cost, hcl_cost, lm1_cost, lm2_cost,
                               nlkh, hcl_arc.weight.Value(),
                               lm1_arc.weight.Value(), lm2_arc.weight.Value(),
                               adaptive_beam, &next_cutoff);
            }
          }
        }
        // Process unmatched arcs from LM1
        for (const Arc& lm1_arc : lm1_unmatched_arcs_) {
          const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
          const BaseFloat tot_cost = nolm_cost + lm1_cost;
          AddEmittingToken(curr_state, tok,
                           hcl_arc.nextstate, lm1_arc.nextstate,
                           fst::kNoStateId, frame, hcl_arc.ilabel,
                           lm1_arc.olabel, tot_cost, hcl_cost,
                           lm1_cost, Weight::Zero().Value(), nlkh,
                           hcl_arc.weight.Value(), lm1_arc.weight.Value(),
                           Weight::Zero().Value(), adaptive_beam, &next_cutoff);
        }
        // Process unmatched arcs from LM2
        for (const Arc& lm2_arc : lm2_unmatched_arcs_) {
          const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
          const BaseFloat tot_cost = nolm_cost + lm2_cost;
          AddEmittingToken(curr_state, tok,
                           hcl_arc.nextstate, fst::kNoStateId,
                           lm2_arc.nextstate, frame, hcl_arc.ilabel,
                           lm2_arc.olabel, tot_cost, hcl_cost,
                           Weight::Zero().Value(), lm2_cost, nlkh,
                           hcl_arc.weight.Value(), Weight::Zero().Value(),
                           lm2_arc.weight.Value(), adaptive_beam, &next_cutoff);
        }
      }
    }
    e_tail = e->tail;
    toks_.Delete(e); // delete Elem
  }
  return next_cutoff;
}

void LatticeFasterInterpLmDecoder::ProcessNonemitting(BaseFloat cutoff) {
  KALDI_ASSERT(!active_toks_.empty());
  int32 frame = static_cast<int32>(active_toks_.size()) - 2;
  // Note: "frame" is the time-index we just processed, or -1 if
  // we are processing the nonemitting transitions before the
  // first frame (called from InitDecoding()).

  // Processes nonemitting arcs for one frame.  Propagates within toks_.
  // Note-- this queue structure is is not very optimal as
  // it may cause us to process states unnecessarily (e.g. more than once),
  // but in the baseline code, turning this vector into a set to fix this
  // problem did not improve overall speed.

  KALDI_ASSERT(queue_.empty());
  for (const Elem *e = toks_.GetList(); e != NULL;  e = e->tail)
    queue_.push_back(e->key);

  if (queue_.empty()) {
    if (!warned_) {
      KALDI_WARN << "Error, no surviving tokens: frame is " << frame;
      warned_ = true;
    }
  }

  while (!queue_.empty()) {
    const StateTriplet curr_state = queue_.back();
    queue_.pop_back();

    Token *tok = toks_.Find(curr_state)->val;
    if (tok->tot_cost >= cutoff) continue;

    // If "tok" has any existing forward links, delete them,
    // because we're about to regenerate them.  This is a kind
    // of non-optimality (remember, this is the simple decoder),
    // but since most states are emitting it's not a huge issue.
    tok->DeleteForwardLinks(); // necessary when re-visiting
    tok->links = NULL;

    // Process all arcs with input epsilon in HCL
    for (fst::ArcIterator<Fst> hcl_aiter(hcl_, curr_state[0]);
         !hcl_aiter.Done(); hcl_aiter.Next()) {
      const Arc &hcl_arc = hcl_aiter.Value();
      if (hcl_arc.ilabel != 0) continue;  // Skip arcs with non-epsilon input
      const BaseFloat hcl_cost = tok->hcl_cost + hcl_arc.weight.Value();
      if (hcl_arc.olabel == 0) {    // HCL arc has an epsilon output label
        const BaseFloat tot_cost = tok->tot_cost + hcl_arc.weight.Value();
        AddNonemittingToken(tok, hcl_arc.nextstate, curr_state[1],
                            curr_state[2], frame, 0, tot_cost, hcl_cost,
                            tok->lm1_cost, tok->lm2_cost,
                            hcl_arc.weight.Value(), 0.0, 0.0, cutoff);
      } else {                      // HCL arc has a non-epsilon output label
        // Total cost of the token, minus interpolated LM cost
        const BaseFloat nolm_cost =
            tok->tot_cost + hcl_arc.weight.Value() - tok->LMCost();
        FindILabelArcs(curr_state[1], hcl_arc.olabel, &matcher1_, &lm1_arcs_,
                       &lm1_unmatched_arcs_);
        FindILabelArcs(curr_state[2], hcl_arc.olabel, &matcher2_, &lm2_arcs_,
                       &lm2_unmatched_arcs_);
        // Both LM have arcs with hcl_arc.olabel input label
        if (!lm1_arcs_.empty() && !lm2_arcs_.empty()) {
          for (const Arc& lm1_arc : lm1_arcs_) {
            const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
            for (const Arc& lm2_arc : lm2_arcs_) {
              if (lm1_arc.olabel != lm2_arc.olabel) continue;
              lm1_unmatched_arcs_.erase(lm1_arc);
              lm2_unmatched_arcs_.erase(lm2_arc);
              const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
              const BaseFloat tot_cost =
                  nolm_cost - kaldi::LogAdd(-lm1_cost, -lm2_cost);
              AddNonemittingToken(tok, hcl_arc.nextstate, lm1_arc.nextstate,
                                  lm2_arc.nextstate, frame, lm1_arc.olabel,
                                  tot_cost, hcl_cost, lm1_cost, lm2_cost,
                                  hcl_arc.weight.Value(),
                                  lm1_arc.weight.Value(),
                                  lm2_arc.weight.Value(), cutoff);
            }
          }
        }
        // Process unmatched arcs from LM1
        for (const Arc& lm1_arc : lm1_unmatched_arcs_) {
          const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
          const BaseFloat tot_cost = nolm_cost + lm1_cost;
          AddNonemittingToken(tok, hcl_arc.nextstate, lm1_arc.nextstate,
                              fst::kNoStateId, frame, lm1_arc.olabel,
                              tot_cost, hcl_cost, lm1_cost,
                              Weight::Zero().Value(), hcl_arc.weight.Value(),
                              lm1_arc.weight.Value(), Weight::Zero().Value(),
                              cutoff);
        }
        // Process unmatched arcs from LM2
        for (const Arc& lm2_arc : lm2_unmatched_arcs_) {
          const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
          const BaseFloat tot_cost = nolm_cost + lm2_cost;
          AddNonemittingToken(tok, hcl_arc.nextstate, fst::kNoStateId,
                              lm2_arc.nextstate, frame, lm2_arc.olabel,
                              tot_cost, hcl_cost, Weight::Zero().Value(),
                              lm2_cost, hcl_arc.weight.Value(),
                              Weight::Zero().Value(), lm2_arc.weight.Value(),
                              cutoff);
        }
      }
    } // for all arcs with input epsilon in HCL

    // Process all arcs with input epsilon in LM1
    if (curr_state[1] != fst::kNoStateId) {
      for (fst::ArcIterator<Fst> lm1_aiter(lm1_, curr_state[1]);
           !lm1_aiter.Done(); lm1_aiter.Next()) {
        const Arc& lm1_arc = lm1_aiter.Value();
        if (lm1_arc.ilabel != 0) continue;
        const BaseFloat lm1_cost = tok->lm1_cost + lm1_arc.weight.Value();
        const BaseFloat lm2_cost = tok->lm2_cost;
        const BaseFloat tot_cost =
            tok->tot_cost - tok->LMCost() - kaldi::LogAdd(-lm1_cost, -lm2_cost);
        AddNonemittingToken(
            tok, curr_state[0], lm1_arc.nextstate, curr_state[2], frame,
            lm1_arc.olabel, tot_cost, 0.0, lm1_cost, lm2_cost, 0.0,
            lm1_arc.weight.Value(), 0.0, cutoff);
      }
    }

    // Process all arcs with input epsilon in LM2
    if (curr_state[2] != fst::kNoStateId) {
      for (fst::ArcIterator<Fst> lm2_aiter(lm2_, curr_state[2]);
           !lm2_aiter.Done(); lm2_aiter.Next()) {
        const Arc& lm2_arc = lm2_aiter.Value();
        if (lm2_arc.ilabel != 0) continue;
        const BaseFloat lm1_cost = tok->lm1_cost;
        const BaseFloat lm2_cost = tok->lm2_cost + lm2_arc.weight.Value();
        const BaseFloat tot_cost =
            tok->tot_cost - tok->LMCost() - kaldi::LogAdd(-lm1_cost, -lm2_cost);
        AddNonemittingToken(
            tok, curr_state[0], curr_state[1], lm2_arc.nextstate, frame,
            lm2_arc.olabel, tot_cost, 0.0, lm1_cost, lm2_cost, 0.0,
            0.0, lm2_arc.weight.Value(), cutoff);
      }
    }
  } // while queue not empty
}


void LatticeFasterInterpLmDecoder::DeleteElems(Elem *list) {
  for (Elem *e = list, *e_tail = NULL; e != NULL; e = e_tail) {
    e_tail = e->tail;
    toks_.Delete(e);
  }
}

void LatticeFasterInterpLmDecoder::ClearActiveTokens() { // a cleanup routine, at utt end/begin
  for (size_t i = 0; i < active_toks_.size(); i++) {
    // Delete all tokens alive on this frame, and any forward
    // links they may have.
    for (Token *tok = active_toks_[i].toks; tok != NULL; ) {
      tok->DeleteForwardLinks();
      Token *next_tok = tok->next;
      delete tok;
      num_toks_--;
      tok = next_tok;
    }
  }
  active_toks_.clear();
  KALDI_ASSERT(num_toks_ == 0);
}

// static
void LatticeFasterInterpLmDecoder::TopSortTokens(
    Token *tok_list, std::vector<Token*> *topsorted_list) {
  unordered_map<Token*, int32> token2pos;
  typedef unordered_map<Token*, int32>::iterator IterType;
  int32 num_toks = 0;
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    num_toks++;
  int32 cur_pos = 0;
  // We assign the tokens numbers num_toks - 1, ... , 2, 1, 0.
  // This is likely to be in closer to topological order than
  // if we had given them ascending order, because of the way
  // new tokens are put at the front of the list.
  for (Token *tok = tok_list; tok != NULL; tok = tok->next)
    token2pos[tok] = num_toks - ++cur_pos;

  unordered_set<Token*> reprocess;

  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter) {
    Token *tok = iter->first;
    int32 pos = iter->second;
    for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
      if (link->ilabel == 0) {
        // We only need to consider epsilon links, since non-epsilon links
        // transition between frames and this function only needs to sort a list
        // of tokens from a single frame.
        IterType following_iter = token2pos.find(link->next_tok);
        if (following_iter != token2pos.end()) { // another token on this frame,
                                                 // so must consider it.
          int32 next_pos = following_iter->second;
          if (next_pos < pos) { // reassign the position of the next Token.
            following_iter->second = cur_pos++;
            reprocess.insert(link->next_tok);
          }
        }
      }
    }
    // In case we had previously assigned this token to be reprocessed, we can
    // erase it from that set because it's "happy now" (we just processed it).
    reprocess.erase(tok);
  }

  // max_loop is to detect epsilon cycles.
  size_t max_loop = 1000000, loop_count;
  for (loop_count = 0;
       !reprocess.empty() && loop_count < max_loop; ++loop_count) {
    std::vector<Token*> reprocess_vec;
    for (unordered_set<Token*>::iterator iter = reprocess.begin();
         iter != reprocess.end(); ++iter)
      reprocess_vec.push_back(*iter);
    reprocess.clear();
    for (std::vector<Token*>::iterator iter = reprocess_vec.begin();
         iter != reprocess_vec.end(); ++iter) {
      Token *tok = *iter;
      int32 pos = token2pos[tok];
      // Repeat the processing we did above (for comments, see above).
      for (ForwardLink *link = tok->links; link != NULL; link = link->next) {
        if (link->ilabel == 0) {
          IterType following_iter = token2pos.find(link->next_tok);
          if (following_iter != token2pos.end()) {
            int32 next_pos = following_iter->second;
            if (next_pos < pos) {
              following_iter->second = cur_pos++;
              reprocess.insert(link->next_tok);
            }
          }
        }
      }
    }
  }
  KALDI_ASSERT(loop_count < max_loop && "Epsilon loops exist in your decoding "
               "graph (this is not allowed!)");

  topsorted_list->clear();
  topsorted_list->resize(cur_pos, NULL);  // create a list with NULLs in between.
  for (IterType iter = token2pos.begin(); iter != token2pos.end(); ++iter)
    (*topsorted_list)[iter->second] = iter->first;
}

void LatticeFasterInterpLmDecoder::FindILabelArcs(
    StateId s, Label l, fst::SortedMatcher<Fst>* m, std::vector<Arc>* arcs,
    std::unordered_set<Arc>* unmatched_arcs) {
  KALDI_ASSERT(m != NULL);
  KALDI_ASSERT(arcs != NULL);
  KALDI_ASSERT(unmatched_arcs != NULL);
  arcs->clear();
  unmatched_arcs->clear();
  if (s != fst::kNoStateId) {
    m->SetState(s);
    for(m->Find(l); !m->Done(); m->Next()) {
      const Arc& arc = m->Value();
      auto insert_ret = unmatched_arcs->insert(arc);
      // If an arc to the same state and same input/output labels already
      // exists, update the previous arc's weight to be the log-sum of the two.
      if (!insert_ret.second) {
        const Arc cur_arc = *insert_ret.first;
        const Arc new_arc(arc.ilabel, arc.olabel,
                          Weight(-kaldi::LogAdd(-cur_arc.weight.Value(),
                                                -arc.weight.Value())),
                          arc.nextstate);
        auto hint = insert_ret.first;
        ++hint;
        unmatched_arcs->erase(insert_ret.first);
        unmatched_arcs->insert(hint, new_arc);
      }
    }
  }
  arcs->assign(unmatched_arcs->begin(), unmatched_arcs->end());
}


} // end namespace kaldi.
