// decoder/faster-decoder-interp-lm.h

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

#ifndef KALDI_DECODER_FASTER_DECODER_H_
#define KALDI_DECODER_FASTER_DECODER_H_

#include "util/stl-utils.h"
#include "itf/options-itf.h"
#include "util/hash-list.h"
#include "fst/fstlib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h" // for CompactLatticeArc

namespace kaldi {

struct FasterDecoderInterpLmOptions {
  BaseFloat beam;
  int32 max_active;
  int32 min_active;
  BaseFloat beam_delta;
  BaseFloat hash_ratio;
  BaseFloat interp_weight;
  FasterDecoderInterpLmOptions(): beam(16.0),
                                  max_active(std::numeric_limits<int32>::max()),
                                  min_active(20), // This decoder mostly used for
                                                  // alignment, use small default.
                                  beam_delta(0.5),
                                  hash_ratio(2.0),
                                  interp_weight(0.5) { }
  void Register(OptionsItf *opts, bool full) {  /// if "full", use obscure
    /// options too.
    /// Depends on program.
    opts->Register("beam", &beam, "Decoder beam");
    opts->Register("max-active", &max_active, "Decoder max active states.");
    opts->Register("min-active", &min_active,
                   "Decoder min active states (don't prune if #active less than this).");
    opts->Register("interp-weight", &interp_weight,
                   "Interpolation weight for the first language model (0 <= weight <= 1).");
    if (full) {
      opts->Register("beam-delta", &beam_delta,
                     "Increment used in decoder [obscure setting]");
      opts->Register("hash-ratio", &hash_ratio,
                     "Setting used in decoder to control hash behavior");
    }
  }
};

class FasterDecoderInterpLm {
 public:
  typedef fst::StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  FasterDecoderInterpLm(const fst::Fst<fst::StdArc> &hcl,
                        fst::DeterministicOnDemandFst<fst::StdArc> &lm1,
                        fst::DeterministicOnDemandFst<fst::StdArc> &lm2,
                        const FasterDecoderInterpLmOptions &config);

  void SetOptions(const FasterDecoderInterpLmOptions &config) { config_ = config; }

  ~FasterDecoderInterpLm() { ClearToks(toks_.Clear()); }

  void Decode(DecodableInterface *decodable);

  /// Returns true if a final state was active on the last frame.
  bool ReachedFinal();

  /// GetBestPath gets the decoding traceback. If "use_final_probs" is true
  /// AND we reached a final state, it limits itself to final states;
  /// otherwise it gets the most likely token not taking into account
  /// final-probs. Returns true if the output best path was not the empty
  /// FST (will only return false in unusual circumstances where
  /// no tokens survived).
  bool GetBestPath(fst::MutableFst<LatticeArc> *fst_out,
                   bool use_final_probs = true);

  /// As a new alternative to Decode(), you can call InitDecoding
  /// and then (possibly multiple times) AdvanceDecoding().
  void InitDecoding();


  /// This will decode until there are no more frames ready in the decodable
  /// object, but if max_num_frames is >= 0 it will decode no more than
  /// that many frames.
  void AdvanceDecoding(DecodableInterface *decodable,
                       int32 max_num_frames = -1);

  /// Returns the number of frames already decoded.
  int32 NumFramesDecoded() const { return num_frames_decoded_; }

 protected:

  // Struct used to represent a state
  struct StateTriplet {
    StateId state[3];
    StateTriplet() {
      state[0] = state[1] = state[2] = fst::kNoStateId;
    }
    StateTriplet(StateId state0, StateId state1, StateId state2) {
      state[0] = state0;
      state[1] = state1;
      state[2] = state2;
    }

    // Support for static_cast<size_t>(StateTriplet()), used by HashList
    operator size_t() {
      return static_cast<size_t>(state[0] ^ state[1] ^ state[2]);
    }
    // Shortcut for state.state[i]
    StateId operator[](const size_t i) const {
#ifdef KALDI_PARANOID
      KALDI_ASSERT(i < 3);
#endif
      return state[i];
    }
  };

  class Token {
   public:
    StateTriplet state_;
    BaseFloat cost_[4];
    Label ilabel_;
    Label olabel_;
    // we can work out the acoustic part from difference between
    // "cost_" and prev->cost_.
    Token *prev_;
    int32 ref_count_;
    inline Token(const Arc &arc_hcl, const Arc &arc_lm1, const Arc &arc_lm2,
                 BaseFloat ac_cost, Token *prev):
        state_(arc_hcl.nextstate, arc_lm1.nextstate, arc_lm2.nextstate),
        ilabel_(arc_hcl.ilabel), olabel_(arc_lm1.olabel), prev_(prev), ref_count_(1) {
#ifdef KALDI_PARANOID
      KALDI_ASSERT(arc_lm1.olabel == arc_lm2.olabel);
#endif
      if (prev) {
        prev->ref_count_++;
        cost_[0] = prev->cost_[0] + arc_hcl.weight.Value();
        cost_[1] = prev->cost_[1] + arc_lm1.weight.Value();
        cost_[2] = prev->cost_[2] + arc_lm2.weight.Value();
        cost_[3] = prev->cost_[3] + ac_cost;
      } else {
        cost_[0] = arc_hcl.weight.Value();
        cost_[1] = arc_lm1.weight.Value();
        cost_[2] = arc_lm2.weight.Value();
        cost_[3] = ac_cost;
      }
    }
    inline Token(const Arc &arc_hcl, const Arc &arc_lm1, const Arc &arc_lm2,
                 Token *prev):
        state_(arc_hcl.nextstate, arc_lm1.nextstate, arc_lm2.nextstate),
        ilabel_(arc_hcl.ilabel), olabel_(arc_lm1.olabel), prev_(prev), ref_count_(1) {
#ifdef KALDI_PARANOID
      KALDI_ASSERT(arc_lm1.olabel == arc_lm2.olabel);
#endif
      if (prev) {
        prev->ref_count_++;
        cost_[0] = prev->cost_[0] + arc_hcl.weight.Value();
        cost_[1] = prev->cost_[1] + arc_lm1.weight.Value();
        cost_[2] = prev->cost_[2] + arc_lm2.weight.Value();
        cost_[3] = prev->cost_[3];
      } else {
        cost_[0] = arc_hcl.weight.Value();
        cost_[1] = arc_lm1.weight.Value();
        cost_[2] = arc_lm2.weight.Value();
        cost_[3] = 0.0;
      }
    }

    inline BaseFloat TotalCost() const {
      return cost_[0] - kaldi::LogAdd(-cost_[1], -cost_[2]) + cost_[3];
    }

    inline BaseFloat TotalFinalCost(
        const fst::Fst<fst::StdArc>& hcl,
        const fst::DeterministicOnDemandFst<fst::StdArc>& lm1,
        const fst::DeterministicOnDemandFst<fst::StdArc>& lm2) const {
      const BaseFloat infinity = std::numeric_limits<BaseFloat>::infinity();
      const BaseFloat f_hcl =
          state_[0] != fst::kNoStateId ? hcl.Final(state_[0]).Value() : infinity;
      const BaseFloat f_lm1 =
          state_[1] != fst::kNoStateId ? hcl.Final(state_[1]).Value() : infinity;
      const BaseFloat f_lm2 =
          state_[2] != fst::kNoStateId ? hcl.Final(state_[2]).Value() : infinity;
      return (cost_[0] + f_hcl)                                   // Cost through HCL
          - kaldi::LogAdd(-cost_[1] - f_lm1, -cost_[2] - f_lm2)   // Cost through LM
          + cost_[3];                                             // Acoustic cost
    }

    inline bool operator < (const Token &other) {
      return TotalCost() > other.TotalCost();
    }

    inline static void TokenDelete(Token *tok) {
      while (--tok->ref_count_ == 0) {
        Token *prev = tok->prev_;
        delete tok;
        if (prev == NULL) return;
        else tok = prev;
      }
#ifdef KALDI_PARANOID
      KALDI_ASSERT(tok->ref_count_ > 0);
#endif
    }
  };
  typedef HashList<StateTriplet, Token*>::Elem Elem;


  /// Gets the weight cutoff.  Also counts the active tokens.
  double GetCutoff(Elem *list_head, size_t *tok_count,
                   BaseFloat *adaptive_beam, Elem **best_elem);

  void PossiblyResizeHash(size_t num_toks);

  // ProcessEmitting returns the likelihood cutoff used.
  // It decodes the frame num_frames_decoded_ of the decodable object
  // and then increments num_frames_decoded_
  double ProcessEmitting(DecodableInterface *decodable);

  // TODO: first time we go through this, could avoid using the queue.
  void ProcessNonemitting(double cutoff);

  // Find arcs from the current state in each LM with the given label.
  void FindLmArcs(StateId state_lm1, StateId state_lm2, Label label,
                  Arc *arc_lm1, Arc *arc_lm2);

  // HashList defined in ../util/hash-list.h.  It actually allows us to maintain
  // more than one list (e.g. for current and previous frames), but only one of
  // them at a time can be indexed by StateId.
  HashList<StateTriplet, Token*> toks_;
  const fst::Fst<fst::StdArc> &hcl_;
  fst::DeterministicOnDemandFst<fst::StdArc> &lm1_;
  fst::DeterministicOnDemandFst<fst::StdArc> &lm2_;
  FasterDecoderInterpLmOptions config_;
  std::vector<StateTriplet> queue_;  // temp variable used in ProcessNonemitting,
  std::vector<BaseFloat> tmp_array_;  // used in GetCutoff.
  // make it class member to avoid internal new/delete.

  // Keep track of the number of frames decoded in the current file.
  int32 num_frames_decoded_;

  // It might seem unclear why we call ClearToks(toks_.Clear()).
  // There are two separate cleanup tasks we need to do at when we start a new file.
  // one is to delete the Token objects in the list; the other is to delete
  // the Elem objects.  toks_.Clear() just clears them from the hash and gives ownership
  // to the caller, who then has to call toks_.Delete(e) for each one.  It was designed
  // this way for convenience in propagating tokens from one frame to the next.
  void ClearToks(Elem *list);

  KALDI_DISALLOW_COPY_AND_ASSIGN(FasterDecoderInterpLm);
};


} // end namespace kaldi.


#endif
