// fb/simple-common.h

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

#ifndef KALDI_FB_SIMPLE_COMMON_H_
#define KALDI_FB_SIMPLE_COMMON_H_

#include "fst/fstlib.h"
#include "util/stl-utils.h"
#include "hmm/posterior.h"
#include "itf/decodable-itf.h"

namespace kaldi {

typedef fst::StdArc::StateId StateId;
typedef fst::StdArc::Label Label;
typedef unordered_map<Label, double>  LabelMap;

/// This structure is used by the forward/backward algorithms to
/// store the active states in the trellis at different timesteps.
struct Token {
  double cost;         // total cost to the state
  double last_cost;    // cost to the state, since the last extraction from
                       // the shortest-distance algorithm queue (see [1]).

  Token(double c) : cost(c), last_cost(-kaldi::kLogZeroDouble) { }

  // Update token when processing non-epsilon edges
  void UpdateEmitting(
      const double prev_cost, const double edge_cost,
      const double acoustic_cost);

  // Update token when processing epsilon edges
  bool UpdateNonEmitting(
      const double prev_cost, const double edge_cost,
      const double threshold);
};
typedef unordered_map<StateId, Token> TokenMap;

/// Prune all tokens (states) in a TokenMap whose cost is larger than
/// the minimum cost + the beam.
void PruneToks(double beam, TokenMap *toks);

/// Prune all tokens (states) from the forward and backward passes whose
/// cost is larger than the total likelihood + the beam.
void PruneToksForwardBackward(
    double lkh, double beam, TokenMap *fwd, TokenMap *bkw);

/// Rescale all tokens, such that the log-sum of the costs is 0.0 (1.0)
double RescaleToks(TokenMap* toks);

/// Given the forward and the backward tokens at time 0, compute the
/// likelihood of the obverved sequence.
double ComputeLikelihood(const TokenMap& fwd0, const TokenMap& bkw0);

/// Compute label (typically transition-ids) posteriors for each timestep.
/// Used for EM-reestimation of the transition/emission probabilities.
/// NOTE: This ignores epsilon arcs, since I am assuming epsilon arcs
/// weights are not tuneable (they do not correspond to any transition-id).
void ComputeLabelsPosterior(
    const fst::Fst<fst::StdArc>& fst,
    const std::vector<TokenMap>& fwd,
    const std::vector<TokenMap>& bkw,
    DecodableInterface* decodable,
    std::vector<LabelMap>* pst);


/// Compute label (typically transition-ids) posteriors at time t.
/// Useful to cumpte the label posteriors iteratively without having full
/// forward and backward trellis in memory.
/// The posteriors can be used for the EM-reestimation of the
/// transition/emission probabilities.
/// NOTE: This ignores epsilon arcs, since I am assuming epsilon arcs weights
/// are not tuneable (the do not correspond to any transition-id).
void ComputeLabelsPosteriorAtTimeT(
    const fst::Fst<fst::StdArc>& fst, DecodableInterface* decodable,
    const int32 t,  const TokenMap& fwd_t,  const TokenMap& bkw_tp1,
    LabelMap* pst);

/// Debug utils.
void PrintTokenMap(const TokenMap& toks, const string& name = "", int32 t = -1);
void PrintTokenTable(const vector<TokenMap>& table, const string& name = "");

}  // namespace kaldi

#endif  // KALDI_FB_SIMPLE_COMMON_H_
