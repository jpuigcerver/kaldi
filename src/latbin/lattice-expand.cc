// latbin/lattice-expand.cc

// Copyright 2016       Joan Puigcerver

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace fst {

template <typename Arc>
class ExpandFstWithBreakLabels {
 public:
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;
  typedef StateIterator< Fst<Arc> > MyStateIterator;
  typedef ArcIterator< Fst<Arc> > MyArcIterator;
  typedef std::unordered_map< std::vector<Label>, Label, kaldi::VectorHasher<Label> > LabelMap;

  ExpandFstWithBreakLabels(const Fst<Arc>& ifst, MutableFst<Arc>* ofst,
                           const std::unordered_set<Label>& break_labels,
                           const bool break_input = false) :
      ifst_(ifst), ofst_(ofst), break_labels_(break_labels), break_input_(break_input) {
    KALDI_ASSERT(ofst_ != NULL);
    Expand();
  }

  inline const LabelMap& InputMap() const { return isym_map_; }

  inline const LabelMap& OutputMap() const { return osym_map_; }

 private:
  const Fst<Arc>& ifst_;
  MutableFst<Arc>* ofst_;
  const std::unordered_set<Label>& break_labels_;
  const bool break_input_;
  std::unordered_set<StateId> expanded_from_;
  LabelMap isym_map_;
  LabelMap osym_map_;

  static Label FindOrAssignLabel(const std::vector<Label>& v, LabelMap* m) {
    KALDI_ASSERT(m != NULL);
    return m->insert(std::make_pair(v, m->size())).first->second;
  }

  void Expand() {
    ofst_->DeleteStates();
    if (ifst_.Start() < 0) return;

    // Output FST will have as many states as the input FST.
    // NOTE: We require that all final states have no output arcs.
    for (MyStateIterator siter(ifst_); !siter.Done(); siter.Next()) {
      const StateId s = siter.Value();
      KALDI_ASSERT(ifst_.Final(s) == Weight::Zero() || ifst_.NumArcs(s) == 0);
      ofst_->SetFinal(ofst_->AddState(), ifst_.Final(s));
    }

    // Initialize symbols table with epsilons
    isym_map_[std::vector<Label>()] = 0;
    osym_map_[std::vector<Label>()] = 0;

    // Add arcs to the output FST which are labeled with any of the break symbols.
    // It also adds the corresponding symbols to the SymbolMap tables.
    // NOTE: Some of these arcs may be unreachable once the whole input FST is
    // expanded, that's why we trim the output FST later.
    for (MyStateIterator siter(ifst_); !siter.Done(); siter.Next()) {
      for (MyArcIterator aiter(ifst_, siter.Value()); !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        if (break_labels_.count(break_input_ ? arc.ilabel : arc.olabel) > 0) {
          if (arc.ilabel != 0)
            arc.ilabel = FindOrAssignLabel(std::vector<Label>(1, arc.ilabel), &isym_map_);
          if (arc.olabel != 0)
            arc.olabel = FindOrAssignLabel(std::vector<Label>(1, arc.olabel), &osym_map_);
          ofst_->AddArc(siter.Value(), arc);
        }
      }
    }

    // Start expansion from the initial state
    expanded_from_.insert(ifst_.Start());
    std::vector<Label> tmp_isym, tmp_osym;
    Weight tmp_w = Weight::One();
    ofst_->SetStart(ifst_.Start());
    ExpandFromState(ofst_->Start(), ifst_.Start(), &tmp_w, &tmp_isym, &tmp_osym);

    // Connect output FST to remove unnecessary states and arcs.
    Connect(ofst_);
  }

  // Expand path from state s to t, with the current weight and sequence of
  // input/output labels.
  // Note: Weight and input/output labels are passed as reference to avoid
  // extra memory during recursion.
  void ExpandFromState(StateId s, StateId t, Weight* w,
                       std::vector<Label>* isym, std::vector<Label>* osym) {
    if (ifst_.Final(t) != Weight::Zero()) {
      Arc newarc(FindOrAssignLabel(*isym, &isym_map_),
                 FindOrAssignLabel(*osym, &osym_map_), *w, t);
      ofst_->AddArc(s, newarc);
    } else {
      for (MyArcIterator aiter(ifst_, t); !aiter.Done(); aiter.Next()) {
        const Arc& arc = aiter.Value();
        if (break_labels_.count(break_input_ ? arc.ilabel : arc.olabel) > 0) {
          Arc newarc(FindOrAssignLabel(*isym, &isym_map_),
                     FindOrAssignLabel(*osym, &osym_map_), *w, t);
          ofst_->AddArc(s, newarc);
          if (expanded_from_.count(arc.nextstate) == 0) {
            expanded_from_.insert(arc.nextstate);
            std::vector<Label> tmp_isym, tmp_osym;
            Weight tmp_w = Weight::One();
            ExpandFromState(arc.nextstate, arc.nextstate, &tmp_w, &tmp_isym, &tmp_osym);
          }
        } else {
          if (arc.ilabel != 0) isym->push_back(arc.ilabel);
          if (arc.olabel != 0) osym->push_back(arc.olabel);
          *w = Times(*w, arc.weight);
          ExpandFromState(s, arc.nextstate, w, isym, osym);
          *w = Divide(*w, arc.weight, DIVIDE_RIGHT);
          if (arc.ilabel != 0) isym->pop_back();
          if (arc.olabel != 0) osym->pop_back();
        }
      }
    }
  }
};

}  // namespace fst


size_t CountNumArcs(const kaldi::CompactLattice& lat) {
  size_t n = 0;
  for (fst::StateIterator<kaldi::CompactLattice> s(lat); !s.Done(); s.Next())
    n += lat.NumArcs(s.Value());
  return n;
}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Expand lattices so that each arc represents a \"word\", a word is "
        "any sequence of symbols surrounded by any of the separator symbols "
        "in --separator-symbols.\n"
        "For each lattice, a table containing the mapping from new label IDs "
        "to old label sequences is saved.\n\n"
        "Usage:  lattice-expand lats_rspecifier lats_wspecifier "
        "syms_wspecifier\n\n";

    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat lm_scale = 1.0;
    BaseFloat beam = std::numeric_limits<BaseFloat>::infinity();
    std::string break_symbols_str = "";
    po.Register("separator-symbols", &break_symbols_str,
                "Space-separated list of separator output symbols in the "
                "original lattices.");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for graph/lm costs");
    po.Register("beam", &beam,
                "Pruning beam [applied after acoustic and lm scaling]");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::unordered_set<kaldi::int32> break_symbols;
    std::istringstream break_symbols_iss(break_symbols_str);
    kaldi::int32 tmp;
    while (break_symbols_iss >> tmp) break_symbols.insert(tmp);

    std::string lats_rspecifier = po.GetArg(1),
        lats_wspecifier = po.GetArg(2),
        syms_wspecifier = po.GetArg(3);

    if (acoustic_scale == 0.0 || lm_scale == 0.0)
      KALDI_ERR << "Do not use a zero scales (cannot be inverted)";

    std::vector<std::vector<double> > scale(2);
    scale[0].resize(2, 0.0);
    scale[1].resize(2, 0.0);
    scale[0][0] = lm_scale;
    scale[1][1] = acoustic_scale;

    std::vector<std::vector<double> > inv_scale(2);
    inv_scale[0].resize(2, 0.0);
    inv_scale[1].resize(2, 0.0);
    inv_scale[0][0] = 1.0 / lm_scale;
    inv_scale[1][1] = 1.0 / acoustic_scale;

    // Read as regular lattice-- this is the form we need it in for efficient
    // pruning.
    SequentialCompactLatticeReader lattice_reader(lats_rspecifier);

    // Write as compact lattice.
    CompactLatticeWriter lattice_writer(lats_wspecifier);

    // Write symbols table.
    Int32VectorVectorWriter symbols_writer(syms_wspecifier);

    int32 n_done = 0, n_error = 0;
    for (; !lattice_reader.Done(); lattice_reader.Next()) {
      std::string key = lattice_reader.Key();
      CompactLattice lat = lattice_reader.Value();
      lattice_reader.FreeCurrent();
      // Scale input lattice
      fst::ScaleLattice(scale, &lat);
      // Prune input lattice, before expansion
      const size_t original_num_states = lat.NumStates();
      const size_t original_num_arcs   = CountNumArcs(lat);
      if (KALDI_ISFINITE(beam) && !PruneLattice(beam, &lat)) {
        KALDI_WARN << "Error pruning lattice for utterance " << key;
        n_error++;
        continue;
      }

      // Replace all final states with epsilon transitions to a single final state.
      // This is required by the expansion algorithm to work.
      {
        const CompactLatticeArc::StateId tmp_final = lat.AddState();
        for (fst::StateIterator<CompactLattice> siter(lat); !siter.Done(); siter.Next()) {
          const CompactLatticeWeight& final_weight = lat.Final(siter.Value());
          if (final_weight != CompactLatticeWeight::Zero()) {
            lat.AddArc(siter.Value(), CompactLatticeArc(0, 0, final_weight, tmp_final));
            lat.SetFinal(siter.Value(), CompactLatticeWeight::Zero());
          }
        }
        lat.SetFinal(tmp_final, CompactLatticeWeight::One());
      }

      // Get expanded lattice
      CompactLattice flat;
      fst::ExpandFstWithBreakLabels<CompactLatticeArc> expander(lat, &flat, break_symbols);

      const size_t expanded_num_states = flat.NumStates();
      const size_t expanded_num_arcs   = CountNumArcs(flat);
      KALDI_LOG << "Lattice " << key << " expanded from "
                << original_num_states << " states and "
                << original_num_arcs << " arcs to "
                << expanded_num_states << " states and "
                << expanded_num_arcs << " arcs.";

      // Scale Lattice back to the original scale
      fst::ScaleLattice(inv_scale, &flat);
      // Write CompactLattice
      lattice_writer.Write(key, flat);
      // Write output symbols
      //symbols_writer.Write(key, osymbols);
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " lattices, errors on " << n_error;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
