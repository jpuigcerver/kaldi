// latbin/lattice-determinize.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)

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
#include "lat/minimize-lattice.h"

template<class Arc, class I, class O>
void FstExpandWithBreaksRecursive(
    const fst::Fst<Arc>& fst,
    const std::unordered_set<typename Arc::Label>& break_labels,
    fst::MutableFst<Arc>* ofst,
    std::unordered_map<typename Arc::StateId,
    typename Arc::StateId>* state_mapping,
    std::unordered_map<std::vector<I>, typename Arc::Label,
    kaldi::VectorHasher<I> >* isym_mapping,
    std::unordered_map<std::vector<O>, typename Arc::Label,
    kaldi::VectorHasher<O> >* osym_mapping,
    std::unordered_set<typename Arc::StateId>* expanded_from,
    typename Arc::StateId state,
    typename Arc::StateId src_state,
    typename Arc::Weight weight,
    std::vector<I>* isym, std::vector<O>* osym) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  if (isym->empty() && osym->empty())
    expanded_from->insert(state);

  for (fst::ArcIterator<fst::Fst<Arc>> aiter(fst, state); !aiter.Done();
       aiter.Next()) {
    const Arc& arc = aiter.Value();
    if (break_labels.count(arc.olabel) > 0 ||
        fst.Final(arc.nextstate) != Weight::Zero()) {
      Arc newarc(
          isym_mapping->insert(
              make_pair(*isym, isym_mapping->size())).first->second,
          osym_mapping->insert(
              make_pair(*osym, osym_mapping->size())).first->second,
          Times(weight, arc.weight), fst::kNoStateId);
      if (state_mapping->count(arc.nextstate) == 0)
        newarc.nextstate = (*state_mapping)[arc.nextstate] = ofst->AddState();
      else
        newarc.nextstate = (*state_mapping)[arc.nextstate];
      ofst->AddArc(src_state, newarc);
      ofst->SetFinal(newarc.nextstate, fst.Final(arc.nextstate));
      if (break_labels.count(arc.olabel) > 0 &&
          expanded_from->count(arc.nextstate) == 0) {
        std::vector<I> tmp_inp;
        std::vector<O> tmp_out;
        FstExpandWithBreaksRecursive(fst, break_labels, ofst,
                                     state_mapping, isym_mapping, osym_mapping,
                                     expanded_from,
                                     arc.nextstate, newarc.nextstate,
                                     Weight::One(), &tmp_inp, &tmp_out);

      } else {
        if (arc.ilabel != 0) isym->push_back(arc.ilabel);
        if (arc.olabel != 0) osym->push_back(arc.olabel);
        FstExpandWithBreaksRecursive(fst, break_labels, ofst,
                                     state_mapping, isym_mapping, osym_mapping,
                                     expanded_from,
                                     arc.nextstate, newarc.nextstate,
                                     Weight::One(), isym, osym);
        if (arc.ilabel != 0) isym->pop_back();
        if (arc.olabel != 0) osym->pop_back();
      }
    } else {
      if (arc.ilabel != 0) isym->push_back(arc.ilabel);
      if (arc.olabel != 0) osym->push_back(arc.olabel);
      FstExpandWithBreaksRecursive(fst, break_labels, ofst,
                                   state_mapping, isym_mapping, osym_mapping,
                                   expanded_from,
                                   arc.nextstate, src_state,
                                   Times(weight, arc.weight), isym, osym);
      if (arc.ilabel != 0) isym->pop_back();
      if (arc.olabel != 0) osym->pop_back();
    }
  }
}

template<class Arc, class I, class O>
void FstExpandWithBreaks(
    const fst::Fst<Arc>& fst,
    const std::unordered_set<typename Arc::Label>& break_labels,
    fst::MutableFst<Arc>* ofst,
    std::vector< std::vector<I> >* symbols_inp,
    std::vector< std::vector<O> >* symbols_out) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  KALDI_ASSERT_IS_INTEGER_TYPE(O);
  KALDI_ASSERT(ofst != NULL);
  ofst->DeleteStates();
  if (fst.Start() < 0) return;

  std::unordered_map<StateId, StateId> state_mapping;
  std::unordered_map<std::vector<I>, Label, kaldi::VectorHasher<I>>
      isym_mapping;
  std::unordered_map<std::vector<O>, Label, kaldi::VectorHasher<O>>
      osym_mapping;
  std::unordered_set<StateId> expanded_from;
  ofst->SetStart(ofst->AddState());
  state_mapping[fst.Start()] = ofst->Start();
  isym_mapping[std::vector<I>()] = 0;
  osym_mapping[std::vector<O>()] = 0;
  expanded_from.insert(fst.Start());

  std::vector<I> tmp_inp;
  std::vector<O> tmp_out;
  FstExpandWithBreaksRecursive(fst, break_labels, ofst,
                               &state_mapping, &isym_mapping, &osym_mapping,
                               &expanded_from, fst.Start(), ofst->Start(),
                               Weight::One(), &tmp_inp, &tmp_out);

  if (symbols_inp) {
    symbols_inp->resize(isym_mapping.size());
    for (auto it = isym_mapping.begin(); it != isym_mapping.end(); ++it)
      (*symbols_inp)[it->second] = it->first;
  }
  if (symbols_out) {
    symbols_out->resize(osym_mapping.size());
    for (auto it = osym_mapping.begin(); it != osym_mapping.end(); ++it)
      (*symbols_out)[it->second] = it->first;
  }
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
                "original lattices");
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
      if (KALDI_ISFINITE(beam) && !PruneLattice(beam, &lat)) {
        KALDI_WARN << "Error pruning lattice for utterance " << key;
        n_error++;
        continue;
      }
      CompactLattice flat;
      std::vector< std::vector<fst::StdArc::Label> > isymbols;
      std::vector< std::vector<fst::StdArc::Label> > osymbols;
      FstExpandWithBreaks(lat, break_symbols, &flat, &isymbols, &osymbols);
      // Scale Lattice back to the original scale
      fst::ScaleLattice(inv_scale, &flat);
      // Write CompactLattice
      lattice_writer.Write(key, flat);
      // Write output symbols
      symbols_writer.Write(key, osymbols);
      n_done++;
    }
    KALDI_LOG << "Done " << n_done << " lattices, errors on " << n_error;
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
