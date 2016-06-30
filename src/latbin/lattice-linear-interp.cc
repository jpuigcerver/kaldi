// latbin/lattice-interp.cc

// Copyright 2016  Joan Puigcerver <joapuipe@gmail.com>

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
void AddToFinalCost(const typename Arc::Weight& weight,
                    MutableFst<Arc>* fst) {
  KALDI_ASSERT(fst != NULL);
  for (StateIterator< Fst<Arc> > siter(*fst); !siter.Done(); siter.Next()) {
    const typename Arc::Weight final_cost = fst->Final(siter.Value());
    if (final_cost != Arc::Weight::Zero()) {
      fst->SetFinal(siter.Value(), Times(final_cost, weight));
    }
  }
}

}  // namespace fst

bool debug_location = false;
void signal_handler(int) { debug_location = true; }

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Usage: lattice-linear-interp [options] lattice-rspecifier-a "
        "lattice-rspecifier-b lattice-wspecifier\n"
        "  e.g.: lattice-linear-interp ark:lat1.ark ark:lat2.ark ark:fst.ark\n";

    ParseOptions po(usage);
    BaseFloat alpha = 0.5; // Interpolation weight for lattice 1
    BaseFloat acoustic_scale1 = 1.0, acoustic_scale2 = 1.0;
    BaseFloat lm_scale1 = 1.0, lm_scale2 = 1.0;
    BaseFloat ins_penalty1 = 0.0, ins_penalty2 = 0.0;
    BaseFloat delta = fst::kDelta;
    BaseFloat beam = std::numeric_limits<BaseFloat>::infinity();
    int max_states = -1;
    bool minimize = true;

    po.Register("alpha", &alpha,
                "Scale of the first lattice in the pair. It must be in the "
                "range [0, 1]");
    po.Register("beam", &beam,
                "Pruning beam [applied after scaling and union]");
    po.Register("delta", &delta,
                "Delta value used to determine equivalence of weights");
    po.Register("max-states", &max_states,
                "Maximum number of states in determinized FST before it will "
                "abort");
    po.Register("minimize", &minimize,
                "If true, push and minimize after determinization");
    po.Register("acoustic-scale1", &acoustic_scale1,
                "Acoustic costs scaling factor of the first lattice");
    po.Register("acoustic-scale2", &acoustic_scale2,
                "Acoustic costs scaling factor of the second lattice");
    po.Register("lm-scale1", &lm_scale1,
                "Graph/LM costs scaling factor of the first lattice");
    po.Register("lm-scale2", &lm_scale2,
                "Graph/LM costs scaling factor of the second lattice");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    // Avoid numerical problems with 1.0 - alpha
    if (alpha < std::numeric_limits<float>::epsilon()) {
      alpha = 0.0f;
      KALDI_WARN << "--alpha is rounded to 0.0";
    } else if (alpha > (1.0f - std::numeric_limits<float>::epsilon())) {
      alpha = 1.0f;
      KALDI_WARN << "--alpha is rounded to 1.0";
    }

    std::vector<std::vector<double> > scale1{
      std::vector<double>{lm_scale1, 0.0},
      std::vector<double>{0.0, acoustic_scale1}};
    std::vector<std::vector<double> > scale2{
      std::vector<double>{lm_scale2, 0.0},
      std::vector<double>{0.0, acoustic_scale2}};

    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2),
        fsts_wspecifier = po.GetArg(3);

    SequentialCompactLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessCompactLatticeReader lattice_reader2(lats_rspecifier2);
    TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);

    int32 n_processed=0, n_success=0, n_no_2ndlat=0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      if (!lattice_reader2.HasKey(key)) {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats_rspecifier2 << ". Not producing output";
        ++n_no_2ndlat;
        continue;
      }
      ++n_processed;
      kaldi::CompactLattice clat1, clat2;
      fst::VectorFst<fst::StdArc> fst1, fst2;
      if (alpha > 0.0f) {
        clat1 = lattice_reader1.Value();
        lattice_reader1.FreeCurrent();
        // Add interpolation weight to lattice 1
        fst::AddToFinalCost(
            CompactLatticeWeight(
                LatticeWeight(-log(alpha), 0.0), std::vector<int>{}),
            &clat1);
        // Scale lattice 1
        fst::ScaleLattice(scale1, &clat1);
        kaldi::AddWordInsPenToCompactLattice(ins_penalty1, &clat1);
        // Convert lattice to
        ConvertLattice(clat1, &fst1);
        clat1.DeleteStates();              // Free memory
      }
      if (alpha < 1.0f) {
        clat2 = lattice_reader2.Value(key);
        // Add interpolation weight to lattice 2
        fst::AddToFinalCost(
            CompactLatticeWeight(
                LatticeWeight(-log(1.0f - alpha), 0.0), std::vector<int>{}),
            &clat2);
        // Scale lattice 2
        fst::ScaleLattice(scale2, &clat2);
        kaldi::AddWordInsPenToCompactLattice(ins_penalty2, &clat2);
        ConvertLattice(clat2, &fst2);
        clat2.DeleteStates();              // Free memory
      }
      fst::VectorFst<fst::StdArc>* fst_out;
      if (alpha > 0.0f && alpha < 1.0f) {
        fst_out = &fst1;
        fst::Union(fst_out, fst2);  // Modifies fst1, actually
      } else if (alpha == 1.0f) {
        fst_out = &fst1;
      } else if (alpha == 0.0f) {
        fst_out = &fst2;
      }
      if (beam < std::numeric_limits<BaseFloat>::infinity()) {
        const size_t ns0 = fst_out->NumStates(), na0 = fst::NumArcs(*fst_out);
        fst::Prune(fst_out, beam);
        const size_t ns1 = fst_out->NumStates(), na1 = fst::NumArcs(*fst_out);
        KALDI_VLOG(1) << "Utterance " << key << ": Union prunned from "
                      << ns0 << " states and " << na0 << " arcs to "
                      << ns1 << " states and " << na1 << " arcs.";
      }
      fst::DeterminizeStarInLog(fst_out, delta, &debug_location, max_states);
      if (minimize) { fst::MinimizeEncoded(fst_out, delta); }
      fst_writer.Write(key, *fst_out);
      ++n_success;
    }
    KALDI_LOG << "Done " << n_processed << " lattices; "
              << n_success << " had nonempty result; "
              << n_no_2ndlat << " had empty second lattice.";
    return (n_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
