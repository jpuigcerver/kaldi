// fstbin/fstlinearinterp.cc

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
#ifndef _MSC_VER
#include <signal.h> // Comment this line and the call to signal below if
// it causes compilation problems.  It is only to enable a debugging procedure
// when determinization does not terminate.
#endif

bool debug_location = false;
void signal_handler(int) {
  debug_location = true;
}

namespace fst {

// Multiplies (in the apropiate semiring meaning) the outgoing edges of the
// initial state by some weight.
template <typename Arc>
void LinearScale(const typename Arc::Weight& weight, MutableFst<Arc>* fst) {
  KALDI_ASSERT(fst != NULL);
  KALDI_ASSERT(fst->Start() != kNoStateId);
  for (MutableArcIterator< MutableFst<Arc> > aiter(fst, fst->Start());
       !aiter.Done(); aiter.Next()) {
    Arc arc = aiter.Value();
    arc.weight = Times(arc.weight, weight);
    aiter.SetValue(arc);
  }
}

}  // namespace fst

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Reads two tables of FSTs and linearly interpolates the individual\n"
        "FST pairs (one from each table). The interpolation weight is set\n"
        "--alpha, which is the weight of the first FST (the second is 1.0 -\n"
        "alpha). The inputs can also be individual FSTs.\n"
        "The interpolated FST is determinized (and optionally minimized)\n"
        "before being written.\n\n"
        "Usage: fstlinearinterp [options] fst-rspecifier1 fst-rspecifier2 "
        "fst-wspecifier\n"
        "  e.g.: fstlinearinterp ark:1.fsts ark:2.fsts ark:interp.fsts\n"
        "  e.g.: fstlinearinterp a.fst b.fst out.fst\n";

    ParseOptions po(usage);
    BaseFloat alpha1 = 0.5; // Scale of 1st in the pair.
    bool minimize = true;
    BaseFloat delta = fst::kDelta;
    int max_states = -1;

    po.Register("alpha", &alpha1,
                "Scale of the first lattice in the pair. It must be in the "
                "range [0, 1]");
    po.Register("minimize", &minimize,
                "If true, push and minimize after determinization");
    po.Register("delta", &delta,
                "Delta value used to determine equivalence of weights.");
    po.Register("max-states", &max_states,
                "Maximum number of states in determinized FST before it will "
                "abort.");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    KALDI_ASSERT(alpha1 >= 0.0 && alpha1 <= 1.0);
    BaseFloat alpha2 = 1.0 - alpha1;

    std::string fsts_rspecifier1 = po.GetArg(1),
        fsts_rspecifier2 = po.GetArg(2),
        fsts_wspecifier = po.GetArg(3);

    if (ClassifyRspecifier(fsts_rspecifier1, NULL, NULL) == kNoRspecifier) {
      // Deal with interpolation between two fsts
      KALDI_ASSERT(ClassifyRspecifier(fsts_rspecifier2, NULL, NULL) ==
                   kNoRspecifier);
      // Read the two fsts
      fst::VectorFst<fst::StdArc>* fst1 = fst::ReadFstKaldi(fsts_rspecifier1);
      fst::VectorFst<fst::StdArc>* fst2 = fst::ReadFstKaldi(fsts_rspecifier2);
      if (alpha2 == 0.0) {
        delete fst2;
      } else if (alpha1 == 0.0) {
        delete fst1;
        fst1 = fst2;
      } else {
        // Linearly scale the two fsts
        LinearScale(fst::TropicalWeight(-log(alpha1)), fst1);
        LinearScale(fst::TropicalWeight(-log(alpha2)), fst2);
        // Caution: now fst1 is the union of the two linear-ly scaled FSTs
        fst::Union(fst1, *fst2);
        delete fst2;  // fst2 no longer needed
      }
      // Determinize the interpolated FSTS in the log semiring to sum-up
      // weights. CAUTION: This might fail if the union FST is not
      // determinizable! To be safe, make sure that the FSTs are acceptors
      // and you'll be fine.
      fst::DeterminizeStarInLog(fst1, delta, &debug_location, max_states);
      // Optionally, minimize the interpolated FST.
      if (minimize) {
        //fst::Minimize<fst::StdArc>(fst1, NULL, delta);
        fst::MinimizeEncoded(fst1, delta);
      }
      fst::WriteFstKaldi(*fst1, fsts_wspecifier);
      delete fst1;
    } else {
      // Deal with interpolation between archives.
      KALDI_ASSERT(ClassifyRspecifier(fsts_rspecifier2, NULL, NULL) !=
                   kNoRspecifier);
      SequentialTableReader<fst::VectorFstHolder> fst_reader1(fsts_rspecifier1);
      RandomAccessTableReader<fst::VectorFstHolder> fst_reader2(
          fsts_rspecifier2);
      TableWriter<fst::VectorFstHolder> fst_writer(fsts_wspecifier);
      int32 n_processed=0, n_success=0, n_no_2ndlat=0;
      for (; !fst_reader1.Done(); fst_reader1.Next()) {
        // Read first fst and linearly scale its costs.
        std::string key = fst_reader1.Key();
        fst::VectorFst<fst::StdArc> fst1 = fst_reader1.Value();
        fst_reader1.FreeCurrent();
        if (alpha1 > 0.0 && alpha2 > 0.0) {
          LinearScale(fst::TropicalWeight(-log(alpha1)), &fst1);
        }
        if (fst_reader2.HasKey(key)) {
          ++n_processed;
          // Read second fst, linearly scale its costs and compute the union
          // (linear interpolation) of the two fsts.
          if (alpha1 == 0.0) {
            fst1 = fst_reader2.Value(key);
          } else if (alpha2 > 0.0) {
            fst::VectorFst<fst::StdArc> fst2 = fst_reader2.Value(key);
            LinearScale(fst::TropicalWeight(-log(alpha2)), &fst2);
            // Caution: now fst1 is the union of the two linear-ly scaled FSTs
            fst::Union(&fst1, fst2);
          }
          // Determinize the interpolated FSTS in the log semiring to sum-up
          // weights. CAUTION: This might fail if the union FST is not
          // determinizable! To be safe, make sure that the FSTs are acceptors
          // and you'll be fine.
          fst::DeterminizeStarInLog(&fst1, delta, &debug_location, max_states);
          // Optionally, minimize the interpolated FST.
          if (minimize) {
            //fst::Minimize<fst::StdArc>(&fst1, NULL, delta);
            fst::MinimizeEncoded(&fst1, delta);
          }
          fst_writer.Write(key, fst1);
          ++n_success;
        } else {
          KALDI_WARN << "No lattice found for utterance " << key << " in "
                     << fsts_rspecifier2 << ". Not producing output";
          ++n_no_2ndlat;
        }
      }
      KALDI_LOG << "Done " << n_processed << " fsts; "
                << n_success << " had nonempty result; "
                << n_no_2ndlat << " had empty second lattice.";
      return (n_success != 0 ? 0 : 1);
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
