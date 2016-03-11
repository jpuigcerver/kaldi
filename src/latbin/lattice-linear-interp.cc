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

namespace fst {

template <typename Arc>
void LinearScaleFst(const typename Arc::Weight& weight, MutableFst<Arc>* fst) {
  KALDI_ASSERT(fst != NULL);
  KALDI_ASSERT(fst->Start() != kNoStateId);
  for (MutableArcIterator< MutableFst<Arc> > aiter(fst, fst->Start());
       !aiter.Done(); aiter.Next()) {
    Arc arc = aiter.Value();
    arc.weight = Times(arc.weight, weight);
    aiter.SetValue(arc);
  }
}

template <typename FloatType>
inline int Compare(const LogWeightTpl<FloatType>& w1,
                   const LogWeightTpl<FloatType>& w2) {
  if (w1.Value() == w2.Value()) return 0;
  else if (w1.Value() > w2.Value()) return -1;
  else return 1;
}

template <typename T, typename IntType>
inline CompactLatticeWeightTpl<LogWeightTpl<T>, IntType> Plus(
    const CompactLatticeWeightTpl<LogWeightTpl<T>, IntType>& w1,
    const CompactLatticeWeightTpl<LogWeightTpl<T>, IntType>& w2) {
  if (Compare(w1, w2) >= 0) {
    return CompactLatticeWeightTpl<LogWeightTpl<T>, IntType>(
        Plus(w1.Weight(), w2.Weight()), w1.String());
  } else {
    return CompactLatticeWeightTpl<LogWeightTpl<T>, IntType>(
        Plus(w1.Weight(), w2.Weight()), w2.String());
  }
}

template <typename FloatType, typename IntType>
void ConvertLatticeWeight(
    const CompactLatticeWeightTpl<LatticeWeightTpl<FloatType>, IntType>& w_in,
    CompactLatticeWeightTpl<LogWeightTpl<FloatType>, IntType>* w_out) {
  w_out->SetWeight(LogWeightTpl<FloatType>(
      w_in.Weight().Value1() + w_in.Weight().Value2()));
  w_out->SetString(w_in.String());
}

template <typename FloatType, typename IntType>
void ConvertLatticeWeight(
    const CompactLatticeWeightTpl<LogWeightTpl<FloatType>, IntType>& w_in,
    CompactLatticeWeightTpl<LatticeWeightTpl<FloatType>, IntType>* w_out) {
  w_out->SetWeight(LatticeWeightTpl<FloatType>(w_in.Weight().Value(), 0.0));
  w_out->SetString(w_in.String());
}

template <>
uint64 CompactLatticeWeightTpl<LogWeight, kaldi::int32>::Properties() {
  return kLeftSemiring | kRightSemiring;
}

typedef CompactLatticeWeightTpl<LogWeight, kaldi::int32>
CustomCompactLatticeWeight;

typedef ArcTpl<CustomCompactLatticeWeight> CustomCompactLatticeArc;

typedef VectorFst<CustomCompactLatticeArc> CustomCompactLattice;

}  // namespace fst

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Takes two archives of lattices (indexed by utterances) and linearly\n"
        "interpolates the individual lattice pairs (one from each archive).\n"
        "You can control the interpolation weights with --alpha, which is the\n"
        "weight of the first lattices (the second is 1-alpha).\n"
        "The interpolated lattice is determinized (and optionally minimized)\n"
        "before being written. The determinization algorithm keeps the best\n"
        "alignment for each sequence of output symbols (typically, words).\n\n"
        "Usage: lattice-linear-interp [options] lattice-rspecifier-a "
        "lattice-rspecifier-b lattice-wspecifier\n"
        "  e.g.: lattice-linear-interp ark:1.lats ark:2.lats ark:interp.lats\n";

    ParseOptions po(usage);
    BaseFloat alpha = 0.5; // Scale of 1st in the pair.
    BaseFloat acoustic_scale1 = 1.0;
    BaseFloat acoustic_scale2 = 1.0;
    BaseFloat lm_scale1 = 1.0;
    BaseFloat lm_scale2 = 1.0;
    BaseFloat acoustic2lm_scale1 = 0.0;
    BaseFloat acoustic2lm_scale2 = 0.0;
    BaseFloat lm2acoustic_scale1 = 0.0;
    BaseFloat lm2acoustic_scale2 = 0.0;
    bool minimize = true;

    po.Register("alpha", &alpha,
                "Scale of the first lattice in the pair. It must be in the "
                "range (0, 1)");
    po.Register("minimize", &minimize,
                "If true, push and minimize after determinization");
    po.Register("acoustic-scale1", &acoustic_scale1,
                "Acoustic costs scaling factor of the first lattice");
    po.Register("acoustic-scale2", &acoustic_scale2,
                "Acoustic costs scaling factor of the second lattice");
    po.Register("lm-scale1", &lm_scale1,
                "Graph/lm costs scaling factor of the first lattice");
    po.Register("lm-scale2", &lm_scale2,
                "Graph/lm costs scaling factor of the second lattice");
    po.Register("acoustic2lm-scale1", &acoustic2lm_scale1,
                "Add this times original acoustic costs to LM costs "
                "of the first lattice");
    po.Register("acoustic2lm-scale2", &acoustic2lm_scale2,
                "Add this times original acoustic costs to LM costs "
                "of the second lattice");
    po.Register("lm2acoustic-scale1", &lm2acoustic_scale1,
                "Add this times original LM costs to acoustic costs "
                "of the first lattice");
    po.Register("lm2acoustic-scale2", &lm2acoustic_scale2,
                "Add this times original LM costs to acoustic costs "
                "of the second lattice");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    KALDI_ASSERT(alpha > 0.0 && alpha < 1.0);

    std::vector<std::vector<double> > scale1(2);
    scale1[0].resize(2);
    scale1[1].resize(2);
    scale1[0][0] = lm_scale1;
    scale1[0][1] = acoustic2lm_scale1;
    scale1[1][0] = lm2acoustic_scale1;
    scale1[1][1] = acoustic_scale1;
    std::vector<std::vector<double> > scale2(2);
    scale2[0].resize(2);
    scale2[1].resize(2);
    scale2[0][0] = lm_scale2;
    scale2[0][1] = acoustic2lm_scale2;
    scale2[1][0] = lm2acoustic_scale2;
    scale2[1][1] = acoustic_scale2;

    std::string lats_rspecifier1 = po.GetArg(1),
        lats_rspecifier2 = po.GetArg(2),
        lats_wspecifier = po.GetArg(3);

    SequentialCompactLatticeReader lattice_reader1(lats_rspecifier1);
    RandomAccessCompactLatticeReader lattice_reader2(lats_rspecifier2);

    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_processed=0, n_success=0, n_no_2ndlat=0;

    for (; !lattice_reader1.Done(); lattice_reader1.Next()) {
      std::string key = lattice_reader1.Key();
      // Read first compact lattice, convert it to the custom format and
      // linearly scale with alpha.
      fst::CustomCompactLattice cclat1;
      {
        CompactLattice temp = lattice_reader1.Value();
        lattice_reader1.FreeCurrent();
        fst::ScaleLattice(scale1, &temp);
        ConvertLattice(temp, &cclat1);
        fst::LinearScaleFst(fst::CustomCompactLatticeWeight(
            fst::LogWeight(-log(alpha)),
            std::vector<kaldi::int32>()), &cclat1);
      }

      if (lattice_reader2.HasKey(key)) {
        ++n_processed;
        // Read second compact lattice, convert it to the custom format and
        // linearly scale it with 1.0 - alpha.
        fst::CustomCompactLattice cclat2;
        {
          CompactLattice temp = lattice_reader2.Value(key);
          fst::ScaleLattice(scale2, &temp);
          ConvertLattice(temp, &cclat2);
          fst::LinearScaleFst(fst::CustomCompactLatticeWeight(
              fst::LogWeight(-log(1.0-alpha)),
              std::vector<kaldi::int32>()), &cclat2);
        }
        // Deterministic union of the two custom and scaled lattices,
        // to create the linearly interpolated lattice. Optionally, also
        // minimizes the output lattice.
        CompactLattice lat3;
        {
          fst::CustomCompactLattice temp, temp2;
          fst::Determinize(
              fst::UnionFst<fst::CustomCompactLatticeArc>(cclat1, cclat2),
              &temp);
          if (minimize) {
            fst::Minimize(&temp, &temp2);
          }
          ConvertLattice(temp2, &lat3);
        }

        compact_lattice_writer.Write(key, lat3);
        ++n_success;
      } else {
        KALDI_WARN << "No lattice found for utterance " << key << " in "
                   << lats_rspecifier2 << ". Not producing output";
        ++n_no_2ndlat;
      }
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
