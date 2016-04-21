// gmmbin/gmm-latgen-faster.cc

// Copyright 2009-2012  Microsoft Corporation
//           2013-2014  Johns Hopkins University (author: Daniel Povey)
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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "gmm/decodable-am-diag-gmm.h"
#include "base/timer.h"
#include "feat/feature-functions.h"  // feature reversal
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"

namespace fst {

// Same as Kaldi's TableCompose but returns an On-Demand FST
template<class Arc>
ComposeFst<Arc> TableComposeFst(
    const Fst<Arc> &ifst1, const Fst<Arc> &ifst2,
    const CacheOptions& cache_opts = CacheOptions(),
    const TableComposeOptions &opts = TableComposeOptions()) {
  typedef Fst<Arc> F;
  if (opts.table_match_type == MATCH_OUTPUT) {
    // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
    ComposeFstImplOptions<TableMatcher<F>, SortedMatcher<F> >
        impl_opts(cache_opts);
    impl_opts.matcher1 = new TableMatcher<F>(ifst1, MATCH_OUTPUT, opts);
    return ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  } else {
    assert(opts.table_match_type == MATCH_INPUT) ;
    // ComposeFstImplOptions templated on matcher for fst1, matcher for fst2.
    ComposeFstImplOptions<SortedMatcher<F>, TableMatcher<F> >
        impl_opts(cache_opts);
    impl_opts.matcher2 = new TableMatcher<F>(ifst2, MATCH_INPUT, opts);
    return ComposeFst<Arc>(ifst1, ifst2, impl_opts);
  }
}

}  // namespace fst

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices using GMM-based model.\n"
        "User supplies the HCL and the G transducers independently and the "
        "composition is done on-demand during the decoding.\n"
        "This is very useful when the language model is very big and HCLG "
        "would result in a too large FST.\n\n"
        "Usage: gmm-latgen-lazylm-faster [options] model-in "
        "(hcl-fst|hcl-rspecifier) (lm-fst|lm-rspecifier) "
        "features-rspecifier lattice-wspecifier "
        "[ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    LatticeFasterDecoderConfig decoder_config;
    fst::TableComposeOptions compose_config;
    fst::CacheOptions cache_config;
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    int gc_limit = 536870912;  // 512MB
    std::string match_side = "left";
    std::string compose_filter = "sequence";
    std::string word_syms_filename;
    decoder_config.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("word-symbol-table", &word_syms_filename,
                "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial,
                "If true, produce output even if end state was not reached.");
    po.Register("match-side", &match_side, "Side of composition to do table "
                "match, one of: \"left\" or \"right\".");
    po.Register("compose-filter", &compose_filter, "Composition filter to use, "
                "one of: \"alt_sequence\", \"auto\", \"match\", \"sequence\"");
    po.Register("compose-gc", &cache_config.gc,
                "If false, any expanded state during the composition will be "
                "cached. If true, the cache will be garbage collected when it "
                "grows past --compose-gc-limit bytes.");
    po.Register("compose-gc-limit", &gc_limit,
                "Number of bytes allowed in the composition cache before "
                "garbage collection.");
    po.Read(argc, argv);
    cache_config.gc_limit = gc_limit;

    if (match_side == "left") {
      compose_config.table_match_type = fst::MATCH_OUTPUT;
    } else if (match_side == "right") {
      compose_config.table_match_type = fst::MATCH_INPUT;
    } else {
      KALDI_ERR << "Invalid match-side option: " << match_side;
    }

    if (compose_filter == "alt_sequence") {
      compose_config.filter_type = fst::ALT_SEQUENCE_FILTER;
    } else if (compose_filter == "auto") {
      compose_config.filter_type = fst::AUTO_FILTER;
    } else  if (compose_filter == "match") {
      compose_config.filter_type = fst::MATCH_FILTER;
    } else  if (compose_filter == "sequence") {
      compose_config.filter_type = fst::SEQUENCE_FILTER;
    } else {
      KALDI_ERR << "Invalid compose-filter option: " << compose_filter;
    }

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        hcl_in_str = po.GetArg(2),
        lm_in_str =  po.GetArg(3),
        feature_rspecifier = po.GetArg(4),
        lattice_wspecifier = po.GetArg(5),
        words_wspecifier = po.GetOptArg(6),
        alignment_wspecifier = po.GetOptArg(7);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    bool determinize = decoder_config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_done = 0, num_err = 0;

    const bool is_table_hcl =
        ClassifyRspecifier(hcl_in_str, NULL, NULL) != kNoRspecifier;
    const bool is_table_lm =
        ClassifyRspecifier(lm_in_str, NULL, NULL) != kNoRspecifier;

    if (!is_table_hcl && !is_table_lm) {
      // Input FSTs are just two FSTs, not two tables of FSTs.
      VectorFst<StdArc> *hcl_fst = fst::ReadFstKaldi(hcl_in_str);
      VectorFst<StdArc> *lm_fst = fst::ReadFstKaldi(lm_in_str);
      // On-demand composition of HCL and G
      fst::ComposeFst<StdArc> decode_fst = fst::TableComposeFst(
          *hcl_fst, *lm_fst, cache_config, compose_config);

      LatticeFasterDecoder decoder(decode_fst, decoder_config);

      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        Matrix<BaseFloat> features (feature_reader.Value());
        feature_reader.FreeCurrent();
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                               acoustic_scale);
        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, gmm_decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          num_done++;
        } else num_err++;
      }
    } else if (is_table_hcl && is_table_lm) {
      // Both HCL and LM FSTs are actually tables of FSTs
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      RandomAccessTableReader<fst::VectorFstHolder> hcl_reader(hcl_in_str);
      RandomAccessTableReader<fst::VectorFstHolder> lm_reader(lm_in_str);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        if (!hcl_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no HCL is available.";
          num_err++;
          continue;
        }
        if (!lm_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no G is available.";
          num_err++;
          continue;
        }

        const Matrix<BaseFloat> features(feature_reader.Value());
        feature_reader.FreeCurrent();
        if (features.NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_err++;
          continue;
        }

        // On-demand composition of HCL and G
        fst::ComposeFst<StdArc> decode_fst = fst::TableComposeFst(
            hcl_reader.Value(utt), lm_reader.Value(utt), cache_config,
            compose_config);

        LatticeFasterDecoder decoder(decode_fst, decoder_config);
        DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                               acoustic_scale);
        double like;
        if (DecodeUtteranceLatticeFaster(
                decoder, gmm_decodable, trans_model, word_syms, utt,
                acoustic_scale, determinize, allow_partial, &alignment_writer,
                &words_writer, &compact_lattice_writer, &lattice_writer,
                &like)) {
          tot_like += like;
          frame_count += features.NumRows();
          num_done++;
        } else num_err++;
      }
    } else {
      KALDI_ERR << "The decoding of tables/non-tables and match-type that you "
                << "supplied is not currently supported. Either implement "
                << "this, ask the maintainers to implement it, or call this "
                << "program differently.";
    }

    const double elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_done << " utterances, failed for "
              << num_err;
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like/frame_count) << " over "
              << frame_count << " frames.";

    delete word_syms;
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
