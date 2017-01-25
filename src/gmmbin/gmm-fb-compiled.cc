#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "fstext/fstext-lib.h"
#include "gmm/decodable-am-diag-gmm.h"

#include "fb/simple-common.h"
#include "fb/fast-forward-backward.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::VectorFst;
    using fst::StdArc;
    typedef fst::StdArc::Label Label;

    const char *usage =
        "gmm-fb-compiled 1.mdl ark:graphs.fsts scp:train.scp ark:1.psts.ark\n";

    ParseOptions po(usage);
    BaseFloat beam_fwd = std::numeric_limits<BaseFloat>::infinity();
    BaseFloat beam_bkw = std::numeric_limits<BaseFloat>::infinity();
    BaseFloat delta = 0.000976562; // same delta as in fstshortestdistance
    BaseFloat acoustic_scale = 1.0;
    BaseFloat transition_scale = 1.0;
    BaseFloat self_loop_scale = 1.0;

    po.Register("beam-backward", &beam_bkw, "Beam prunning threshold during "
                "backward pass [sensitive, use a wide beam]");
    po.Register("beam-forward", &beam_fwd, "Beam prunning threshold during "
                "forward pass [non-sensitive, can use a narrow beam]");
    po.Register("delta", &delta, "Comparison delta [see fstshortestdistance]");
    po.Register("transition-scale", &transition_scale,
                "Transition-probability scale [relative to acoustics]");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("self-loop-scale", &self_loop_scale,
                "Scale of self-loop versus non-self-loop log probs "
                "[relative to acoustics]");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        fst_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        posterior_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmDiagGmm am_gmm;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

    SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_rspecifier);
    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    PosteriorWriter posterior_writer(posterior_wspecifier);

    int num_done = 0, num_err = 0;
    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string utt = fst_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        num_err++;
        KALDI_WARN << "No features for utterance " << utt;
      } else {
        const Matrix<BaseFloat> &features = feature_reader.Value(utt);
        fst::ForwardBackwardFst<StdArc> fst(fst_reader.Value());
        fst_reader.FreeCurrent();  // this stops copy-on-write of the fst
        // by deleting the fst inside the reader, since we're about to mutate
        // the fst by adding transition probs.

        {  // Add transition-probs to the FST.
          std::vector<int32> disambig_syms;  // empty.
          AddTransitionProbs(trans_model, disambig_syms,
                             transition_scale, self_loop_scale,
                             &fst);
        }

        FastForwardBackward fb(fst, beam_bkw, beam_fwd, delta);
        DecodableAmDiagGmm gmm_decodable(am_gmm, trans_model, features);

        if (!fb.ForwardBackward(&gmm_decodable)) {
          KALDI_WARN << "Forward-Backward failed for utt " << utt;
          ++num_err;
          continue;
        }
        const double lkh = fb.LogLikelihood();
        const int64 nfrm = fb.NumFramesDecoded();
        if (isinf(lkh)) {
          KALDI_WARN << "Forward-Backward gave an infinity likelihood for utt " << utt;
          ++num_err;
          continue;
        } else {
          tot_like += lkh;
          frame_count += nfrm;
          ++num_done;
        }

        const std::vector<LabelMap>& pst_map = fb.LabelPosteriors();
        Posterior pst(pst_map.size());
        for (size_t t = 0; t < pst.size(); ++t) {
          for(LabelMap::const_iterator l = pst_map[t].begin();
              l != pst_map[t].end(); ++l) {
            pst[t].push_back(*l);
            pst[t].back().second = exp(pst[t].back().second);
          }
        }
        posterior_writer.Write(utt, pst);
      }
    }
    KALDI_LOG << "Overall log-likelihood per frame is "
              << (tot_like / frame_count) << " over " << frame_count
              << " frames";
    KALDI_LOG << "Done " << num_done << ", errors on " << num_err;


  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }

}
