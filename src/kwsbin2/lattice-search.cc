#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/kaldi-fst-io.h"

namespace kaldi {

void AddInsPenToLattice(BaseFloat ins_penalty, Lattice *lat,
                        bool add_when_output = true) {
  for (int32 state = 0; state < lat->NumStates(); ++state) {
    for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
         aiter.Next()) {
      LatticeArc arc(aiter.Value());
      if ((add_when_output && arc.olabel != 0) ||
          (!add_when_output && arc.ilabel != 0)) {
        LatticeWeight weight = arc.weight.Weight();
        weight.SetValue1(weight.Value1() + ins_penalty);
        arc.weight.SetWeight(weight);
        aiter.SetValue(arc);
      }
    }
  }
}

}  // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage = "";


    ParseOptions po(usage);
    BaseFloat acoustic_scale = 1.0;
    BaseFloat graph_scale = 1.0;
    BaseFloat insertion_penalty = 0.0;
    BaseFloat beam = std::numeric_limits<BaseFloat>::infinity();
    bool use_log = true;

    po.Register("use-log", &use_log,
                "If true, compute scores using the log semiring (a.k.a. "
                "forward), otherwise use the tropical semiring (a.k.a. "
                "viterbi).");
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods in the lattices.");
    po.Register("graph-scale", &graph_scale,
                "Scaling factor for graph probabilities in the lattices.");
    po.Register("insertion-penalty", &insertion_penalty,
                "Add this penalty to the lattice arcs with non-epsilon output "
                "label (typically, equivalent to word insertion penalty).");
    po.Register("beam", &beam, "Pruning beam (applied after acoustic scaling "
                "and adding the insertion penalty).");
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    const std::string lattice_in_str = po.GetArg(1);
    const std::string query_in_str = po.GetArg(2);
    const bool lattice_is_table =
        (ClassifyRspecifier(lattice_in_str, NULL, NULL) != kNoRspecifier);
    const bool query_is_table =
        (ClassifyRspecifier(query_in_str, NULL, NULL) != kNoRspecifier);


    if (query_is_table) {
      SequentialAccessTableReader<fst::VectorFstHolder> query_reader(
          query_in_str);
      for (; !query_reader.Done(); query_reader.Next()) {
        const std::string key = query_reader.Key();
        const fst::VectorFst<StdArc> fst = query_reader.Value();
        query_reader.FreeCurrent();
      }
    } else {
    }

    if (lattice_is_table) {
      SequentialLatticeReader lattice_reader;
      for (; !lattice_reader.Done(); lattice_reader.Next()) {
        const std::string key = lattice_reader.Key();
        fst::VectorFst<StdArc> fst;
        {
          Lattice lat = lattice_reader.Value();
          lattice_reader.FreeCurrent();
          // Acoustic scale
          if (acoustic_scale != 1.0)
            fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
          // Word insertion penalty
          if (insertion_penalty != 0.0)
            kaldi::AddInsPenToLattice(insertion_penalty, &lat);
          // Lattice prunning
          if (beam != std::numeric_limits<BaseFloat>::infinity())
            fst::PruneLattice(beam, &lat);
          // Convert lattice to FST
          fst::ConvertLattice(lat, &fst);
        }


      }

    } else {

    }








    RandomAccessTableReader< VectorFstTplHolder<Arc> > index_reader(index_rspecifier);
    SequentialTableReader<VectorFstHolder> keyword_reader(keyword_rspecifier);
    TableWriter<BasicVectorHolder<double> > result_writer(result_wspecifier);

    // Index has key "global"
    KwsLexicographicFst index = index_reader.Value("global");

    // First we have to remove the disambiguation symbols. But rather than
    // removing them totally, we actually move them from input side to output
    // side, making the output symbol a "combined" symbol of the disambiguation
    // symbols and the utterance id's.
    // Note that in Dogan and Murat's original paper, they simply remove the
    // disambiguation symbol on the input symbol side, which will not allow us
    // to do epsilon removal after composition with the keyword FST. They have
    // to traverse the resulting FST.
    int32 label_count = 1;
    unordered_map<uint64, uint32> label_encoder;
    unordered_map<uint32, uint64> label_decoder;
    for (StateIterator<KwsLexicographicFst> siter(index); !siter.Done(); siter.Next()) {
      StateId state_id = siter.Value();
      for (MutableArcIterator<KwsLexicographicFst>
           aiter(&index, state_id); !aiter.Done(); aiter.Next()) {
        Arc arc = aiter.Value();
        // Skip the non-final arcs
        if (index.Final(arc.nextstate) == Weight::Zero())
          continue;
        // Encode the input and output label of the final arc, and this is the
        // new output label for this arc; set the input label to <epsilon>
        uint64 osymbol = EncodeLabel(arc.ilabel, arc.olabel);
        arc.ilabel = 0;
        if (label_encoder.find(osymbol) == label_encoder.end()) {
          arc.olabel = label_count;
          label_encoder[osymbol] = label_count;
          label_decoder[label_count] = osymbol;
          label_count++;
        } else {
          arc.olabel = label_encoder[osymbol];
        }
        aiter.SetValue(arc);
      }
    }
    ArcSort(&index, fst::ILabelCompare<Arc>());

    int32 n_done = 0;
    int32 n_fail = 0;
    for (; !keyword_reader.Done(); keyword_reader.Next()) {
      std::string key = keyword_reader.Key();
      VectorFst<StdArc> keyword = keyword_reader.Value();
      keyword_reader.FreeCurrent();

      // Process the case where we have confusion for keywords
      if (keyword_beam != -1) {
        Prune(&keyword, keyword_beam);
      }
      if (keyword_nbest != -1) {
        VectorFst<StdArc> tmp;
        ShortestPath(keyword, &tmp, keyword_nbest, true, true);
        keyword = tmp;
      }

      KwsLexicographicFst keyword_fst;
      KwsLexicographicFst result_fst;
      Map(keyword, &keyword_fst, VectorFstToKwsLexicographicFstMapper());
      Compose(keyword_fst, index, &result_fst);
      Project(&result_fst, PROJECT_OUTPUT);
      Minimize(&result_fst);
      ShortestPath(result_fst, &result_fst, n_best);
      RmEpsilon(&result_fst);

      // No result found
      if (result_fst.Start() == kNoStateId)
        continue;

      // Got something here
      double score;
      int32 tbeg, tend, uid;
      for (ArcIterator<KwsLexicographicFst>
           aiter(result_fst, result_fst.Start()); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();

        // We're expecting a two-state FST
        if (result_fst.Final(arc.nextstate) != Weight::One()) {
          KALDI_WARN << "The resulting FST does not have the expected structure for key " << key;
          n_fail++;
          continue;
        }

        uint64 osymbol = label_decoder[arc.olabel];
        uid = (int32)DecodeLabelUid(osymbol);
        tbeg = arc.weight.Value2().Value1().Value();
        tend = arc.weight.Value2().Value2().Value();
        score = arc.weight.Value1().Value();

        if (score < 0) {
          if (score < negative_tolerance) {
            KALDI_WARN << "Score out of expected range: " << score;
          }
          score = 0.0;
        }
        vector<double> result;
        result.push_back(uid);
        result.push_back(tbeg);
        result.push_back(tend);
        result.push_back(score);
        result_writer.Write(key, result);
      }

      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " keywords";
    if (strict == true)
      return (n_done != 0 ? 0 : 1);
    else
      return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
