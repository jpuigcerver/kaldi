#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace kaldi {

void AddInsPenToLattice(BaseFloat penalty, Lattice *lat,
                        bool output_penalty = true) {
  for (int32 state = 0; state < lat->NumStates(); ++state) {
    for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
         aiter.Next()) {
      LatticeArc arc(aiter.Value());
      if ((output_penalty && arc.olabel != 0) ||
          (!output_penalty && arc.ilabel != 0)) {
        LatticeWeight weight = arc.weight;
        weight.SetValue1(weight.Value1() + penalty);
        arc.weight = weight;
        aiter.SetValue(arc);
      }
    }
  }
}

template <typename Arc>
double ComputeLikelihood(const fst::Fst<Arc>& fst) {
  std::vector<typename Arc::Weight> state_likelihoods;
  fst::ShortestDistance(fst, &state_likelihoods);
  typename Arc::Weight total_likelihood = Arc::Weight::Zero();
  for (typename Arc::StateId s = 0; s < state_likelihoods.size(); ++s) {
    total_likelihood = fst::Times(total_likelihood, state_likelihoods[s]);
  }
  return total_likelihood.Value();
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


    std::vector<fst::VectorFst<fst::StdArc>*> query_std_fsts;
    std::vector<fst::VectorFst<fst::LogArc>*> query_log_fsts;
    std::vector<std::string> query_keys;
    if (query_is_table) {
      SequentialTableReader<fst::VectorFstHolder> query_reader(query_in_str);
      for (; !query_reader.Done(); query_reader.Next()) {
        query_keys.push_back(query_reader.Key());
        if (use_log) {
          query_log_fsts.push_back(new fst::VectorFst<fst::LogArc>());
          fst::ArcMap(query_reader.Value(), query_log_fsts.back(),
                      fst::WeightConvertMapper<fst::StdArc, fst::LogArc>());
        } else {
          query_std_fsts.push_back(new fst::VectorFst<fst::StdArc>());
          *query_std_fsts.back() = query_reader.Value();
        }
        query_reader.FreeCurrent();
      }
    } else {
      if (use_log) {
        fst::VectorFst<fst::StdArc> tmp;
        fst::ReadFstKaldi(query_in_str, &tmp);
        query_log_fsts.push_back(new fst::VectorFst<fst::LogArc>());
        fst::ArcMap(tmp, query_log_fsts.back(),
                    fst::WeightConvertMapper<fst::StdArc, fst::LogArc>());
        fst::ArcSort(query_log_fsts.back(), fst::ILabelCompare<fst::LogArc>());
      } else {
        query_std_fsts.push_back(new fst::VectorFst<fst::StdArc>());
        fst::ReadFstKaldi(query_in_str, query_std_fsts.back());
        fst::ArcSort(query_std_fsts.back(), fst::ILabelCompare<fst::StdArc>());
      }
    }

    kaldi::TableWriter<kaldi::BasicHolder<double>> table_writer;

    if (lattice_is_table) {
      SequentialLatticeReader lattice_reader;
      for (; !lattice_reader.Done(); lattice_reader.Next()) {
        const std::string lattice_key = lattice_reader.Key();
        fst::VectorFst<fst::StdArc> lattice_std_fst;
        fst::VectorFst<fst::LogArc> lattice_log_fst;
        {
          Lattice lat = lattice_reader.Value();
          lattice_reader.FreeCurrent();
          // Acoustic scale
          if (acoustic_scale != 1.0)
            fst::ScaleLattice(fst::AcousticLatticeScale(acoustic_scale), &lat);
          // Word insertion penalty
          if (insertion_penalty != 0.0)
            AddInsPenToLattice(insertion_penalty, &lat);
          // Lattice prunning
          if (beam != std::numeric_limits<BaseFloat>::infinity())
            PruneLattice(beam, &lat);
          // Convert lattice to FST
          fst::ConvertLattice(lat, &lattice_std_fst);
          if (use_log) {
            fst::ArcMap(lattice_std_fst, &lattice_log_fst,
                        fst::WeightConvertMapper<fst::StdArc, fst::LogArc>());
            lattice_std_fst.DeleteStates();
            fst::ArcSort(&lattice_log_fst, fst::OLabelCompare<fst::LogArc>());
          } else {
            fst::ArcSort(&lattice_std_fst, fst::OLabelCompare<fst::StdArc>());
          }
        }
        // Compute total log-likelihood of the lattice
        const double lattice_likelihood = use_log ?
            ComputeLikelihood(lattice_log_fst) :
            ComputeLikelihood(lattice_std_fst);
        // Compute the log-likelihood of each of the queries
        const size_t num_queries = std::max(query_log_fsts.size(),
                                            query_std_fsts.size());
        for (size_t i = 0; i < num_queries; ++i) {
          const double query_likelihood = use_log ?
              ComputeLikelihood(
                  fst::ComposeFst<fst::LogArc>(lattice_log_fst,
                                               *query_log_fsts[i])) :
              ComputeLikelihood(
                  fst::ComposeFst<fst::StdArc>(lattice_std_fst,
                                               *query_std_fsts[i]));
          if (i < query_keys.size()) {
            table_writer.Write(lattice_key + "____" + query_keys[i],
                               query_likelihood - lattice_likelihood);
          } else {
            table_writer.Write(lattice_key,
                               query_likelihood - lattice_likelihood);
          }
        }
      }
    } else {
      KALDI_ERR << "NOT IMPLEMENTED!";
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
