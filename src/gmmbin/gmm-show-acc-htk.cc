// gmmbin/gmm-show-acc-htk.cc

// Copyright 2015 Joan Puigcerver

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

#include "util/common-utils.h"
#include "gmm/mle-am-diag-gmm.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Show occupancy statistics for each HMM state as HTK would"
        "Usage: gmm-sum-accs [options] <mdl> <stats-in>\n"
        "E.g.: gmm-sum-accs 1.mdl 1.acc\n";

    kaldi::ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        stats_in_filename = po.GetArg(2);

    kaldi::TransitionModel trans_model;
    {
      bool binary;
      kaldi::Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
    }

    kaldi::Vector<double> transition_accs;
    {
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      transition_accs.Read(ki.Stream(), binary_read, true);
    }

    kaldi::Vector<double> phone_state_accs;
    std::vector<double> acc_per_state(trans_model.NumTransitionStates() + 1);
    for (int32 i = 1; i < transition_accs.Dim(); ++i) {
      const double a = transition_accs(i);
      const int32 ts = trans_model.TransitionIdToTransitionState(i);
      acc_per_state[ts] += a;
    }

    for (size_t i = 1; i < acc_per_state.size(); ++i) {
      const int32 phone = trans_model.TransitionStateToPhone(i);
      const int32 hmmst = trans_model.TransitionStateToHmmState(i);
      if (hmmst == 0) {
        if (phone > 1) std::cout << std::endl;
        std::cout << phone;
      }
      std::cout << " " << acc_per_state[i];
    }
    std::cout << std::endl;



  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
