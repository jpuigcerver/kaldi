#undef NDEBUG   // Tests cannot run if NDEBUG flag is defined

#include "fstext/forward-backward-fst.h"
#include "fst/float-weight.h"
#include "util/kaldi-io.h"

typedef fst::ForwardBackwardFst<fst::StdArc> Fst;

void CreateArbitraryFst(fst::MutableFst<fst::StdArc>* fst) {
  fst->DeleteStates();
  for (int i = 0; i < 5; ++i) {
    const fst::StdArc::StateId s = fst->AddState();
    KALDI_ASSERT(s == i);
  }
  fst->AddArc(0, fst::StdArc(0, 0, 0.3, 0));   // loop 0->0
  fst->AddArc(0, fst::StdArc(1, 2, 0.0, 0));   // loop 0->0
  fst->AddArc(0, fst::StdArc(2, 1, 1.2, 1));   // 0->1
  fst->AddArc(1, fst::StdArc(1, 0, 0.1, 1));   // loop 1->1
  fst->AddArc(1, fst::StdArc(0, 2, -0.1, 2));  // 1->2
  fst->AddArc(2, fst::StdArc(1, 1, -0.1, 3));  // 2->3
  fst->AddArc(3, fst::StdArc(0, 0, 0.0, 3));   // loop 3->3
  fst->AddArc(3, fst::StdArc(0, 1, 0.0, 4));   // 3->4
  fst->AddArc(3, fst::StdArc(0, 2, 0.0, 4));   // 3->4
  fst->AddArc(4, fst::StdArc(0, 0, 0.0, 1));   // 4->1
  fst->SetStart(0);
  fst->SetFinal(0, 1.0);
  fst->SetFinal(2, 2.0);
  fst->SetFinal(4, -1.0);
}

void TestForwardBackwardArc() {
  // Test empty constructor
  {
    fst::ForwardBackwardArc<fst::LogArc> arc;
    arc.weight = fst::LogWeight::One();
    arc.ilabel = 1;
    arc.olabel = 2;
    arc.nextstate = 3;
    arc.prevstate = 4;
  }
  // Test constructor from parent class object
  {
    fst::StdArc std_arc(1, 2, fst::TropicalWeight::One(), 3);
    fst::ForwardBackwardArc<fst::StdArc> fb_arc(std_arc);
    fb_arc.prevstate = 4;
    KALDI_ASSERT(fb_arc.weight == fst::TropicalWeight::One());
    KALDI_ASSERT(fb_arc.ilabel == 1);
    KALDI_ASSERT(fb_arc.olabel == 2);
    KALDI_ASSERT(fb_arc.nextstate == 3);
    KALDI_ASSERT(fb_arc.prevstate == 4);
  }
  // Test constructor receiveing parent class object + prevstate
  {
    fst::StdArc std_arc(1, 2, fst::TropicalWeight::One(), 3);
    fst::ForwardBackwardArc<fst::StdArc> fb_arc(std_arc, 4);
    KALDI_ASSERT(fb_arc.weight == fst::TropicalWeight::One());
    KALDI_ASSERT(fb_arc.ilabel == 1);
    KALDI_ASSERT(fb_arc.olabel == 2);
    KALDI_ASSERT(fb_arc.nextstate == 3);
    KALDI_ASSERT(fb_arc.prevstate == 4);
  }

}

void TestForwardBackwardFst(const Fst& fst) {
  // Check number of states
  KALDI_ASSERT(fst.NumStates() == 5);
  // Check start state
  KALDI_ASSERT(fst.Start() == 0);
  // Check final probs
  KALDI_ASSERT(fst.Final(0) == 1.0);
  KALDI_ASSERT(fst.Final(1) == fst::StdArc::Weight::Zero());
  KALDI_ASSERT(fst.Final(2) == 2.0);
  KALDI_ASSERT(fst.Final(3) == fst::StdArc::Weight::Zero());
  KALDI_ASSERT(fst.Final(4) == -1.0);
  // Check number of input/output arcs
  KALDI_ASSERT(fst.NumArcs<false>(0) == 3);
  KALDI_ASSERT(fst.NumArcs<true>(0) == 2);
  KALDI_ASSERT(fst.NumArcs<false>(1) == 2);
  KALDI_ASSERT(fst.NumArcs<true>(1) == 3);
  KALDI_ASSERT(fst.NumArcs<false>(2) == 1);
  KALDI_ASSERT(fst.NumArcs<true>(2) == 1);
  KALDI_ASSERT(fst.NumArcs<false>(3) == 3);
  KALDI_ASSERT(fst.NumArcs<true>(3) == 2);
  KALDI_ASSERT(fst.NumArcs<false>(4) == 1);
  KALDI_ASSERT(fst.NumArcs<true>(4) == 2);
  // Check number of input epsilons in the input/output arcs of each node
  KALDI_ASSERT(fst.NumInputEpsilons<false>(0) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<true>(0) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<false>(1) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<true>(1) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<false>(2) == 0);
  KALDI_ASSERT(fst.NumInputEpsilons<true>(2) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<false>(3) == 3);
  KALDI_ASSERT(fst.NumInputEpsilons<true>(3) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<false>(4) == 1);
  KALDI_ASSERT(fst.NumInputEpsilons<true>(4) == 2);
  // Check number of output epsilons in the input/output arcs of each node
  KALDI_ASSERT(fst.NumOutputEpsilons<false>(0) == 1);
  KALDI_ASSERT(fst.NumOutputEpsilons<true>(0) == 1);
  KALDI_ASSERT(fst.NumOutputEpsilons<false>(1) == 1);
  KALDI_ASSERT(fst.NumOutputEpsilons<true>(1) == 2);
  KALDI_ASSERT(fst.NumOutputEpsilons<false>(2) == 0);
  KALDI_ASSERT(fst.NumOutputEpsilons<true>(2) == 0);
  KALDI_ASSERT(fst.NumOutputEpsilons<false>(3) == 1);
  KALDI_ASSERT(fst.NumOutputEpsilons<true>(3) == 1);
  KALDI_ASSERT(fst.NumOutputEpsilons<false>(4) == 1);
  KALDI_ASSERT(fst.NumOutputEpsilons<true>(4) == 0);
  // Check state and arc iterators
  {
    size_t ns = 0;
    for (fst::StateIterator<Fst> siter(fst); !siter.Done(); siter.Next(),
             ++ns) {
      const int s = siter.Value();
      KALDI_ASSERT(
          (s == 0 && fst.Final(s) == 1.0) ||
          (s == 1 && fst.Final(s) == fst::StdArc::Weight::Zero()) ||
          (s == 2 && fst.Final(s) == 2.0) ||
          (s == 3 && fst.Final(s) == fst::StdArc::Weight::Zero()) ||
          (s == 4 && fst.Final(s) == -1.0));
      // Forward arc iterator
      size_t na = 0;
      for (fst::ArcIterator<Fst> aiter(fst, s); !aiter.Done(); aiter.Next(),
               ++na) {
        const fst::StdArc& arc = aiter.Value();
        const int ns = arc.nextstate;
        const int il = arc.ilabel;
        const int ol = arc.olabel;
        const float w = arc.weight.Value();
        KALDI_ASSERT(
            (s == 0 && ns == 0 && il == 0 && ol == 0 && w == 0.3f) ||
            (s == 0 && ns == 0 && il == 1 && ol == 2 && w == 0.0f) ||
            (s == 0 && ns == 1 && il == 2 && ol == 1 && w == 1.2f) ||
            (s == 1 && ns == 1 && il == 1 && ol == 0 && w == 0.1f) ||
            (s == 1 && ns == 2 && il == 0 && ol == 2 && w == -0.1f) ||
            (s == 2 && ns == 3 && il == 1 && ol == 1 && w == -0.1f) ||
            (s == 3 && ns == 3 && il == 0 && ol == 0 && w == 0.0f) ||
            (s == 3 && ns == 4 && il == 0 && ol == 1 && w == 0.0f) ||
            (s == 3 && ns == 4 && il == 0 && ol == 2 && w == 0.0f) ||
            (s == 4 && ns == 1 && il == 0 && ol == 0 && w == 0.0f));
      }
      KALDI_ASSERT(
          (s == 0 && na == 3) || (s == 1 && na == 2) || (s == 2 && na == 1) ||
          (s == 3 && na == 3) || (s == 4 && na == 1));
      // Backward arc iterator
      na = 0;
      for (fst::ArcIterator<Fst> aiter(fst, s, true); !aiter.Done();
           aiter.Next(), ++na) {
        const fst::ForwardBackwardArc<fst::StdArc>& arc = aiter.Value();
        const int ps = arc.prevstate;
        const int il = arc.ilabel;
        const int ol = arc.olabel;
        const float w = arc.weight.Value();
        KALDI_ASSERT(
            (ps == 0 && s == 0 && il == 0 && ol == 0 && w == 0.3f) ||
            (ps == 0 && s == 0 && il == 1 && ol == 2 && w == 0.0f) ||
            (ps == 0 && s == 1 && il == 2 && ol == 1 && w == 1.2f) ||
            (ps == 1 && s == 1 && il == 1 && ol == 0 && w == 0.1f) ||
            (ps == 1 && s == 2 && il == 0 && ol == 2 && w == -0.1f) ||
            (ps == 2 && s == 3 && il == 1 && ol == 1 && w == -0.1f) ||
            (ps == 3 && s == 3 && il == 0 && ol == 0 && w == 0.0f) ||
            (ps == 3 && s == 4 && il == 0 && ol == 1 && w == 0.0f) ||
            (ps == 3 && s == 4 && il == 0 && ol == 2 && w == 0.0f) ||
            (ps == 4 && s == 1 && il == 0 && ol == 0 && w == 0.0f));
      }
      KALDI_ASSERT(
          (s == 0 && na == 2) || (s == 1 && na == 3) || (s == 2 && na == 1) ||
          (s == 3 && na == 2) || (s == 4 && na == 2));
    }
    KALDI_ASSERT(ns == 5);
  }
}

int main(int argc, char** argv) {
  TestForwardBackwardArc();

  Fst fst;
  CreateArbitraryFst(&fst);
  TestForwardBackwardFst(fst);
  std::cout << "Test OK!" << std::endl;
  return 0;
}
