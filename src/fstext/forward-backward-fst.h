// fb/forward-backward-fst.h

// Copyright 2015  Joan Puigcerver

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

#ifndef KALDI_FSTEXT_FORWARD_BACKWARD_FST_H_
#define KALDI_FSTEXT_FORWARD_BACKWARD_FST_H_

#include <string>
#include <vector>
using std::vector;

#include <fst/fstlib.h>
#include <fst/fst.h>
#include <fst/expanded-fst.h>
#include <fst/mutable-fst.h>
#include <fst/test-properties.h>

namespace fst {

template <class A, class U> class ForwardBackwardFst;
template <class F, class G> void Cast(const F &, G *);


/// Forward-Backward FST needs a special arc type that keeps track of both
/// input (nextstate) and output (prevstate) of each arc.
/// This arc has exactly the same attributes than other arcs, plus includes
/// the prevstate attribute.
template <class A>
struct ForwardBackwardArc : public A {
  typedef typename A::Weight Weight;
  typedef typename A::Label Label;
  typedef typename A::StateId StateId;

  ForwardBackwardArc() : A() {}
  explicit ForwardBackwardArc(const A& arc) : A(arc) {}
  ForwardBackwardArc(const A& arc, StateId p)
      : A(arc), prevstate(p) {}

  StateId prevstate;
};

template <class A, class U>
struct ForwardBackwardFstState {
  typedef typename A::Weight Weight;
  vector<ForwardBackwardArc<A>*> iarcs_;
  vector<ForwardBackwardArc<A>*> oarcs_;
  Weight final_;
  U niarcs_iepsilons_;  // number of input arcs with input epsilons
  U niarcs_oepsilons_;  // number of input arcs with output epsilons
  U noarcs_iepsilons_;  // number of output arcs with input epsilons
  U noarcs_oepsilons_;  // number of output arcs with output epsilons

  ForwardBackwardFstState()
      : final_(Weight::Zero()), niarcs_iepsilons_(0), niarcs_oepsilons_(0),
        noarcs_iepsilons_(0), noarcs_oepsilons_(0) {}

  template <bool input_arcs = false>
  void DeleteArcs(const unordered_set<A*>& arcs_to_delete) {
    if (arcs_to_delete.empty()) return;
    vector<ForwardBackwardArc<A>*>& arcs = input_arcs ? iarcs_ : oarcs_;
    U& niepsilons = input_arcs ? niarcs_iepsilons_ : noarcs_iepsilons_;
    U& noepsilons = input_arcs ? niarcs_oepsilons_ : noarcs_oepsilons_;
    size_t na = 0;
    for (size_t a = 0; a < arcs.size(); ++a) {
      if (arcs_to_delete.count(arcs[a])) {
        arcs[na++] = arcs[a];
        if (arcs[a]->ilabel == 0) --niepsilons;
        if (arcs[a]->olabel == 0) --noepsilons;
      }
    }
    arcs.resize(na);
  }
};


/// Implementation of the Forward-Backward FST. Notice that it can be
/// instantiated with any arc type (A), but internally it will use the
/// ForwardBackwardArc inherited from A.
template <class A, class U>
class ForwardBackwardFstImpl :  public FstImpl<A> {
 public:
  typedef A Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef ForwardBackwardFstState<A, U> State;

  using FstImpl<A>::SetInputSymbols;
  using FstImpl<A>::SetOutputSymbols;
  using FstImpl<A>::Properties;
  using FstImpl<A>::SetProperties;
  using FstImpl<A>::SetType;

  friend class ArcIterator< ForwardBackwardFst<A, U> >;
  friend class MutableArcIterator< ForwardBackwardFst<A, U> >;

 public:

  ForwardBackwardFstImpl() : start_(kNoStateId) {
    SetType("forwardbackward");
    SetProperties(kNullProperties | kStaticProperties);
  }

  explicit ForwardBackwardFstImpl(const Fst<A>& fst) {
    SetType("forwardbackward");
    SetInputSymbols(fst.InputSymbols());
    SetOutputSymbols(fst.OutputSymbols());
    SetStart(fst.Start());
    if (fst.Properties(kExpanded, false))
      ReserveStates(CountStates(fst));
    for (StateIterator< Fst<A> > siter(fst); !siter.Done(); siter.Next()) {
      const StateId s = AddState();
      assert(s == siter.Value());
      SetFinal(s, fst.Final(s));
      ReserveArcs(s, fst.NumArcs(s));
    }
    for (StateIterator< Fst<A> > siter(fst); !siter.Done(); siter.Next()) {
      const StateId s = siter.Value();
      for (ArcIterator< Fst<A> > aiter(fst, s); !aiter.Done(); aiter.Next()) {
        const A& arc = aiter.Value();
        AddArc(s, arc);
      }
    }
    SetProperties(fst.Properties(kCopyProperties, false) | kStaticProperties);
  }

  virtual ~ForwardBackwardFstImpl() {
    DeleteStates();
  }

  StateId Start() const { return start_; }

  Weight Final(StateId s) const { return states_[s]->final_; }

  StateId NumStates() const { return states_.size(); }

  template <bool input_arcs = false>
  size_t NumInputEpsilons(StateId s) const {
    return input_arcs ? states_[s]->niarcs_iepsilons_ : states_[s]->noarcs_iepsilons_;
  }

  template <bool input_arcs = false>
  size_t NumOutputEpsilons(StateId s) const {
    return input_arcs ? states_[s]->niarcs_oepsilons_ : states_[s]->noarcs_oepsilons_;
  }

  template <bool input_arcs = false>
  size_t NumArcs(StateId s) const {
    return input_arcs ? states_[s]->iarcs_.size() : states_[s]->oarcs_.size();
  }

  void SetStart(StateId s) {
    start_ = s;
    SetProperties(SetStartProperties(Properties()));
  }

  void SetFinal(StateId s, Weight w) {
    const Weight& ow = states_[s]->final_;
    states_[s]->final_ = w;
    SetProperties(SetFinalProperties(Properties(), ow, w));
  }

  StateId AddState() {
    states_.push_back(new State);
    SetProperties(AddStateProperties(Properties()));
    return states_.size() - 1;
  }

  // Add arc using the
  void AddArc(StateId s, const A& arc) {
    const StateId n = arc.nextstate;
    ForwardBackwardArc<A>* new_arc = new ForwardBackwardArc<A>(arc, s);
    const ForwardBackwardArc<A> *prev_arc =
        states_[s]->oarcs_.empty() ? 0 : states_[s]->oarcs_.back();
    states_[s]->oarcs_.push_back(new_arc);
    states_[n]->iarcs_.push_back(new_arc);
    if (arc.ilabel == 0) {
      ++states_[s]->noarcs_iepsilons_;
      ++states_[n]->niarcs_iepsilons_;
    }
    if (arc.olabel == 0) {
      ++states_[s]->noarcs_oepsilons_;
      ++states_[n]->niarcs_oepsilons_;
    }
    SetProperties(AddArcProperties(Properties(), s, *new_arc, prev_arc));
  }

  void DeleteStates(const vector<StateId>& dstates) {
    if ( dstates.empty() ) return;
    // Get new id for all states (deleted states will have kNoStateId)
    vector<StateId> newid(states_.size(), 0);
    for (size_t i = 0; i < dstates.size(); ++i) {
      newid[dstates[i]] = kNoStateId;
    }
    // Delete states, and mark its input/output arcs as ready to delete
    unordered_set<A*> arcs_to_delete;
    StateId nstates = 0;
    for (size_t s = 0; s < states_.size(); ++s) {
      if (newid[s] != kNoStateId) {
        // change state id
        newid[s] = nstates;
        if (s != nstates)
          states_[nstates] = states_[s];
        ++nstates;
      } else {
        arcs_to_delete.insert(
            states_[s]->iarcs_.begin(), states_[s]->iarcs_.end());
        arcs_to_delete.insert(
            states_[s]->oarcs_.begin(), states_[s]->oarcs_.end());
        delete states_[s];
      }
    }
    states_.resize(nstates);
    // We need to fix the nextnode/prevnode from the arcs
    for (size_t s = 0; s < states_.size(); ++s) {
      states_[s]->DeleteArcs<false>(arcs_to_delete);
      states_[s]->DeleteArcs<true>(arcs_to_delete);
    }
    // Now, delete arcs
    for (A* arc: arcs_to_delete) {
      delete arc;
    }
    // Fix start state
    if (Start() != kNoStateId)
      SetStart(newid[Start()]);
    SetProperties(DeleteStatesProperties(Properties()));
  }

  void DeleteStates() {
    for (StateId s = 0; s < states_.size(); ++s) {
      for (size_t a = 0; a < states_[s]->oarcs_.size(); ++a)
        delete states_[s]->oarcs_[a];
      delete states_[s];
    }
    states_.clear();
    SetStart(kNoStateId);
    SetProperties(DeleteAllStatesProperties(Properties(), kStaticProperties));
  }

  /// Delete last n input/output arcs to/from state s
  template <bool input_arcs = false>
  void DeleteArcs(StateId s, size_t n) {
    // Delete input or output arcs from state s?
    vector<ForwardBackwardArc<A>*>& arcs =
        input_arcs ? states_[s]->iarcs_ : states_[s]->oarcs_;
    // Make sure n <= arcs.size()
    if (n > arcs.size()) n = arcs.size();
    // Traverse all the arcs that we want to delete
    unordered_set<A*> arcs_to_delete;
    unordered_set<StateId> other_states;
    for (size_t a = arcs.size() - n; a < arcs.size(); ++a) {
      arcs_to_delete.insert(arcs[a]);
      other_states.insert(input_arcs ? arcs[a]->prevstate : arcs[a]->nextstate);
    }
    arcs.resize(arcs.size() - n);
    for (StateId t : other_states) {
      states_[t]->DeleteArcs<!input_arcs>(arcs_to_delete);
    }
    for (A* arc : arcs_to_delete) {
      delete arc;
    }
    SetProperties(DeleteArcsProperties(Properties()));
  }

  // Delete all input/output arcs to/from state s
  template <bool input_arcs = false>
  void DeleteArcs(StateId s) {
    DeleteArcs<input_arcs>(s, std::numeric_limits<size_t>::max());
  }

  State* GetState(StateId s) { return states_[s]; }

  const State* GetState(StateId s) const { return states_[s]; }

  void ReserveStates(StateId n) { states_.reserve(n); }

  template <bool input_arcs = false>
  void ReserveArcs(StateId s, size_t n) {
    if (input_arcs)
      states_[s]->iarcs_.reserve(n);
    else
      states_[s]->oarcs_.reserve(n);
  }

  ///  Provide information needed for generic state iterator
  void InitStateIterator(StateIteratorData<A> *data) const {
    data->base = 0;
    data->nstates = states_.size();
  }

  ///  Provide information needed for generic arc iterator
  void InitArcIterator(StateId s, ArcIteratorData<A> *data) const {
    /// This should not be called!
    assert(false);
  }

  static ForwardBackwardFstImpl<A, U>* Read(
      istream& strm, const FstReadOptions& opts) {
    ForwardBackwardFstImpl<A, U> *impl = new ForwardBackwardFstImpl<A, U>;
    FstHeader hdr;
    if (!impl->ReadHeader(strm, opts, kMinFileVersion, &hdr)) {
      delete impl;
      return 0;
    }
    impl->SetStart(hdr.Start());
    if (hdr.NumStates() != kNoStateId)
      impl->ReserveStates(hdr.NumStates());
    StateId s = 0;
    for (; hdr.NumStates() == kNoStateId || s < hdr.NumStates(); ++s) {
      // Read final weight
      typename A::Weight final;
      if (!final.Read(strm)) break;
      const StateId s2 = impl->AddState();
      assert(s == s2);
      impl->states_[s]->final = final;
      // Read state arcs
      int64 narcs;
      ReadType(strm, &narcs);
      if (!strm) {
        LOG(ERROR) << "ForwardBackwardFst::Read: read failed: " << opts.source;
        delete impl;
        return 0;
      }
      impl->ReserveArcs(s, narcs);
      for (size_t j = 0; j < narcs; ++j) {
        A arc;
        ReadType(strm, &arc.ilabel);
        ReadType(strm, &arc.olabel);
        arc.weight.Read(strm);
        ReadType(strm, &arc.nextstate);
        if (!strm) {
          LOG(ERROR) << "ForwardBackwardFst::Read: read failed: "
                     << opts.source;
          delete impl;
          return 0;
        }
        impl->AddArc(s, arc);
      }
    }
    if (hdr.NumStates() != kNoStateId && s != hdr.NumStates()) {
      LOG(ERROR) << "ForwardBackwardFst::Read: unexpected end of file: "
                 << opts.source;
      delete impl;
      return 0;
    }
    return impl;
  }

  static const uint64 kStaticProperties = kExpanded | kMutable;

 private:
  ///  Current file format version
  static const int kFileVersion = 1;
  ///  Minimum file format version supported
  static const int kMinFileVersion = 1;

  vector< ForwardBackwardArc<A> > arcs_;
  vector<State*> states_;
  StateId start_;

  DISALLOW_COPY_AND_ASSIGN(ForwardBackwardFstImpl);
};

template <class A, class U>
const uint64 ForwardBackwardFstImpl<A, U>::kStaticProperties;
template <class A, class U>
const int ForwardBackwardFstImpl<A, U>::kFileVersion;
template <class A, class U>
const int ForwardBackwardFstImpl<A, U>::kMinFileVersion;


template <class A, class U = size_t>
class ForwardBackwardFst :
      public ImplToMutableFst< ForwardBackwardFstImpl<A, U> > {
 public:
  typedef ForwardBackwardFstImpl<A, U> Impl;
  typedef typename Impl::Arc Arc;
  typedef typename Arc::StateId StateId;

  friend class StateIterator< ForwardBackwardFst<A, U> >;
  friend class ArcIterator< ForwardBackwardFst<A, U> >;
  friend class MutableArcIterator< ForwardBackwardFst<A, U> >;

  ForwardBackwardFst() : ImplToMutableFst<Impl>(new Impl) {}

  ForwardBackwardFst(const ForwardBackwardFst<A, U>& fst)
      : ImplToMutableFst<Impl>(fst) {}

  explicit ForwardBackwardFst(const Fst<A>& fst)
      : ImplToMutableFst<Impl>(new Impl(fst)) {}

  ///  Get a copy of this ForwardBackwardFst
  virtual ForwardBackwardFst<A, U>* Copy(bool safe = false) const {
    return new ForwardBackwardFst<A, U>(*this);
  }

  ForwardBackwardFst<A, U>& operator=(const ForwardBackwardFst<A, U>& fst) {
    if (this != &fst) SetImpl(fst.GetImpl(), false);
    return *this;
  }

  virtual ForwardBackwardFst<A, U>& operator=(const Fst<A>& fst) {
    if (this != &fst) SetImpl(new Impl(fst), true);
    return *this;
  }

  ///  Read a ForwardBackwardFst from an input stream; return NULL on error
  static ForwardBackwardFst<A, U>* Read(
      istream &strm, const FstReadOptions &opts) {
    Impl* impl = Impl::Read(strm, opts);
    return impl ? new ForwardBackwardFst<A, U>(impl) : 0;
  }

  ///  Read a ForwardBackwardFst from a file; return NULL on error
  ///  Empty filename reads from standard input
  static ForwardBackwardFst<A, U> *Read(const string &filename) {
    Impl* impl = ImplToExpandedFst<Impl, MutableFst<A> >::Read(filename);
    return impl ? new ForwardBackwardFst<A, U>(impl) : 0;
  }

  virtual bool Write(ostream& strm, const FstWriteOptions& opts) const {
    return WriteFst(*this, strm, opts);
  }

  virtual bool Write(const string& filename) const {
    return Fst<A>::WriteFile(filename);
  }

  template <class F>
  static bool WriteFst(
      const F& fst, ostream& strm, const FstWriteOptions& opts) {
    static const int kFileVersion = 2;
    bool update_header = true;
    FstHeader hdr;
    hdr.SetStart(fst.Start());
    hdr.SetNumStates(kNoStateId);
    size_t start_offset = 0;
    if (fst.Properties(kExpanded, false) ||
        (start_offset = strm.tellp()) != -1) {
      hdr.SetNumStates(CountStates(fst));
      update_header = false;
    }
    uint64 properties = fst.Properties(kCopyProperties, false) |
        ForwardBackwardFstImpl<A, U>::kStaticProperties;
    FstImpl<A>::WriteFstHeader(
        fst, strm, opts, kFileVersion, "vector", properties, &hdr);
    StateId num_states = 0;
    for (StateIterator<F> siter(fst); !siter.Done(); siter.Next()) {
      typename A::StateId s = siter.Value();
      fst.Final(s).Write(strm);
      int64 narcs = fst.NumArcs(s);
      WriteType(strm, narcs);
      for (ArcIterator<F> aiter(fst, s); !aiter.Done(); aiter.Next()) {
        const A &arc = aiter.Value();
        WriteType(strm, arc.ilabel);
        WriteType(strm, arc.olabel);
        arc.weight.Write(strm);
        WriteType(strm, arc.nextstate);
      }
      num_states++;
    }
    strm.flush();
    if (!strm) {
      LOG(ERROR) << "ForwardBackwardFst::Write: write failed: " << opts.source;
      return false;
    }
    if (update_header) {
      hdr.SetNumStates(num_states);
      return FstImpl<A>::UpdateFstHeader(
          fst, strm, opts, kFileVersion, "vector", properties, &hdr,
          start_offset);
    } else {
      if (num_states != hdr.NumStates()) {
        LOG(ERROR) << "Inconsistent number of states observed during write";
        return false;
      }
    }
    return true;
  }

  virtual void InitStateIterator(StateIteratorData<A> *data) const {
    GetImpl()->InitStateIterator(data);
  }

  virtual void InitArcIterator(StateId s, ArcIteratorData<A> *data) const {
    data->base = new ArcIterator< ForwardBackwardFst<A, U> >(*this, s, false);
  }

  virtual inline
  void InitMutableArcIterator(StateId s, MutableArcIteratorData<A>* data) {
    data->base = new MutableArcIterator< ForwardBackwardFst<A, U> >(this, s);
  }

  template <bool input_arcs = false>
  size_t NumInputEpsilons(StateId s) const {
    return GetImpl()->NumInputEpsilons<input_arcs>(s);
  }

  template <bool input_arcs = false>
  size_t NumOutputEpsilons(StateId s) const {
    return GetImpl()->NumOutputEpsilons<input_arcs>(s);
  }

  template <bool input_arcs = false>
  size_t NumArcs(StateId s) const {
    return GetImpl()->NumArcs<input_arcs>(s);
  }

  template <bool input_arcs = false>
  void DeleteArcs(StateId s, size_t n) {
    MutateCheck();
    GetImpl()->DeleteArcs(s, n);
  }

  template <bool input_arcs = false>
  void DeleteArcs() {
    MutateCheck();
    GetImpl()->DeleteArcs();
  }

  template <bool input_arcs = false>
  void ReserveArcs(StateId s, size_t n) {
    MutateCheck();
    GetImpl()->ReserveArcs<input_arcs>(s, n);
  }

 private:
  explicit ForwardBackwardFst(Impl *impl) : ImplToMutableFst<Impl>(impl) {}

  ///  Makes visible to friends.
  Impl *GetImpl() const { return ImplToFst< Impl, MutableFst<A> >::GetImpl(); }

  void SetImpl(Impl *impl, bool own_impl = true) {
    ImplToFst< Impl, MutableFst<A> >::SetImpl(impl, own_impl);
  }

  void MutateCheck() { ImplToMutableFst<Impl>::MutateCheck(); }
};

///  Specialization for ForwadBackwardFst; see generic version in fst.h
///  for sample usage (but use the ForwardBackwardFst type!). This version
///  should inline.
template <class A, class U>
class StateIterator< ForwardBackwardFst<A, U> > {
 public:
  typedef typename A::StateId StateId;

  explicit StateIterator(const ForwardBackwardFst<A, U> &fst)
      : nstates_(fst.GetImpl()->NumStates()), s_(0) {}

  bool Done() const { return s_ >= nstates_; }

  StateId Value() const { return s_; }

  void Next() { ++s_; }

  void Reset() { s_ = 0; }

 private:
  StateId nstates_;
  StateId s_;

  DISALLOW_COPY_AND_ASSIGN(StateIterator);
};

///  Specialization for ForwardBackwardFst; see generic version in fst.h
///  for sample usage (but use the ForwardBackwardFst type!). This version
///  should inline.
template <class A, class U>
class ArcIterator< ForwardBackwardFst<A, U> >
    : public ArcIteratorBase<A> {
 public:
  typedef typename ForwardBackwardFst<A, U>::Arc Arc;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;

  ArcIterator(
      const ForwardBackwardFst<A, U> &fst, StateId s, bool backward = false)
      : arcs_(
            backward ?
            fst.GetImpl()->states_[s]->iarcs_ :
            fst.GetImpl()->states_[s]->oarcs_),
        i_(0) {}

  bool Done() const { return i_ >= arcs_.size(); }

  const ForwardBackwardArc<A>& Value() const { return *arcs_[i_]; }

  void Next() { ++i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  size_t Position() const { return i_; }

  uint32 Flags() const { return kArcValueFlags; }

  void SetFlags(uint32 f, uint32 m) {}

 private:
  virtual bool Done_() const { return Done(); }
  virtual const A& Value_() const { return Value(); }
  virtual void Next_() { Next(); }
  virtual size_t Position_() const { return Position(); }
  virtual void Reset_() { Reset(); }
  virtual void Seek_(size_t a) { Seek(a); }
  virtual uint32 Flags_() const { return Flags(); }
  virtual void SetFlags_(uint32 flags, uint32 mask) { SetFlags(flags, mask); }

  const vector<ForwardBackwardArc<A>*>& arcs_;
  size_t i_;

  DISALLOW_COPY_AND_ASSIGN(ArcIterator);
};

///  Specialization for ForwardBackwardFst; see generic version in fst.h
///  for sample usage (but use the ForwardBackwardFst type!). This version
///  should inline.
template <class A, class U>
class MutableArcIterator< ForwardBackwardFst<A, U> >
    : public MutableArcIteratorBase<A> {
 public:
  typedef typename A::StateId StateId;
  typedef typename A::Weight Weight;

  MutableArcIterator(ForwardBackwardFst<A, U>* fst, StateId s)
      : fst_(fst), i_(0) {
    fst->MutateCheck();
    state_ = fst->GetImpl()->GetState(s);
    properties_ = &fst->GetImpl()->properties_;
  }

  bool Done() const { return i_ >= state_->oarcs_.size(); }

  const A& Value() const { return *state_->oarcs_[i_]; }

  void Next() { ++i_; }

  size_t Position() const { return i_; }

  void Reset() { i_ = 0; }

  void Seek(size_t a) { i_ = a; }

  void SetValue(const A &arc) {
    ForwardBackwardArc<A>* oarc = state_->oarcs_[i_];
    ForwardBackwardFstState<A, U>* nstate =
        fst_->GetImpl()->GetState(oarc->nextstate);
    if (oarc->ilabel != oarc->olabel)
      *properties_ &= ~kNotAcceptor;
    if (oarc->ilabel == 0) {
      --state_->noarcs_iepsilons_;
      --nstate->niarcs_iepsilons_;
      *properties_ &= ~kIEpsilons;
      if (oarc->olabel == 0)
        *properties_ &= ~kEpsilons;
    }
    if (oarc->olabel == 0) {
      --state_->noarcs_oepsilons_;
      --nstate->niarcs_oepsilons_;
      *properties_ &= ~kOEpsilons;
    }
    if (oarc->weight != Weight::Zero() && oarc->weight != Weight::One())
      *properties_ &= ~kWeighted;
    // Change ilabel, olabel and weight from the arc
    oarc->ilabel = arc.ilabel;
    oarc->olabel = arc.olabel;
    oarc->weight = arc.weight;
    // Change nextstate (and modify input arcs from the old and new nextstate)
    if (arc.nextstate != oarc->nextstate) {
      nstate->iarcs_.erase(
          find(nstate->iarcs_.begin(), nstate->iarcs_.end(), oarc));
      nstate = fst_->GetImpl()->GetState(arc.nextstate);
      nstate->iarcs_.push_back(oarc);
      oarc->nextstate = arc.nextstate;
    }
    if (arc.ilabel != arc.olabel) {
      *properties_ |= kNotAcceptor;
      *properties_ &= ~kAcceptor;
    }
    if (arc.ilabel == 0) {
      ++state_->noarcs_iepsilons_;
      ++nstate->niarcs_iepsilons_;
      *properties_ |= kIEpsilons;
      *properties_ &= ~kNoIEpsilons;
      if (arc.olabel == 0) {
        *properties_ |= kEpsilons;
        *properties_ &= ~kNoEpsilons;
      }
    }
    if (arc.olabel == 0) {
      ++state_->noarcs_oepsilons_;
      ++nstate->niarcs_oepsilons_;
      *properties_ |= kOEpsilons;
      *properties_ &= ~kNoOEpsilons;
    }
    if (arc.weight != Weight::Zero() && arc.weight != Weight::One()) {
      *properties_ |= kWeighted;
      *properties_ &= ~kUnweighted;
    }
    *properties_ &= kSetArcProperties | kAcceptor | kNotAcceptor |
        kEpsilons | kNoEpsilons | kIEpsilons | kNoIEpsilons |
        kOEpsilons | kNoOEpsilons | kWeighted | kUnweighted;
  }

  uint32 Flags() const {
    return kArcValueFlags;
  }

  void SetFlags(uint32 f, uint32 m) {}

 private:
  ///  This allows base-class virtual access to non-virtual derived-
  ///  class members of the same name. It makes the derived class more
  ///  efficient to use but unsafe to further derive.
  virtual bool Done_() const { return Done(); }
  virtual const A& Value_() const { return Value(); }
  virtual void Next_() { Next(); }
  virtual size_t Position_() const { return Position(); }
  virtual void Reset_() { Reset(); }
  virtual void Seek_(size_t a) { Seek(a); }
  virtual void SetValue_(const A &a) { SetValue(a); }
  uint32 Flags_() const { return Flags(); }
  void SetFlags_(uint32 f, uint32 m) { SetFlags(f, m); }

  ForwardBackwardFst<A, U>* fst_;
  ForwardBackwardFstState<A, U>* state_;
  uint64 *properties_;
  size_t i_;

  DISALLOW_COPY_AND_ASSIGN(MutableArcIterator);
};

typedef ForwardBackwardFst<StdArc, size_t> StdForwardBackwardFst;
typedef ForwardBackwardFst<LogArc, size_t> LogForwardBackwardFst;

}  // namespace fst


#endif  // KALDI_FSTEXT_FORWARD_BACKWARD_FST_H_
