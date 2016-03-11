// fb/queue-set.h


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

#ifndef KALDI_FB_QUEUE_SET_H_
#define KALDI_FB_QUEUE_SET_H_

// This is like a regular queue, where only one copy of an element can be
// queued.
// It is used in the shortest-distance algorithm described in
// `Semiring Frameworks and Algorithms for Shortest-Distance Problems', by
// M. Mohri, Journal of Automata, Languages and Computation, 2002.

template <typename T>
class QueueSet {
 private:
  std::queue<T> queue_;
  std::set<T> set_;

 public:
  bool empty() const {
    return queue_.empty();
  }

  size_t size() const {
    return queue_.size();
  }

  void push(const T& n) {
    if (set_.insert(n).second)
      queue_.push(n);
  }

  const T& front() const {
    return queue_.front();
  }

  T& front() {
    return queue_.front();
  }

  void pop() {
    set_.erase(queue_.front());
    queue_.pop();
  }
};

#endif  // KALDI_FB_QUEUE_SET_H_
