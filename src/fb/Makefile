all:

EXTRA_CXXFLAGS = -Wno-sign-compare -O3
include ../kaldi.mk

TESTFILES = simple-forward-test simple-backward-test simple-posteriors-test \
	simple-forward-backward-test fast-forward-backward-test

OBJFILES = simple-forward-backward.o simple-forward.o simple-backward.o \
	simple-common.o fast-forward-backward.o test-common.o

LIBNAME = kaldi-fb

ADDLIBS = ../transform/kaldi-transform.a ../tree/kaldi-tree.a ../lat/kaldi-lat.a \
     ../sgmm/kaldi-sgmm.a ../gmm/kaldi-gmm.a ../hmm/kaldi-hmm.a ../util/kaldi-util.a \
     ../base/kaldi-base.a ../matrix/kaldi-matrix.a

include ../makefiles/default_rules.mk
