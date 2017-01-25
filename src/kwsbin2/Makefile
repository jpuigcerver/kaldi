
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = lattice-search

OBJFILES =

TESTFILES =

ADDLIBS = ../kws/kaldi-kws.a ../lat/kaldi-lat.a ../fstext/kaldi-fstext.a \
        ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
        ../util/kaldi-util.a ../thread/kaldi-thread.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk
