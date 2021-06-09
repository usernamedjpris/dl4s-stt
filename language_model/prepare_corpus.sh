#!/bin/bash

set -xe

if [ ! -f "wiki_fr_lower.txt" ]; then
	curl -sSL https://github.com/Common-Voice/commonvoice-fr/releases/download/lm-0.1/wiki.txt.xz | pixz -d | tr '[:upper:]' '[:lower:]' > wiki_fr_lower.txt
fi;

if [ ! -f "debats-assemblee-nationale.txt" ]; then
	curl -sSL https://github.com/Common-Voice/commonvoice-fr/releases/download/lm-0.1/debats-assemblee-nationale.txt.xz | pixz -d | tr '[:upper:]' '[:lower:]' > debats-assemblee-nationale.txt
fi;

# Remove special-char <s> that will make KenLM tools choke:
# kenlm/lm/builder/corpus_count.cc:179 in void lm::builder::{anonymous}::ComplainDisallowed(StringPiece, lm::WarningAction&) threw FormatLoadException.
# Special word <s> is not allowed in the corpus.  I plan to support models containing <unk> in the future.  Pass --skip_symbols to convert these symbols to whitespace.
if [ ! -f "sources_lm.txt" ]; then
	cat wiki_fr_lower.txt debats-assemblee-nationale.txt | sed -e 's/<s>/ /g' > sources_lm.txt
fi;

split -n 5 sources_lm.txt
mv xaa sources_lm_small.txt
rm x??* 

