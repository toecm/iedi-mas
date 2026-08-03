# Manuscript package

- `main.tex` is the double-anonymous Elsevier `elsarticle` protocol/preprint
  draft. It must contain no author details, named repository URL, unpublished
  submission history, or direct identity-bearing artifact link.
- The non-anonymous title page is not included in this public artifact. Keep it
  access-controlled until every named author confirms authorship, order,
  affiliation, and public release; submit it separately only where requested.
- `references.bib` contains the sources currently cited by the draft.
  Published prior work remains cited neutrally; the executable notebook uses a
  review-only anonymized artifact placeholder whose exact pins are retained in
  the excluded `../CLAIM_LINEAGE.md` file.
- `highlights.txt` contains five submission highlights.

`main.tex` intentionally declares `\resultslockedtrue`. Do not switch or
remove this guard until the sealed run and evidence register are complete.
Each `\resultcell` must then be replaced through a traceable table-generation
script, not by copying values from a notebook or design simulation.

The working journal target is *Engineering Applications of Artificial
Intelligence*. Confirm the journal's current scope, article type, word limits,
author instructions, and declarations immediately before submission; journal
quartiles and indexing can change over time.

The review manuscript remains a protocol/preprint until empirical data exist.
It also prespecifies a bounded, consented text-prototype application pilot and
a minimum deidentified release bundle. Neither plan may be described as a
completed deployment or reproducibility result before its evidence gate is
met.
