# Sources & provenance: `mfv_country_factors.csv`

Audit trail for the human country rows in the MFV value map. One entry per `source`
tag in the CSV. Each entry records: full citation, URL/DOI, exactly which table/figure
the numbers came from, and any data transformation applied (so a reviewer can reproduce
the row from the primary source).

Author of these notes: Claude (pair-programming with wassname), not wassname.

## Instrument & how the map uses these numbers

All rows are the Clifford et al. (2015) Moral Foundations Vignettes (MFV): 2nd-person
"You see ..." vignettes rated for moral wrongness on a 1-5 scale, coded to six
foundations (care, fairness, liberty, authority, loyalty/ingroup, sanctity/purity).

The value map (`scripts/plot_steer_showcase.py::read_human_mfv`) reads **only the `mean`
column**, then **z-scores each country across its six foundations** (ipsative). Because
z-scoring is affine-invariant, any per-country scale/offset (translation bias, digitizing
bias, differing scale anchors) cancels: the map shows *relative* foundation emphasis, not
a calibrated cross-country ranking. Absolute means/SDs are still stored honestly for any
non-map reuse.

Caveat carried by two of the source papers (Jimenez-Leal, Hopp): the MFV shows
measurement non-invariance / differential item functioning across countries, so raw
between-country mean comparisons are a rough reference, not a validated ranking.

## Sources

### `JimenezLeal2025_LatAm` -- Argentina, Colombia, Peru, US
- Jimenez Leal, W., Carmona, G., Murray, S., & Amaya, S. (2025). Validation of the Moral
  Foundation Vignettes in Latin America. *Collabra: Psychology*, 11(1), 128178.
- DOI: https://doi.org/10.1525/collabra.128178 (open access, CC BY)
- Numbers: per-country foundation means/SD/N from the paper's descriptive tables
  (N = 1,650 across 3 Latin-American countries via polling agency, plus a US comparison).
- Transformation: none (means used as tabulated on the native 1-5 scale).

### `Yamada2025_MFV-J` -- Japan
- Yamada, J., Nakawake, Y., & Suyama, M. (2026). Developing a Japanese version of the
  Moral Foundations Vignettes (MFV-J). *The Japanese Journal of Psychology*.
  (Tag says 2025 = preprint/advance-pub year; journal assigns 2026.)
- DOI: https://doi.org/10.4992/jjpsy.97.24228 (advance publication PDF, open access)
- Numbers: MFV-J foundation means/SD, N = 564, from the paper's descriptive table.
- Transformation: none.

### `Hopp2024_DutchMFV` -- Netherlands
- Hopp, F. R., Jargow, B., Kouwen, E., & Bakker, B. N. (2024). The Dutch moral foundations
  stimulus database. *Judgment and Decision Making*, 19, e10.
- DOI: https://doi.org/10.1017/jdm.2024.5 (open access, CC BY). OSF: https://osf.io/9gnza/
- Numbers: foundation means + 95% CIs from the paper's Table 1 (N = 586 Dutch crowdworkers,
  120 translated MFVs). Per-foundation N varies by item allocation (that's why the CSV N
  differs per row).
- Transformation: **Care collapsed** from the paper's split physical-care (4.09) +
  emotional-care (3.53) into a single care = 3.81 (mean of the two). Other foundations
  taken as tabulated. SD/SE/CI from the paper's reported CIs.

### `Marques2020_BrazilMFV_fig3digitized_affinecal` -- Brazil
- Marques, L. M., et al. (2020). Translation and validation of the Moral Foundations
  Vignettes for the Portuguese language in a Brazilian sample. *Judgment and Decision
  Making*.
- URL: http://journal.sjdm.org/19/190809a/jdm190809a.html
  PDF: http://journal.sjdm.org/19/190809a/jdm190809a.pdf
- Numbers: the paper tabulates no per-foundation means, so means were **digitized from
  Figure 3** (Brazil series) with WebPlotDigitizer by wassname. N = 494 (paper).
- Transformations (two, both by Claude):
  1. **Care collapsed** from digitized Care-E + Care-P into one care value.
  2. **Affine bias-correction** of all digitized means to two paper-stated Purity anchors
     (Brazil purity 3.45, Clifford US purity 3.85; paper text): `true = 0.9155*digitized
     + 0.228`. Reproduces both anchors; Brazil purity lands exactly 3.45. SDs recovered
     from the digitized 95% CI whiskers (Fig 3 caption): `sd = halfwidth/1.96 * sqrt(494)`;
     each whisker pair's midpoint matches its mean to <=0.005 (validated). SDs scaled by
     the affine slope. Provably map-neutral (max |dz| = 0.017, rounding only).
- Digitized source figure staged at `docs/digitize/brazil_fig3_page-08.png`.

### `Crone2021_AusUndergrad_MFV90raw` -- Australia
- Crone, D. L., Rhee, J. J., & Laham, S. M. (2021). Developing brief versions of the Moral
  Foundations Vignettes using a genetic algorithm-based approach. *Behavior Research
  Methods*, 53(3), 1179-1187.
- DOI: https://doi.org/10.3758/s13428-020-01489-y . OSF: https://osf.io/cmwpv/
  (component "Data" = https://osf.io/nv4ty/ , file `dat_rep.sav`).
- Sample: 756 Australian undergraduates (complete cases). The paper's other sample is
  580 US MTurk workers (`dat_amt`), NOT used here -- that would duplicate the US row.
- Numbers: computed by Claude from the **raw participant ratings** in `dat_rep.sav`, the
  full 90-item Clifford MFV (1-5 wrongness), using the author's own item->foundation map
  from `mfv_abbreviation.Rmd` (Care = MFV 1-27 [physical+emotional+other], Fairness 28-39,
  Liberty 40-50, Authority 51-64, Loyalty/Ingroup 65-80, Sanctity/Purity 81-90).
  Per foundation: mean over participants of each participant's item-mean; SD = between-
  participant SD (ddof=1); SE = SD/sqrt(N); CI = mean +/- 1.96*SE. Complete-case exclusion
  (all 90 items present) reproduces the paper's N = 756 exactly.
- Transformation: **NOT the genetic-algorithm-abbreviated subset.** The paper's headline
  contribution is a brief MFV chosen by GAabbreviate; we deliberately used the *full*
  90-item ratings so Australia is comparable to the other (full-instrument) country rows.
  Care spans all 27 care items (no collapse needed; it's the natural mean).
- Reproduce: OSF files fetched via `https://osf.io/download/4psfc/` (dat_rep.sav);
  read with `pyreadstat.read_sav`. (The OSF "Download as zip" gave an empty archive --
  fetch files individually.)
