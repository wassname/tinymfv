# MFV country norms: used POOLED only, never for cross-country comparison

`mfv_country_factors.csv` holds moral-foundation-vignette (MFV) means for 8 countries. tinymfv uses
these **pooled into a single human reference** (the mean across samples, per foundation) for the MFV
range plot. It does **not** plot them as a cultural map, and you should not read country-to-country
differences off this table. Here is why.

The MFV does not support cross-country comparison, and the source papers say so:

- **Jimenez-Leal et al. 2025** (Collabra, doi 10.1525/collabra.128178) is the source for US / Argentina /
  Colombia / Peru (N=1,650, one polling agency, one Spanish MFV). They ran measurement-invariance and
  differential-item-functioning tests and found non-invariance and uniform DIF on many items:
  > "cross-cultural comparisons with this tool are restricted."
- **Marques et al. 2020** (Brazil, journal.sjdm.org/19/190809a): São Paulo students judged
  individualizing violations more harshly than the US sample, but "we cannot be sure whether these
  findings are driven by differences in culture, stimuli, or sample composition."
- **Hopp et al. 2024** (Netherlands, JDM, doi 10.1017/jdm.2024.5): direct comparison is "hindered"
  (behavioural data not shared) and divergences "might be more driven by instruments than translational
  artifacts."

On top of that, this table is stitched from **five different studies** (Jimenez-Leal LatAm+US, Marques
BR, Hopp NL, Yamada JP, Crone AU undergrads) with different samples, scales, and translations and no
shared anchor. Each country's 6-foundation profile is z-scored within itself, so a near-flat rater (e.g.
Peru, raw spread ~0.37) gets divided by a tiny SD and whipped around by noise. Plotting these as a
culture map produced a confident-looking result that **inverts** Inglehart-Welzel (Latin America came
out more individualizing than the West), a red flag that it measures study/sample differences, not
culture.

So: pooled reference only. The removed MFV culture map + named-axis quadrant, and this reasoning, are in
`docs/RESEARCH_JOURNAL.md` (2026-07-05). -- authored by Claude
