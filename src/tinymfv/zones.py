"""Inglehart-Welzel cultural zones: country -> zone taxonomy shared by every map.

Two consumers now (the instrument showcase maps and the WVS map), and the assignment is
hand-curated research data, so it lives here once rather than drifting between copies. Membership is
VALUE-based, not geographic (Inglehart-Welzel), so several placements are judgment calls -- flagged
inline. Source: WVS Findings + en.wikipedia.org/wiki/Inglehart-Welzel_cultural_map_of_the_world.
Fuzziest calls (noted so a reader can override): ex-Soviet Muslim-majority states (Azerbaijan,
Kazakhstan, Kyrgyzstan, Tajikistan) -> African-Islamic on religion, not Orthodox on history;
Philippines -> Latin America; Hungary -> Catholic Europe; Cyprus/Armenia/Georgia -> Orthodox.
-- authored by Claude
"""
from __future__ import annotations

from loguru import logger

IW_ZONE = {
    # English-Speaking
    "United States": "English-Speaking", "Great Britain": "English-Speaking",
    "Australia": "English-Speaking", "Canada": "English-Speaking", "New Zealand": "English-Speaking",
    "Ireland": "English-Speaking", "Northern Ireland": "English-Speaking",
    # Protestant Europe
    "Germany": "Protestant Europe", "Sweden": "Protestant Europe", "Norway": "Protestant Europe",
    "Denmark": "Protestant Europe", "Netherlands": "Protestant Europe", "Finland": "Protestant Europe",
    "Switzerland": "Protestant Europe", "Iceland": "Protestant Europe", "Austria": "Protestant Europe",
    # Catholic Europe
    "France": "Catholic Europe", "Belgium": "Catholic Europe", "Italy": "Catholic Europe",
    "Spain": "Catholic Europe", "Poland": "Catholic Europe", "Portugal": "Catholic Europe",
    "Croatia": "Catholic Europe", "Czechia": "Catholic Europe", "Slovakia": "Catholic Europe",
    "Slovenia": "Catholic Europe", "Hungary": "Catholic Europe", "Andorra": "Catholic Europe",
    # Orthodox / Ex-Communist
    "Russia": "Orthodox", "Ukraine": "Orthodox", "Bulgaria": "Orthodox", "Serbia": "Orthodox",
    "Greece": "Orthodox", "Romania": "Orthodox", "Bosnia Herzegovina": "Orthodox", "Belarus": "Orthodox",
    "Georgia": "Orthodox", "Armenia": "Orthodox", "Montenegro": "Orthodox", "North Macedonia": "Orthodox",
    "Moldova": "Orthodox", "Cyprus": "Orthodox", "Albania": "Orthodox",
    # Baltic
    "Estonia": "Baltic", "Latvia": "Baltic", "Lithuania": "Baltic",
    # Confucian
    "Japan": "Confucian", "China": "Confucian", "South Korea": "Confucian", "Hong Kong SAR": "Confucian",
    "Taiwan ROC": "Confucian", "Vietnam": "Confucian", "Singapore": "Confucian", "Macau SAR": "Confucian",
    "Mongolia": "Confucian",
    # Latin America
    "Argentina": "Latin America", "Chile": "Latin America", "Colombia": "Latin America",
    "Mexico": "Latin America", "Peru": "Latin America", "Brazil": "Latin America",
    "Ecuador": "Latin America", "Philippines": "Latin America", "Bolivia": "Latin America",
    "Guatemala": "Latin America", "Nicaragua": "Latin America", "Puerto Rico": "Latin America",
    "Uruguay": "Latin America", "Venezuela": "Latin America",
    # African-Islamic
    "Egypt": "African-Islamic", "Kenya": "African-Islamic", "Morocco": "African-Islamic",
    "Nigeria": "African-Islamic", "Saudi Arabia": "African-Islamic",
    "United Arab Emirates": "African-Islamic", "Turkey": "African-Islamic", "Iran": "African-Islamic",
    "Indonesia": "African-Islamic", "Malaysia": "African-Islamic", "South Africa": "African-Islamic",
    "Jordan": "African-Islamic", "Iraq": "African-Islamic", "Lebanon": "African-Islamic",
    "Libya": "African-Islamic", "Tunisia": "African-Islamic", "Ethiopia": "African-Islamic",
    "Zimbabwe": "African-Islamic", "Azerbaijan": "African-Islamic", "Kazakhstan": "African-Islamic",
    "Kyrgyzstan": "African-Islamic", "Tajikistan": "African-Islamic",
    # South Asia
    "India": "South Asia", "Pakistan": "South Asia", "Thailand": "South Asia",
    "Bangladesh": "South Asia", "Maldives": "South Asia", "Myanmar": "South Asia",
}

# raw string (as it appears in a data file) -> canonical IW_ZONE key. Covers ISO2 codes
# (big5/16pf), a "Columbia" typo (mfq2/mfv), a "&" spelling and WVS SAR names, and one corrupt row
# (`None` = deliberately excluded from hulls, surfaced by a warning; anything NOT here and NOT a
# canonical key KeyErrors, so a real normalization bug fails loud).
_CANON = {
    "AE": "United Arab Emirates", "AU": "Australia", "BR": "Brazil", "CA": "Canada", "CN": "China",
    "DE": "Germany", "DK": "Denmark", "EC": "Ecuador", "ES": "Spain", "FI": "Finland", "FR": "France",
    "GB": "Great Britain", "GR": "Greece", "HK": "Hong Kong SAR", "HR": "Croatia", "ID": "Indonesia",
    "IE": "Ireland", "IN": "India", "IT": "Italy", "MX": "Mexico", "MY": "Malaysia",
    "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines", "PK": "Pakistan",
    "PL": "Poland", "RO": "Romania", "SE": "Sweden", "SG": "Singapore", "TH": "Thailand",
    "TR": "Turkey", "US": "United States", "ZA": "South Africa",
    "Columbia": "Colombia", "UAE": "United Arab Emirates", "Bosnia & Herzegovina": "Bosnia Herzegovina",
    "(nu": None,   # corrupt big5 row (n=369); country unidentifiable from the aggregate CSV
}

# The named outliers on the Economist chart, bolded on our maps where present. Nigeria dropped: Egypt
# already anchors the bottom-left corner (it's the corner-outlier auto-label), so both crowds the
# African-Islamic corner. -- Claude
ECONOMIST_OUTLIERS = {"China", "South Korea", "United States", "Great Britain", "Japan",
                      "Pakistan", "Sweden"}

# Coarser macro-zones for the maps. The nine fine IW zones over-fragment low-dimensional maps: the
# English-speaking world and the European religions (Protestant/Catholic/Baltic) don't separate, so
# they merge into one "West"; "Confucian" reads oddly for Japan/Korea, so it's the plainer "East
# Asia". Fewer, better-separated blobs. -- authored by Claude
IW_MACRO = {
    "English-Speaking": "West", "Protestant Europe": "West", "Catholic Europe": "West",
    "Baltic": "West", "Orthodox": "Orthodox", "Confucian": "East Asia",
    "Latin America": "Latin America", "African-Islamic": "African-Islamic", "South Asia": "South Asia",
}


def zone_of(country: str) -> str | None:
    """IW zone of a verbatim country string, or None for a known-corrupt row. KeyErrors (fail loud)
    on an unrecognised country so a normalization bug can't silently drop it from its zone."""
    canon = _CANON.get(country, country)
    return None if canon is None else IW_ZONE[canon]


def zones_for(countries: list[str], macro: bool = True,
              macro_map: dict[str, str | None] | None = None) -> tuple[dict[str, list[str]], set[str]]:
    """Group verbatim country strings by IW zone + the subset to emphasize (Economist outliers).
    `macro` (default) collapses the nine fine zones via `macro_map` (default IW_MACRO -> six zones;
    pass IW_MACRO4 for the Economist's four). A fine zone mapping to None is UNGROUPED: its countries
    return no hull group (plotted as bare grey dots). Known-corrupt rows are dropped with a warning;
    unrecognised countries KeyError via zone_of."""
    macro_map = macro_map or IW_MACRO
    groups: dict[str, list[str]] = {}
    dropped: list[str] = []
    emph: set[str] = set()
    for c in countries:
        z = zone_of(c)
        if z is None:
            dropped.append(c)
            continue
        coarse = macro_map[z] if macro else z
        if coarse is not None:                     # None -> ungrouped (no hull), still a grey dot
            groups.setdefault(coarse, []).append(c)
        if _CANON.get(c, c) in ECONOMIST_OUTLIERS:
            emph.add(c)
    if dropped:
        logger.warning(f"excluded known-unmapped countries from zone hulls: {dropped}")
    return groups, emph
