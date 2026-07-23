#!/usr/bin/env python3
"""Screen ATCC-purchasable pathogens that the MIC model never saw in training.

Reviewer 4 asks for broad efficacy against a species or genus the model was
never trained on. This read-only screen produces the candidate shortlist:

1. Recompute the actual guidance-regressor training exposure (1,599 strain IDs
   -> producer-era species names) with the same frozen filtering used by
   ``audit_reviewer4_inhouse_species_coverage.py``.
2. Resolve every training species name *and* every candidate name through NCBI
   Taxonomy, so an unseen call is made on current names. Producer-era labels
   are stale in both directions (``Eubacterium rectale`` -> ``Agathobacter``,
   ``Ochrobactrum anthropi`` -> ``Brucella``), and a raw string diff would
   otherwise invent unseen genera that the model has in fact been trained on.
3. Query the public ATCC catalogue for each candidate and record, per species,
   how many distinct catalogue numbers are orderable, how many already have an
   ATCC Genome Portal assembly, their biosafety level and type-strain status.

The genome flag is the gating asset: without a genome there is no Evo-2
embedding, so the strain cannot enter guided generation without new
sequencing. Multiple orderable isolates per species is the other gate -- a
single isolate cannot support a species- or genus-level efficacy claim.

Nothing here is a clinical recommendation. Target choice stays with the
authors and the microbiology team.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, urlopen

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_reviewer4_inhouse_species_coverage import (  # noqa: E402
    canonicalize_training_species,
)
from apexoracle.data.hierarchical_mic_preparation import (  # noqa: E402
    prepare_hierarchical_mic_data,
)


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


NCBI_TAXONOMY = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/taxonomy/dataset_report"
ATCC_SEARCH = "https://www.atcc.org/coveo/rest/search/v2"
# Sitecore template names taken from the live @z95xtemplatename facet. Fungi sit
# under "Mycology" and protozoa under "Protistology"; guessing these names
# silently returns zero products for every eukaryotic candidate.
ATCC_TEMPLATES = '@z95xtemplatename==("Bacteria and Bacteriophages","Mycology","Protistology")'
ATCC_FIELDS = (
    "atccz32xnumber",
    "organismacceptedname",
    "strainz32xdesignation",
    "biosafetyz32xlevel",
    "genomicz32xdata",
    "genomicz32xdataz32xurl",
    "typez32xstrain",
    "productz32xcategory",
    "webz32xavailable",
    "isolationz32xsource",
    "z95xtemplatename",
)


@dataclass(frozen=True)
class Candidate:
    """A curated clinical pathogen considered as an unseen-target candidate."""

    species: str
    group: str          # coarse organism class, drives assay feasibility
    syndrome: str       # why a clinician would care
    note: str = ""


# Curated from routine clinical-microbiology practice. The list is deliberately
# broader than any final shortlist: the taxonomy diff and the ATCC query do the
# filtering, so a candidate that turns out to be trained-on or genome-less stays
# visible as a negative result instead of being quietly omitted.
CANDIDATES: tuple[Candidate, ...] = (
    # --- Enterobacterales ---
    Candidate("Providencia stuartii", "gram_negative_enterobacterales",
              "catheter-associated UTI, MDR nosocomial bacteraemia",
              "intrinsic polymyxin resistance via lipid A L-Ara4N"),
    Candidate("Providencia rettgeri", "gram_negative_enterobacterales",
              "UTI, wound infection; NDM carbapenemase reservoir",
              "intrinsic polymyxin resistance"),
    Candidate("Providencia alcalifaciens", "gram_negative_enterobacterales",
              "traveller's diarrhoea, enteric infection"),
    Candidate("Morganella morganii", "gram_negative_enterobacterales",
              "UTI, post-operative wound and bloodstream infection",
              "intrinsic polymyxin and 1st-gen cephalosporin resistance"),
    Candidate("Hafnia alvei", "gram_negative_enterobacterales",
              "opportunistic bacteraemia, gastroenteritis"),
    Candidate("Pantoea agglomerans", "gram_negative_enterobacterales",
              "line and infusate-associated bacteraemia, septic arthritis"),
    Candidate("Leclercia adecarboxylata", "gram_negative_enterobacterales",
              "line infection in immunocompromised hosts"),
    Candidate("Kluyvera ascorbata", "gram_negative_enterobacterales",
              "opportunistic UTI/bacteraemia; CTX-M ancestral reservoir"),
    Candidate("Plesiomonas shigelloides", "gram_negative_enterobacterales",
              "gastroenteritis, rare sepsis and meningitis"),
    Candidate("Pluralibacter gergoviae", "gram_negative_enterobacterales",
              "nosocomial outbreaks, preservative-resistant contaminant"),

    # --- Non-fermenting gram negatives ---
    Candidate("Chryseobacterium indologenes", "gram_negative_nonfermenter",
              "ventilator-associated pneumonia, line sepsis",
              "intrinsically resistant to colistin and carbapenems"),
    Candidate("Chryseobacterium gleum", "gram_negative_nonfermenter",
              "nosocomial respiratory and urinary infection"),
    Candidate("Sphingomonas paucimobilis", "gram_negative_nonfermenter",
              "water-borne nosocomial bacteraemia, peritonitis"),
    Candidate("Brevundimonas diminuta", "gram_negative_nonfermenter",
              "line-associated bacteraemia in immunocompromised hosts"),
    Candidate("Brevundimonas vesicularis", "gram_negative_nonfermenter",
              "bacteraemia, endocarditis"),
    Candidate("Comamonas testosteroni", "gram_negative_nonfermenter",
              "bacteraemia, peritonitis, endocarditis"),
    Candidate("Roseomonas mucosa", "gram_negative_nonfermenter",
              "catheter-related bloodstream infection"),
    Candidate("Ochrobactrum anthropi", "gram_negative_nonfermenter",
              "line-associated bacteraemia",
              "reclassified into Brucella; taxonomy check will flag it"),

    # --- Fastidious gram negatives (HACEK and zoonotic) ---
    Candidate("Eikenella corrodens", "gram_negative_fastidious",
              "HACEK endocarditis, human-bite and head/neck abscess"),
    Candidate("Kingella kingae", "gram_negative_fastidious",
              "paediatric septic arthritis, osteomyelitis, endocarditis"),
    Candidate("Cardiobacterium hominis", "gram_negative_fastidious",
              "HACEK endocarditis"),
    Candidate("Capnocytophaga canimorsus", "gram_negative_fastidious",
              "fulminant sepsis after dog bite, asplenic patients"),
    Candidate("Capnocytophaga ochracea", "gram_negative_fastidious",
              "periodontal disease, neutropenic bacteraemia"),
    Candidate("Streptobacillus moniliformis", "gram_negative_fastidious",
              "rat-bite fever, endocarditis"),

    # --- Gram positives ---
    Candidate("Rothia mucilaginosa", "gram_positive",
              "bacteraemia in neutropenic and CF patients, endocarditis"),
    Candidate("Rothia dentocariosa", "gram_positive",
              "endocarditis, dental infection"),
    Candidate("Gemella haemolysans", "gram_positive",
              "endocarditis, brain abscess"),
    Candidate("Gemella morbillorum", "gram_positive",
              "endocarditis, pleuropulmonary infection"),
    Candidate("Granulicatella adiacens", "gram_positive",
              "nutritionally variant streptococcal endocarditis"),
    Candidate("Abiotrophia defectiva", "gram_positive",
              "culture-negative endocarditis"),
    Candidate("Erysipelothrix rhusiopathiae", "gram_positive",
              "erysipeloid, occupational zoonosis, endocarditis"),
    Candidate("Arcanobacterium haemolyticum", "gram_positive",
              "pharyngitis in adolescents, skin and soft-tissue infection"),
    Candidate("Trueperella pyogenes", "gram_positive",
              "zoonotic pyogenic infection"),
    Candidate("Helcococcus kunzii", "gram_positive",
              "diabetic foot and soft-tissue infection"),
    Candidate("Gordonia bronchialis", "gram_positive_actinomycete",
              "sternal wound and catheter infection, pseudo-outbreaks"),
    Candidate("Tsukamurella paurometabola", "gram_positive_actinomycete",
              "catheter-related bacteraemia, keratitis"),
    Candidate("Dietzia maris", "gram_positive_actinomycete",
              "rare bacteraemia and prosthetic infection"),

    # --- Anaerobes ---
    Candidate("Finegoldia magna", "anaerobe",
              "prosthetic joint infection, diabetic foot, soft-tissue abscess"),
    Candidate("Anaerococcus vaginalis", "anaerobe",
              "polymicrobial soft-tissue and genital tract infection"),
    Candidate("Eggerthella lenta", "anaerobe",
              "anaerobic bacteraemia, intra-abdominal infection",
              "notable intrinsic resistance"),
    Candidate("Bilophila wadsworthia", "anaerobe",
              "appendicitis, intra-abdominal abscess"),
    Candidate("Mobiluncus curtisii", "anaerobe", "bacterial vaginosis"),
    Candidate("Fannyhessea vaginae", "anaerobe", "bacterial vaginosis"),
    Candidate("Alistipes putredinis", "anaerobe", "gut commensal / opportunist"),
    Candidate("Odoribacter splanchnicus", "anaerobe", "gut commensal / opportunist"),
    Candidate("Sutterella wadsworthensis", "anaerobe", "gut commensal / opportunist"),
    Candidate("Desulfovibrio piger", "anaerobe", "gut commensal / opportunist"),
    Candidate("Parabacteroides distasonis", "anaerobe",
              "gut commensal, occasional intra-abdominal isolate",
              "already surfaced by the in-house workbook screen"),
    Candidate("Akkermansia muciniphila", "anaerobe", "gut commensal",
              "already surfaced by the in-house workbook screen"),

    # --- Spirochetes and microaerophiles ---
    Candidate("Leptospira interrogans", "spirochete_or_microaerophile",
              "leptospirosis", "slow growth, special media; poor MIC target"),
    Candidate("Borreliella burgdorferi", "spirochete_or_microaerophile",
              "Lyme disease", "slow growth, special media; poor MIC target"),
    Candidate("Treponema denticola", "spirochete_or_microaerophile", "periodontitis"),
    Candidate("Aliarcobacter butzleri", "spirochete_or_microaerophile",
              "enteritis, bacteraemia"),

    # --- Fungi ---
    Candidate("Trichosporon asahii", "yeast",
              "invasive trichosporonosis in neutropenic patients",
              "intrinsically echinocandin-resistant"),
    Candidate("Magnusiomyces capitatus", "yeast",
              "disseminated infection in haematological malignancy"),
    Candidate("Rhizopus arrhizus", "mould", "mucormycosis",
              "CLSI M38 broth assay, harder than yeast MIC"),
    Candidate("Rhizopus microsporus", "mould", "mucormycosis"),
    Candidate("Mucor circinelloides", "mould", "mucormycosis"),
    Candidate("Lichtheimia corymbifera", "mould", "mucormycosis"),
    Candidate("Scedosporium apiospermum", "mould",
              "pulmonary and disseminated scedosporiosis"),
    Candidate("Lomentospora prolificans", "mould",
              "pan-resistant disseminated infection"),
    Candidate("Exophiala dermatitidis", "mould",
              "phaeohyphomycosis, CF airway colonisation"),
    Candidate("Sporothrix schenckii", "mould", "sporotrichosis"),

    # --- Species unseen inside a trained genus (weaker claim, kept for contrast) ---
    Candidate("Klebsiella variicola", "gram_negative_enterobacterales",
              "bacteraemia; frequently misidentified as K. pneumoniae"),
    Candidate("Acinetobacter pittii", "gram_negative_nonfermenter",
              "nosocomial pneumonia and bacteraemia"),
    Candidate("Acinetobacter nosocomialis", "gram_negative_nonfermenter",
              "nosocomial bacteraemia"),
    Candidate("Enterobacter roggenkampii", "gram_negative_enterobacterales",
              "nosocomial MDR infection"),
    Candidate("Citrobacter braakii", "gram_negative_enterobacterales",
              "UTI, bacteraemia"),
    Candidate("Staphylococcus capitis", "gram_positive",
              "neonatal ICU bacteraemia, prosthetic infection",
              "already surfaced by the in-house workbook screen"),
    Candidate("Staphylococcus caprae", "gram_positive",
              "prosthetic joint and bone infection"),
    Candidate("Streptococcus constellatus", "gram_positive",
              "abscess formation, empyema"),
    Candidate("Enterococcus avium", "gram_positive", "bacteraemia, UTI"),
    Candidate("Corynebacterium amycolatum", "gram_positive",
              "MDR line infection, prosthetic device infection"),
    Candidate("Corynebacterium urealyticum", "gram_positive",
              "encrusted cystitis, MDR urinary infection"),
    Candidate("Nocardia brasiliensis", "gram_positive_actinomycete",
              "cutaneous nocardiosis, mycetoma"),
    Candidate("Nocardia cyriacigeorgica", "gram_positive_actinomycete",
              "pulmonary and disseminated nocardiosis"),
    Candidate("Mycobacterium chimaera", "gram_positive_actinomycete",
              "prosthetic valve and disseminated infection"),
    Candidate("Bacteroides caccae", "anaerobe", "intra-abdominal infection"),
    Candidate("Prevotella bivia", "anaerobe", "pelvic inflammatory disease"),
    Candidate("Candida haemulonii", "yeast",
              "MDR invasive candidiasis, amphotericin-resistant"),
    Candidate("Aspergillus terreus", "mould", "invasive aspergillosis",
              "amphotericin-resistant"),
)


def post_json(url: str, payload: dict, headers: dict, retries: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers
            )
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001 - transient network faults
            last = error
            time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"request failed: {url}") from last


def resolve_taxonomy(names: list[str], cache_path: Path, batch: int = 60) -> dict[str, dict]:
    """Map free-text organism names to their current NCBI name and genus."""
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    pending = [name for name in dict.fromkeys(names) if name not in cache]
    for start in range(0, len(pending), batch):
        chunk = pending[start : start + batch]
        payload = {"taxons": chunk, "returned_content": "COMPLETE"}
        report = post_json(
            NCBI_TAXONOMY, payload, {"Content-Type": "application/json"}
        )
        seen: set[str] = set()
        for entry in report.get("reports", []):
            queries = entry.get("query", []) or []
            taxonomy = entry.get("taxonomy", {})
            classification = taxonomy.get("classification", {})
            resolved = {
                "tax_id": taxonomy.get("tax_id"),
                "current_name": (taxonomy.get("current_scientific_name") or {}).get("name"),
                "genus": (classification.get("genus") or {}).get("name"),
                "family": (classification.get("family") or {}).get("name"),
                "rank": taxonomy.get("rank"),
            }
            for query in queries:
                cache[query] = resolved
                seen.add(query)
        for name in chunk:
            if name not in seen:
                cache[name] = {
                    "tax_id": None,
                    "current_name": None,
                    "genus": None,
                    "family": None,
                    "rank": None,
                }
        cache_path.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.35)
    return cache


def atcc_catalogue(query: str, page_size: int = 100, max_results: int = 400) -> list[dict]:
    """Return de-duplicated ATCC catalogue entries matching a free-text query."""
    headers = {
        "Content-Type": "application/json",
        "Referer": "https://www.atcc.org/search",
        "User-Agent": "ApexOracle-reviewer4-target-screen/1.0",
    }
    by_number: dict[str, dict] = {}
    first = 0
    while first < max_results:
        payload = {
            "q": f'"{query}"',
            "aq": ATCC_TEMPLATES,
            "numberOfResults": page_size,
            "firstResult": first,
            "searchHub": "search",
            "fieldsToInclude": list(ATCC_FIELDS),
        }
        response = post_json(ATCC_SEARCH, payload, headers)
        results = response.get("results", [])
        for result in results:
            raw = result.get("raw", {})
            number = str(raw.get("atccz32xnumber") or "").strip()
            if not number:
                continue
            accepted = raw.get("organismacceptedname") or []
            if isinstance(accepted, str):
                accepted = [accepted]
            by_number.setdefault(
                number,
                {
                    "atcc_number": number,
                    "accepted_name": accepted[0] if accepted else "",
                    "strain_designation": raw.get("strainz32xdesignation") or "",
                    "biosafety_level": raw.get("biosafetyz32xlevel") or "",
                    "has_genome": str(raw.get("genomicz32xdata") or "").strip().lower() == "yes",
                    "genome_url": raw.get("genomicz32xdataz32xurl") or "",
                    "type_strain": str(raw.get("typez32xstrain") or "").strip().lower() == "yes",
                    "web_available": str(raw.get("webz32xavailable") or "").strip().lower() == "true",
                    "isolation_source": raw.get("isolationz32xsource") or "",
                    "category": (raw.get("productz32xcategory") or [""])[0]
                    if isinstance(raw.get("productz32xcategory"), list)
                    else raw.get("productz32xcategory") or "",
                },
            )
        total = response.get("totalCount", 0)
        first += page_size
        if first >= total or not results:
            break
        time.sleep(0.4)
    return list(by_number.values())


def species_of(name: str) -> str:
    """First two tokens, so ``X y subsp. z`` collapses onto the species."""
    return " ".join(str(name).split()[:2])


def training_exposure(repo_root: Path):
    with redirect_stdout(StringIO()):
        prepared = prepare_hierarchical_mic_data(repo_root)
    actual_ids = set(map(str, prepared.genome_or_text_records[:, 1]))
    species_by_id: dict[str, str] = {}
    for strain_id in actual_ids:
        species = prepared.atcc_id_to_species.get(
            strain_id
        ) or prepared.original_strain_to_species.get(strain_id)
        if species is None:
            raise AssertionError(f"No species mapping for training strain {strain_id!r}")
        species_by_id[strain_id] = species
    canonical = canonicalize_training_species(
        set(species_by_id.values()), prepared.taxonomy_aliases
    )
    return species_by_id, canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "reviewer4_unseen_targets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "ncbi_taxonomy_cache.json"

    species_by_id, training_species = training_exposure(REPO_ROOT)
    candidate_names = [candidate.species for candidate in CANDIDATES]

    print(
        f"resolving {len(training_species)} training names and "
        f"{len(candidate_names)} candidates through NCBI Taxonomy"
    )
    taxonomy = resolve_taxonomy(
        sorted(training_species) + candidate_names, cache_path
    )

    trained_current_names: set[str] = set()
    trained_genera: set[str] = set()
    # Family exposure is not a gate, but it must be disclosed: an unseen genus
    # inside a trained family (Providencia next to the trained Proteus) is a
    # weaker novelty claim than an unseen family, and a reviewer will ask.
    trained_family_members: dict[str, set[str]] = defaultdict(set)
    unresolved_training: list[str] = []
    for name in training_species:
        record = taxonomy.get(name, {})
        current = record.get("current_name")
        genus = record.get("genus")
        if current:
            trained_current_names.add(current)
            trained_current_names.add(species_of(current))
        else:
            unresolved_training.append(name)
            trained_current_names.add(name)
        if genus:
            trained_genera.add(genus)
        elif len(name.split()) >= 2:
            trained_genera.add(name.replace("[", "").replace("]", "").split()[0])
        family = record.get("family")
        if family:
            trained_family_members[family].add(current or name)

    rows: list[dict] = []
    strain_rows: list[dict] = []
    for index, candidate in enumerate(CANDIDATES, start=1):
        record = taxonomy.get(candidate.species, {})
        current_name = record.get("current_name") or candidate.species
        genus = record.get("genus") or current_name.split()[0]
        family = record.get("family")
        species_seen = species_of(current_name) in trained_current_names
        genus_seen = genus in trained_genera
        family_members = sorted(trained_family_members.get(family, set())) if family else []

        print(f"[{index}/{len(CANDIDATES)}] ATCC catalogue: {candidate.species}")
        entries = [
            entry
            for entry in atcc_catalogue(candidate.species)
            if species_of(entry["accepted_name"]).lower() == species_of(current_name).lower()
            or species_of(entry["accepted_name"]).lower() == species_of(candidate.species).lower()
        ]
        # Both gates must hold for a strain to be usable: orderable online, and
        # already sequenced so the Evo-2 producer has an input.
        with_genome = [
            entry for entry in entries if entry["has_genome"] and entry["web_available"]
        ]
        for entry in entries:
            strain_rows.append({"species": candidate.species, **entry})

        levels = sorted({entry["biosafety_level"] for entry in entries if entry["biosafety_level"]})
        rows.append(
            {
                "species": candidate.species,
                "current_name": current_name,
                "current_genus": genus,
                "tax_id": record.get("tax_id"),
                "family": family,
                "trained_species_in_family": len(family_members),
                "trained_family_examples": "; ".join(family_members[:4]),
                "group": candidate.group,
                "clinical_syndrome": candidate.syndrome,
                "species_in_training": species_seen,
                "genus_in_training": genus_seen,
                "unseen_level": (
                    "trained" if species_seen else ("species_only" if genus_seen else "genus")
                ),
                "atcc_products": len(entries),
                "atcc_orderable_with_genome": len(with_genome),
                "atcc_type_strain_available": any(entry["type_strain"] for entry in entries),
                "biosafety_levels": "; ".join(levels),
                "atcc_numbers_with_genome": "; ".join(
                    entry["atcc_number"] for entry in sorted(with_genome, key=lambda e: e["atcc_number"])
                ),
                "note": candidate.note,
            }
        )
        time.sleep(0.4)

    table = pd.DataFrame(rows)
    strains = pd.DataFrame(strain_rows)

    # Panel feasibility: a species-level efficacy claim needs several genome-ready
    # isolates, not one type strain.
    table["panel_ready"] = (table["unseen_level"] != "trained") & (
        table["atcc_orderable_with_genome"] >= 3
    )
    order = {"genus": 0, "species_only": 1, "trained": 2}
    table = table.sort_values(
        by=["unseen_level", "atcc_orderable_with_genome", "species"],
        key=lambda column: column.map(order) if column.name == "unseen_level" else column,
        ascending=[True, False, True],
    )

    candidates_path = output_dir / "unseen_atcc_pathogen_candidates.csv"
    strains_path = output_dir / "unseen_atcc_pathogen_strains.csv"
    table.to_csv(candidates_path, index=False)
    strains.to_csv(strains_path, index=False)

    genus_totals: dict[str, int] = defaultdict(int)
    for _, row in table.iterrows():
        if row["unseen_level"] == "genus":
            genus_totals[row["current_genus"]] += int(row["atcc_orderable_with_genome"])

    summary = {
        "scope": "reviewer 4 unseen-species/genus ATCC target screen",
        "generated_on": time.strftime("%Y-%m-%d"),
        "training_exposure": {
            "actual_unique_strain_ids": len(species_by_id),
            "producer_era_species_names": len(set(species_by_id.values())),
            "canonical_species_with_aliases": len(training_species),
            "current_genera_after_ncbi_resolution": len(trained_genera),
            "training_names_unresolved_by_ncbi": sorted(unresolved_training),
        },
        "sources": {
            "taxonomy": NCBI_TAXONOMY,
            "atcc_catalogue": ATCC_SEARCH,
            "atcc_templates_filter": ATCC_TEMPLATES,
        },
        "counts": {
            "candidates_screened": len(table),
            "genus_level_unseen": int((table["unseen_level"] == "genus").sum()),
            "species_level_unseen_only": int((table["unseen_level"] == "species_only").sum()),
            "already_trained": int((table["unseen_level"] == "trained").sum()),
            "panel_ready_candidates": int(table["panel_ready"].sum()),
        },
        "genus_unseen_genome_ready_products_by_genus": dict(
            sorted(genus_totals.items(), key=lambda item: -item[1])
        ),
        "evidence_boundaries": [
            "Unseen calls are made on NCBI current names and genera, not on the "
            "producer-era strings, so renamed taxa cannot be reported as unseen.",
            "'atcc_orderable_with_genome' counts catalogue entries that are both web-orderable and flagged with ATCC "
            "Genome Portal data; the assembly still has to be downloaded and run "
            "through the external Evo-2 producer before generation.",
            "ATCC catalogue availability and biosafety level were read from the "
            "public product index and are not a substitute for checking the live "
            "product page, shipping restrictions or institutional biosafety approval.",
            "Clinical relevance and assay feasibility annotations are curatorial and "
            "must be confirmed by the microbiology team before any purchase.",
        ],
        "outputs": {
            "candidates_csv": display_path(candidates_path),
            "strains_csv": display_path(strains_path),
            "taxonomy_cache": display_path(cache_path),
        },
    }
    summary_path = output_dir / "unseen_atcc_pathogen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nCandidates:")
    print(
        table[
            [
                "species",
                "unseen_level",
                "atcc_products",
                "atcc_orderable_with_genome",
                "panel_ready",
                "biosafety_levels",
                "clinical_syndrome",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
