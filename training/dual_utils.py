import re
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "PerturbQA"))
from query_kg import load_gs, load_kg, fetch_rn_summaries, fetch_rn_string


def load_kg_data():
    """Load gene summaries and STRING KG once. Returns (gs, kg)."""
    gs = load_gs()
    kg = load_kg("string")
    return gs, kg


def add_dual(df, gs, kg, k=3, n_random=5):
    """Add dual task columns to the dataset DataFrame.

    Adds:
      - perturbed_gene_summary: text summary of gene_perturbed
      - gene_monitored_rn_summaries: summaries of k random STRING neighbors of gene_monitored,
        with any mention of gene_monitored replaced by [mysterious gene]
      - potential_genes: pipe-separated candidate gene list for the dual task
        (gene_monitored at a random position + n_random random genes drawn from the universe
        excluding gene_monitored, its STRING neighbors, and gene_perturbed)
    """
    perturbed_summaries, monitored_rn_summaries, potential_genes_list = [], [], []
    all_kg_genes = list(kg.keys())

    summary_cache   = {}  # gene_p → summary string
    rn_cache        = {}  # gene_m → anonymized rn_str
    neighbors_cache = {}  # gene_m → set of STRING neighbors
    eligible_cache  = {}  # gene_m → list of KG genes excluding gene_m and its neighbors
    warned          = set()

    for _, row in df.iterrows():
        gene_p = row["gene_perturbed"].upper()
        gene_m = row["gene_monitored"].upper()

        if gene_p not in summary_cache:
            summary_cache[gene_p] = gs.get(gene_p, gs.get(gene_p.lower(), "No summary available."))
        perturbed_summaries.append(summary_cache[gene_p])

        if gene_m not in rn_cache:
            rn = fetch_rn_summaries(gene_m, k=k, ref_kg=kg, mode="g",
                                    gene_summaries=gs, warned=warned)
            rn_str = (
                "\n".join(f"- {g}: {desc}" for g, desc in rn.items())
                if rn else "No neighbor summaries found."
            )
            rn_cache[gene_m] = re.sub(re.escape(gene_m), "[mysterious gene]", rn_str,
                                      flags=re.IGNORECASE)
        monitored_rn_summaries.append(rn_cache[gene_m])

        if gene_m not in neighbors_cache:
            neighbors_cache[gene_m] = (
                set(fetch_rn_string(gene_m, kg)) if gene_m in kg else set()
            )
        neighbors_m = neighbors_cache[gene_m]

        if gene_m not in eligible_cache:
            eligible_cache[gene_m] = [g for g in all_kg_genes if g not in (neighbors_m | {gene_m})]
        eligible_base = eligible_cache[gene_m]

        # gene_p exclusion handled on the small draw — avoids O(N_kg) per row
        draw = random.sample(eligible_base, min(n_random + 1, len(eligible_base)))
        pool = [g for g in draw if g != gene_p][:n_random]
        if len(pool) < n_random:
            taken = set(pool) | {gene_p}
            fill = [g for g in eligible_base if g not in taken]
            pool += random.sample(fill, min(n_random - len(pool), len(fill)))
        insert_pos = random.randint(0, len(pool))
        pool.insert(insert_pos, gene_m)
        potential_genes_list.append("|".join(pool))

    df = df.copy()
    df["perturbed_gene_summary"]      = perturbed_summaries
    df["gene_monitored_rn_summaries"] = monitored_rn_summaries
    df["potential_genes"]             = potential_genes_list
    return df


def build_dual_prompt(gene_perturbed, answer, perturbed_gene_summary,
                      gene_monitored_rn_summaries, potential_genes_str):
    """Build the dual-task prompt conditioned on the first completion's answer."""
    addon = "is" if answer == "yes" else "is not"
    candidates = potential_genes_str.replace("|", ", ")
    return (
        f"You have just predicted that a perturbation of the gene {gene_perturbed} {addon} likely to "
        f"induce differential expression of a another mysterious gene (i.e. this other gene is not {gene_perturbed} itself). Based on the context below and "
        f"your internal knowledge, predict this other mysterious gene.\n"
        f"For context, here is a summary of {gene_perturbed}:\n{perturbed_gene_summary}\n"
        f"For additional context, here are descriptions of some known neighbors of the mysterious "
        f"gene in the knowledge graph. These neighbor genes are clues only — they are NOT the "
        f"answer:\n{gene_monitored_rn_summaries}\n"
        f"You must answer with exactly one gene name from the following list and nothing else. "
        f"Any answer not in this list is wrong:\n{candidates}\n"
    )


if __name__ == "__main__":
    import pandas as pd

    repo_root = Path(__file__).parent.parent
    csv_path = repo_root / "k562-train-v0.3.0.csv"

    print("Loading data...")
    gs, kg = load_kg_data()
    df_raw = pd.read_csv(csv_path).head(20)
    df = add_dual(df_raw, gs, kg)

    failures = []
    for i, row in df.iterrows():
        gene_m = row["gene_monitored"].upper()
        gene_p = row["gene_perturbed"].upper()
        candidates = row["potential_genes"].split("|")
        rn_text = row["gene_monitored_rn_summaries"]

        neighbors_m = set(fetch_rn_string(gene_m, kg)) if gene_m in kg else set()

        if candidates.count(gene_m) != 1:
            failures.append(f"Row {i}: gene_monitored '{gene_m}' count={candidates.count(gene_m)}")

        bad = [c for c in candidates if c != gene_m and c in neighbors_m]
        if bad:
            failures.append(f"Row {i}: neighbors in candidates: {bad}")

        if gene_p in candidates:
            failures.append(f"Row {i}: gene_perturbed '{gene_p}' found in candidates")

        if re.search(re.escape(gene_m), rn_text, flags=re.IGNORECASE):
            failures.append(f"Row {i}: '{gene_m}' found literally in rn_summaries")

        if not row["perturbed_gene_summary"] or row["perturbed_gene_summary"] == "No summary available.":
            failures.append(f"Row {i}: perturbed_gene_summary is empty or missing")

    sample = df.iloc[0]
    prompt = build_dual_prompt(
        sample["gene_perturbed"], "yes",
        sample["perturbed_gene_summary"],
        sample["gene_monitored_rn_summaries"],
        sample["potential_genes"],
    )
    if "here is a summary of" not in prompt:
        failures.append("build_dual_prompt missing 'here is a summary of' block")

    if failures:
        print(f"FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"All checks passed on {len(df)} rows.")
