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


def add_dual(df, gs, kg, k=3, n_random=25):
    """Add dual task columns to the dataset DataFrame.

    Adds:
      - gene_monitored_rn_summaries: summaries of k random STRING neighbors of gene_monitored,
        with any mention of gene_monitored replaced by [mysterious gene]
      - potential_genes: pipe-separated candidate gene list for the dual task
        (gene_monitored at a random position + n_random random genes drawn from the universe
        excluding gene_monitored, its STRING neighbors, and gene_perturbed)
    """
    monitored_rn_summaries, potential_genes_list = [], []
    all_kg_genes = list(kg.keys())

    rn_cache        = {}  # gene_m → anonymized rn_str
    neighbors_cache = {}  # gene_m → set of STRING neighbors
    warned          = set()  # genes already warned about missing summaries

    for _, row in df.iterrows():
        gene_p = row["gene_perturbed"].upper()
        gene_m = row["gene_monitored"].upper()

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

        excluded = neighbors_m | {gene_m, gene_p}
        eligible = [g for g in all_kg_genes if g not in excluded]
        pool = random.sample(eligible, min(n_random, len(eligible)))
        insert_pos = random.randint(0, len(pool))
        pool.insert(insert_pos, gene_m)
        potential_genes_list.append("|".join(pool))

    df = df.copy()
    df["gene_monitored_rn_summaries"] = monitored_rn_summaries
    df["potential_genes"]             = potential_genes_list
    return df


def build_dual_prompt(gene_perturbed, answer, gene_monitored_rn_summaries, potential_genes_str):
    """Build the dual-task prompt conditioned on the first completion's answer."""
    addon = "is" if answer == "yes" else "is not"
    candidates = potential_genes_str.replace("|", ", ")
    return (
        f"You have just predicted that a perturbation of {gene_perturbed} {addon} likely to "
        f"induce differential expression of a mysterious gene. Based on the context below and "
        f"your internal knowledge, predict the mysterious gene.\n"
        f"For additional context, here are descriptions of randomly selected neighbors of the "
        f"mysterious gene in the knowledge graph:\n{gene_monitored_rn_summaries}\n"
        f"The answer must be one of the following genes:\n{candidates}\n"
        f"Answer with only the gene name."
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

        # gene_monitored appears exactly once
        if candidates.count(gene_m) != 1:
            failures.append(f"Row {i}: gene_monitored '{gene_m}' count={candidates.count(gene_m)} in candidates")

        # no distractor is a neighbor of gene_monitored
        bad_neighbors = [c for c in candidates if c != gene_m and c in neighbors_m]
        if bad_neighbors:
            failures.append(f"Row {i}: neighbors in candidates: {bad_neighbors}")

        # gene_perturbed not in candidates
        if gene_p in candidates:
            failures.append(f"Row {i}: gene_perturbed '{gene_p}' found in candidates")

        # gene_monitored not literally in rn summaries
        if re.search(re.escape(gene_m), rn_text, flags=re.IGNORECASE):
            failures.append(f"Row {i}: '{gene_m}' found literally in rn_summaries")

    # build_dual_prompt has no perturbed gene summary block
    sample = df.iloc[0]
    prompt = build_dual_prompt(
        sample["gene_perturbed"], "yes",
        sample["gene_monitored_rn_summaries"], sample["potential_genes"]
    )
    if "here is a summary of" in prompt:
        failures.append("build_dual_prompt still contains 'here is a summary of' block")

    if failures:
        print(f"FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
    else:
        print(f"All checks passed on {len(df)} rows.")
