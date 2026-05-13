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
      - perturbed_gene_summary: text summary of gene_perturbed
      - gene_monitored_rn_summaries: summaries of k random STRING neighbors of gene_monitored
      - potential_genes: pipe-separated candidate gene list for the dual task
        (neighbors of gene_monitored + n_random random KG genes + gene_monitored itself)
    """
    perturbed_summaries, monitored_rn_summaries, potential_genes_list = [], [], []
    all_kg_genes = list(kg.keys())

    summary_cache   = {}  # gene_p → pert_summary string
    rn_cache        = {}  # gene_m → rn_str string
    neighbors_cache = {}  # gene_m → set of STRING neighbors
    warned          = set()  # genes already warned about missing summaries

    for _, row in df.iterrows():
        gene_p = row["gene_perturbed"].upper()
        gene_m = row["gene_monitored"].upper()

        if gene_p not in summary_cache:
            summary_cache[gene_p] = gs.get(gene_p, gs.get(gene_p.lower(), "No summary available."))
        perturbed_summaries.append(summary_cache[gene_p])

        if gene_m not in rn_cache:
            rn = fetch_rn_summaries(gene_m, k=k, ref_kg=kg, mode="g",
                                    gene_summaries=gs, warned=warned)
            rn_cache[gene_m] = (
                "\n".join(f"- {g}: {desc}" for g, desc in rn.items())
                if rn else "No neighbor summaries found."
            )
        monitored_rn_summaries.append(rn_cache[gene_m])

        if gene_m not in neighbors_cache:
            neighbors_cache[gene_m] = (
                set(fetch_rn_string(gene_m, kg)) if gene_m in kg else set()
            )
        neighbors_m = neighbors_cache[gene_m]

        pool = neighbors_m | set(random.sample(all_kg_genes, min(n_random, len(all_kg_genes))))
        pool.discard(gene_m)
        candidates = list(pool)
        random.shuffle(candidates)
        candidates.append(gene_m)
        potential_genes_list.append("|".join(candidates))

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
        f"You have just predicted that a perturbation of {gene_perturbed} {addon} likely to "
        f"induce differential expression of a mysterious gene. Based on the context below and "
        f"your internal knowledge, predict the mysterious gene.\n"
        f"For context, here is a summary of {gene_perturbed}:\n{perturbed_gene_summary}\n"
        f"For additional context, here are descriptions of randomly selected neighbors of the "
        f"mysterious gene in the knowledge graph:\n{gene_monitored_rn_summaries}\n"
        f"The answer must be one of the following genes:\n{candidates}\n"
        f"Answer with only the gene name."
    )
