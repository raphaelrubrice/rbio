import json
import os
from pathlib import Path
import argparse
import random as rd 

FILE_FOLDER = Path(__file__).parent
PATH_KG = FILE_FOLDER / "kg"
PATH_SUMMARIES = FILE_FOLDER / "gene_summary"
ALLOWED_KG_TYPES = ["bioplex", "corum_gsea", "corum", "ensembl", 
                    "go_dict", "go_gsea", "go", "reactome_gsea", 
                    "reactome", "string", "uniprot"]

def load_gs(gs_path=None):
    """
    Load gene summaries.
    """
    if gs_path is None:
        gs_path = PATH_SUMMARIES / "desc_gene.json"
    
    with open(gs_path, "r") as f:
        gs = json.load(f)

    if isinstance(gs, list):
        return gs[0]
    return gs

def load_ps(ps_path=None):
    """
    Load perturbation summaries.
    """
    if ps_path is None:
        ps_path = PATH_SUMMARIES / "desc_pert.json"
    
    with open(ps_path, "r") as f:
        ps = json.load(f)

    if isinstance(ps, list):
        return ps[0]
    return ps

def load_kg(kg_type: str) -> dict:
    """
    Loads one knowledge graph. Returns a dict.
    """
    kg_type = kg_type.lower()
    assert kg_type in ALLOWED_KG_TYPES, f"Unrecognized knowledge graph : {kg_type}, must be one of {ALLOWED_KG_TYPES}"

    path = PATH_KG / f"{kg_type}.json"
    print(f"Loading {kg_type}..")
    with open(path, "r") as f:
        kg = json.load(f)

    if isinstance(kg, list):
        return kg[0]
    return kg

def make_kg_entries():
    kg_entries = {}
    for kg_type in ALLOWED_KG_TYPES:
        kg = load_kg(kg_type)
        kg_entries[kg_type] = list(kg.keys())
    
    path = FILE_FOLDER / "kg_entries.json"
    with open(path, 'w') as f:
        json.dump(kg_entries, f)
    return kg_entries

def query_gene(g: str, kg_entries: dict = None):
    if kg_entries == None:
        try:
            path = FILE_FOLDER / "kg_entries.json"
            with open(path, "r") as f:
                kg_entries = json.load(f)[0]
        except:
            kg_entries = make_kg_entries()

    hits = {}
    print(f"\nSearching for {g}..")
    for kg_type in kg_entries.keys():
        if g in kg_entries[kg_type]:
            kg = load_kg(kg_type)
            hits[kg_type] = kg[g]
    return hits

def get_coverage(g_set, kg_type):
    kg = load_kg(kg_type)
    g_set_len = len(g_set)
    coverage = 0
    found = []
    for g in g_set:
        if g.upper() in kg.keys():
            coverage += 1 
            found.append(g.upper())
    coverage = coverage * 100/g_set_len
    print(f"\n{kg_type.upper()} Knowledge graph covers {coverage:.2f}% of the input gene set.")
    return coverage, found

def all_coverage(g_set):
    res = {}
    combine = []
    for kg_type in ALLOWED_KG_TYPES:
        coverage, found = get_coverage(g_set, kg_type)
        res[kg_type] = coverage
        combine.extend(found)
    all_found = set(combine)
    combined_coverage = 0
    for g in g_set:
        if g.upper() in all_found:
            combined_coverage += 1
    combined_coverage = combined_coverage * 100 / len(g_set)
    print(f"\nThe combined coverage across all databases is {combined_coverage:.2f}")
    return res

def fetch_rn_string(g, kg):
    """
    Handles proper fetching of neighboring nodes from the STRING KG
    """
    res = kg[g]

    nn_list = []
    for entry in res:
        nn_list.append(entry[0])
    return nn_list

def fetch_rn_summaries(g, k=3, ref_kg="string", mode="g", s_path=None,
                       gene_summaries=None, warned=None):
    """
    Retrieves randomly k summaries amongst the gene's neighbors in the Knowledge graph.

    gene_summaries: pre-loaded summaries dict; skips load_gs()/load_ps() when provided.
    warned: a set of gene names already warned about; each missing gene is printed at most once.
    """
    if isinstance(ref_kg, str):
        ref_kg = load_kg(ref_kg)

    if gene_summaries is not None:
        summaries = gene_summaries
    elif mode == "g":
        summaries = load_gs(s_path)
    else:
        summaries = load_ps(s_path)

    nn_summaries = {}
    if g in ref_kg.keys():
        nn_list = fetch_rn_string(g, ref_kg)
        nn_list = list(set(nn_list).intersection(set(summaries.keys())))
        if len(nn_list) == 0:
            if warned is None or g not in warned:
                print(f"[WARNING] No summaries found for query {g}.")
                if warned is not None:
                    warned.add(g)
            return nn_summaries
        nn_list = rd.sample(nn_list, min(k, len(nn_list)))
        for nn_g in nn_list:
            nn_summaries[nn_g] = summaries[nn_g]
    return nn_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", type=str, help="Gene name or ID.")

    args = parser.parse_args()

    print(query_gene(args.g)) 


