# app/main.py
import os
import rdflib
import pandas as pd
import streamlit as st
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from city_matching.city_matching import compute_city_weight
from type_classification.project_type_matching import match_project_type
import random
from rdflib import Namespace, RDF
import hmac

ECRM = Namespace("http://erlangen-crm.org/240307/")
# -----------------------------
# Config
# -----------------------------
TTL_PATH_DEFAULT = "robin.ttl"

PREFIXES = """
PREFIX : <http://www.semanticweb.org/pyt/ontologies/2025/10/robin#>
PREFIX ecrm: <http://erlangen-crm.org/240307/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xml: <http://www.w3.org/XML/1998/namespace>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
IRI_DEFAULT = "http://erlangen-crm.org/240307/"
ttl_path = TTL_PATH_DEFAULT
iri_base = IRI_DEFAULT
# -----------------------------
# Caching helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_graph(ttl_path: str) -> rdflib.Graph:
    g = rdflib.Graph()
    g.parse(ttl_path, format="turtle")
    return g

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)
# -----------------------------
# Random initial-stage recommend
# -----------------------------
def get_project_classes(g: rdflib.Graph) -> list[str]:
    classes = []
    for s in g.subjects(RDF.type, ECRM.E303_Project_Class):
        classes.append(str(s).split("/")[-1].split("#")[-1])
    return sorted(set(classes))

def retrieve_projects_from_one_class(g: rdflib.Graph, project_class: str) -> list[str]:
    query = f"""
{PREFIXES}
SELECT ?project
WHERE {{
  ?project a ecrm:E7_Activity ;
           ecrm:P503_has_project_class ecrm:{project_class} .
}}
"""
    query_result = list(g.query(query))
    return [str(row.project).split("/")[-1].split("#")[-1] for row in query_result]

def retrieve_all_projects_by_class(g: rdflib.Graph, project_classes: list[str]) -> dict[str, list[str]]:
    return {pc: retrieve_projects_from_one_class(g, pc) for pc in project_classes}

def random_pick_one_per_class(all_projects_by_class: dict[str, list[str]]) -> dict[str, str]:
    selection = {}
    for cls, lst in all_projects_by_class.items():
        if lst:
            selection[cls] = random.choice(lst)
    return selection

def build_random_initial_recommendations_df(g: rdflib.Graph, iri_base: str):
    project_classes = get_project_classes(g)
    all_projects_by_class = retrieve_all_projects_by_class(g, project_classes)
    selection = random_pick_one_per_class(all_projects_by_class)

    selected_projects = list(selection.values())
    df_info = retrieve_full_information_of_projects(g, selected_projects, iri_base=iri_base)

    if not df_info.empty:
        project_to_class = {proj: cls for cls, proj in selection.items()}
        df_info.insert(0, "project_class", df_info["project"].map(project_to_class))

    return selection, df_info



# -----------------------------
# KG retrieval
# -----------------------------
def retrieve_projects_with_the_same_project_type(g: rdflib.Graph, project_type: str):
    query = f"""
    {PREFIXES}
    SELECT ?project ?description
    WHERE {{
      ?project a ecrm:E7_Activity ;
               ecrm:P502_has_project_type ecrm:{project_type} ;
               ecrm:P3_has_note ?description .
    }}
    """
    query_result = list(g.query(query))

    projects, descriptions = [], []
    for row in query_result:
        # keep local name for your later IRI + proj concatenation
        proj_local = str(row.project).split("/")[-1].split("#")[-1]
        projects.append(proj_local)

        # P3_has_note is usually a literal -> keep full text
        descriptions.append(str(row.description))

    return projects, descriptions

def retrieve_full_information_of_projects(g: rdflib.Graph, projects: list[str], iri_base: str):
    rows = []

    for proj in projects:
        project_iri = iri_base + proj

        query = f"""
        {PREFIXES}
        SELECT ?project ?description ?project_type ?city ?website
        WHERE {{
        BIND(<{project_iri}> AS ?project)
        
        ?project a ecrm:E7_Activity ;
                   ecrm:P3_has_note ?description .
        
        OPTIONAL {{ ?project ecrm:P502_has_project_type ?project_type . }}
        OPTIONAL {{ ?project ecrm:P7_took_place_at ?city . }}
        OPTIONAL {{ ?project ecrm:P90_has_value ?website . }}
        
        
        }}
        """
        qr = list(g.query(query))
        print("query: ", query)
        if not qr:
            rows.append(
                dict(
                    project=proj,
                    description="",
                    project_type="",
                    city="",
                    website="",
                )
            )
            continue

        # aggregate operators/sponsors to avoid duplicated rows
        description, project_type, city, website = "", "", "", ""

        for r in qr:
            if not description and r.description is not None:
                description = str(r.description)

            if not project_type and r.project_type is not None:
                project_type = str(r.project_type).split("/")[-1].split("#")[-1]

            if not city and r.city is not None:
                city = str(r.city).split("/")[-1].split("#")[-1]

            if not website and r.website is not None:
                website = str(r.website)

        rows.append(
            dict(
                project=str(qr[0].project).split("/")[-1].split("#")[-1],
                description=description,
                project_type=project_type,
                city=city,
                website=website
            )
        )

    return pd.DataFrame(rows)

# -----------------------------
# Similarity
# -----------------------------
def compute_similarity_batched(project_proposal: str, descriptions: list[str], model: SentenceTransformer) -> list[float]:
    proposal_emb = model.encode(project_proposal, convert_to_tensor=True)
    desc_embs = model.encode(descriptions, convert_to_tensor=True)
    sims = util.cos_sim(proposal_emb, desc_embs)[0].cpu().tolist()  # one score per description
    return [float(x) for x in sims]

# -----------------------------
# Build recommendations (returns project INFO, not scores)
# -----------------------------

def build_ranked_recommendations(
    g: rdflib.Graph,
    model: SentenceTransformer,
    project_proposal: str,
    top_k: int,
    iri_base: str,
):
    reasoning = {}

    # Step 1: prefilter by project type
    project_type = match_project_type(project_proposal)
    reasoning["matched_project_type"] = project_type

    projects, descriptions = retrieve_projects_with_the_same_project_type(g, project_type)
    reasoning["candidates_count"] = len(projects)

    if not projects:
        empty_cols = ["project", "description", "project_type", "city", "website"]
        return reasoning, pd.DataFrame(columns=empty_cols)

    top_k = min(int(top_k), len(projects))
    reasoning["effective_top_k"] = top_k

    # Step 2: similarity for ALL candidates (kept per project)
    similarity_scores = compute_similarity_batched(project_proposal, descriptions, model)

    df_sim = pd.DataFrame(
        {"project": projects, "similarity": similarity_scores}
    ).sort_values("similarity", ascending=False)

    reasoning["top_similarity_preview"] = df_sim.head(min(20, len(df_sim))).to_dict(orient="records")

    # Select top_k by similarity
    df_top = df_sim.head(top_k).copy()
    projects_to_recommend = df_top["project"].tolist()

    # Step 3: city weights + rerank (kept per project)
    city_weights = compute_city_weight(project_proposal, projects_to_recommend)
    df_top["city_weight"] = [float(w) for w in city_weights]
    df_top["rerank_score"] = df_top["similarity"] * df_top["city_weight"]

    # Final order after reranking
    df_top = df_top.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    final_projects = df_top["project"].tolist()
    reasoning["final_projects"] = final_projects

    # Store full per-project scoring table in reasoning
    reasoning["scores_table"] = df_top.to_dict(orient="records")

    # Step 4: fetch full info and merge scores so you can display them
    df_info = retrieve_full_information_of_projects(g, final_projects, iri_base=iri_base)

    # Merge keeps the same order as df_info by joining on project
    df_info = df_info.merge(
        df_top[["project", "similarity", "city_weight", "rerank_score"]],
        on="project",
        how="left",
    )

    return reasoning, df_info
def expand_projects_from_seeds(g: rdflib.Graph,seed_projects: list[str],iri_base: str,max_extra: int = 20):
    """
    Given seed projects, find other projects sharing:
      - P11_had_participant
      - P16_used_specific_object
      - P103_was_intended_for
    Returns:
      df_edges: edges explaining why (seed, project, relation, shared_entity)
      df_expanded_info: full project info for expanded projects + simple ranking columns
    """
    if not seed_projects:
        return pd.DataFrame(), pd.DataFrame()

    seed_iris = [f"<{iri_base}{p}>" for p in seed_projects]
    values_block = " ".join(seed_iris)

    query = f"""
    {PREFIXES}
    SELECT DISTINCT ?seed ?other ?shared ?rel
    WHERE {{
      VALUES ?seed {{ {values_block} }}
    
      {{
        ?seed ecrm:P11_had_participant ?shared .
        ?other a ecrm:E7_Activity ;
               ecrm:P11_had_participant ?shared .
        BIND("P11_had_participant" AS ?rel)
      }}
      UNION
      {{
        ?seed ecrm:P16_used_specific_object ?shared .
        ?other a ecrm:E7_Activity ;
               ecrm:P16_used_specific_object ?shared .
        BIND("P16_used_specific_object" AS ?rel)
      }}
      UNION
      {{
        ?seed ecrm:P103_was_intended_for ?shared .
        ?other a ecrm:E7_Activity ;
               ecrm:P103_was_intended_for ?shared .
        BIND("P103_was_intended_for" AS ?rel)
      }}
    
      FILTER(?other != ?seed)
    }}
    """

    rows = []
    for r in g.query(query):
        rows.append(
            {
                "seed": str(r.seed).split("/")[-1].split("#")[-1],
                "project": str(r.other).split("/")[-1].split("#")[-1],
                "relation": str(r.rel),
                "shared_entity": str(r.shared).split("/")[-1].split("#")[-1],
            }
        )

    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df_edges = pd.DataFrame(rows)

    # remove seeds themselves
    df_edges = df_edges[~df_edges["project"].isin(seed_projects)].copy()
    if df_edges.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Rank expanded candidates (separate from your main list)
    df_rank = (
        df_edges.groupby("project")
        .agg(
            shared_links=("shared_entity", "nunique"),
            seeds=("seed", "nunique"),
            relations=("relation", "nunique"),
        )
        .reset_index()
        .sort_values(["shared_links", "seeds", "relations"], ascending=False)
        .head(int(max_extra))
        .reset_index(drop=True)
    )

    df_expanded_info = retrieve_full_information_of_projects(g, df_rank["project"].tolist(), iri_base=iri_base).merge(df_rank, on="project", how="left")

    return df_edges, df_expanded_info

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ROBIN – Similar Project Recommender", layout="wide")

# -----------------------------
# Session state
# -----------------------------
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None  # "idea" or "random"

if "idea_results" not in st.session_state:
    st.session_state.idea_results = None

if "idea_reasoning" not in st.session_state:
    st.session_state.idea_reasoning = None

if "idea_proposal" not in st.session_state:
    st.session_state.idea_proposal = ""

if "idea_expanded_results" not in st.session_state:
    st.session_state.idea_expanded_results = None

if "idea_edges" not in st.session_state:
    st.session_state.idea_edges = None

if "random_results" not in st.session_state:
    st.session_state.random_results = None

if "random_selection" not in st.session_state:
    st.session_state.random_selection = None

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
    <style>
      :root {
        --robin-blue: #04316A;
        --robin-blue-dark: #032a5a;
        --robin-border: #4b4b4b;
        --text-dark: #1f2937;
        --text-muted: #5b6573;
        --bg-light: #f7f8fa;
      }

      .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
      }

      .case-wrapper {
        border: 3px solid var(--robin-border);
        border-radius: 0px;
        background: white;
        padding: 0;
        margin-bottom: 0;
      }

      .case-inner {
        padding: 28px 46px 30px 46px;
      }

      .case-divider {
        border-top: 3px solid var(--robin-border);
        margin: 0;
      }

      .case-title {
        font-size: 22px;
        font-weight: 800;
        color: var(--robin-blue);
        margin-bottom: 14px;
      }

      .case-text {
        font-size: 15px;
        color: var(--text-muted);
        line-height: 1.6;
        margin-bottom: 20px;
      }

      .input-top-space {
        margin-top: 10px;
      }

      div.stButton > button {
        border-radius: 12px !important;
        font-weight: 700 !important;
        min-height: 52px !important;
      }

      div.stButton > button[kind="primary"] {
        background-color: var(--robin-blue) !important;
        color: white !important;
        border: 1px solid var(--robin-blue) !important;
      }

      div.stButton > button[kind="secondary"] {
        background-color: var(--robin-blue) !important;
        color: white !important;
        border: 1px solid var(--robin-blue) !important;
      }

      textarea {
        font-size: 18px !important;
        line-height: 1.65 !important;
        border-radius: 10px !important;
      }

      label[data-testid="stWidgetLabel"] p {
        font-weight: 600 !important;
        color: var(--text-dark) !important;
      }

      [data-testid="stTextArea"] textarea {
        background-color: #f2f4f7 !important;
      }

      [data-testid="stSidebar"] {
        border-left: 1px solid rgba(4, 49, 106, 0.10);
      }

      .result-section {
        margin-top: 26px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("Settings")
    LM_model = st.text_input("Language model", value="mistral-small-latest")
    model_name = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    top_k = st.slider("Number of recommendations", 1, 15, 5, 1)
    show_reasoning = st.toggle("Show reasoning steps", value=True)
    expand_more = st.toggle("Show further recommendations", value=True)
    expand_k = st.slider("Number of further recommendations", 1, 10, 5, 1)

# -----------------------------
# HERO HEADER
# -----------------------------
st.markdown(
    """
    <style>
      .hero-box {
        border-radius: 20px;
        padding: 26px 30px;
        background: linear-gradient(135deg, #EEF4FB 0%, #FFFFFF 75%);
        border: 1px solid rgba(4, 49, 106, 0.12);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        margin-bottom: 20px;
      }

      .hero-title {
        font-size: 30px;
        font-weight: 800;
        color: #04316A;
        margin-bottom: 10px;
      }

      .hero-subtitle {
        font-size: 16px;
        color: #5B6573;
        line-height: 1.6;
      }
    </style>

    <div class="hero-box">
        <div class="hero-title">
            ROBIN – Similar Projects Recommender
        </div>
        <div class="hero-subtitle">
            Discover civic innovation projects in two ways: describe your own idea and find similar projects,
            or explore random examples to get inspired.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# CASE 1
# -----------------------------
st.markdown('<div class="case-inner">', unsafe_allow_html=True)

st.markdown(
    """
    <div class="case-title">💡 Case 1: I already have an idea</div>
    <div class="case-text">
      Describe your civic innovation idea in a few sentences. The system retrieves similar projects and further related recommendations.
    </div>
    """,
    unsafe_allow_html=True,
)

proposal = st.text_area(
    "Your idea",
    height=170,
    placeholder="e.g., a public bookcase in the neighborhood to promote reading culture...",
    key="proposal_input",
)

run_btn = st.button(
    "Recommend similar projects",
    type="primary",
    use_container_width=True,
    key="recommend_btn",
)

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# CASE 1 COMPUTE
# -----------------------------
if run_btn:
    if not proposal.strip():
        st.error("Please enter an idea.")
        st.stop()

    if not os.path.exists(ttl_path):
        st.error(f"TTL file not found: {ttl_path}")
        st.stop()

    with st.spinner("Loading graph and model..."):
        g = load_graph(ttl_path)
        model = load_embedding_model(model_name)

    with st.spinner("Retrieving and ranking projects..."):
        reasoning, df_info = build_ranked_recommendations(
            g=g,
            model=model,
            project_proposal=proposal.strip(),
            top_k=top_k,
            iri_base=iri_base,
        )

    st.session_state.active_mode = "idea"
    st.session_state.idea_results = df_info
    st.session_state.idea_reasoning = reasoning
    st.session_state.idea_proposal = proposal.strip()
    st.session_state.random_results = None
    st.session_state.random_selection = None
    st.session_state.idea_expanded_results = None
    st.session_state.idea_edges = None

    if expand_more and not df_info.empty:
        seed_projects = df_info["project"].tolist()

        with st.spinner("Exploring related projects by identifying shared participants, resources, or target groups..."):
            df_edges, df_expanded = expand_projects_from_seeds(
                g=g,
                seed_projects=seed_projects,
                iri_base=iri_base,
                max_extra=expand_k,
            )

        st.session_state.idea_expanded_results = df_expanded
        st.session_state.idea_edges = df_edges

# -----------------------------
# CASE 2
# Only show if idea mode is NOT active
# -----------------------------
if st.session_state.active_mode != "idea":
    st.markdown('<hr class="case-divider">', unsafe_allow_html=True)

    st.markdown('<div class="case-inner">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="case-title">🎲 Case 2: I do not have an idea yet</div>
        <div class="case-text">
          Explore one random project per project class to get inspired and discover possible directions for your own idea.
        </div>
        """,
        unsafe_allow_html=True,
    )

    random_btn = st.button(
        "Randomly pick some projects",
        use_container_width=True,
        key="random_btn",
    )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    random_btn = False

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# CASE 2 COMPUTE
# -----------------------------
if random_btn:
    if not os.path.exists(ttl_path):
        st.error(f"TTL file not found: {ttl_path}")
        st.stop()

    with st.spinner("Loading graph..."):
        g = load_graph(ttl_path)

    with st.spinner("Sampling random projects..."):
        selection, df_random = build_random_initial_recommendations_df(g=g, iri_base=iri_base)

    st.session_state.active_mode = "random"
    st.session_state.random_selection = selection
    st.session_state.random_results = df_random
    st.session_state.idea_results = None
    st.session_state.idea_reasoning = None
    st.session_state.idea_proposal = ""
    st.session_state.idea_expanded_results = None
    st.session_state.idea_edges = None

# -----------------------------
# CASE 1 DISPLAY
# -----------------------------
if st.session_state.active_mode == "idea" and st.session_state.idea_results is not None:
    df_info = st.session_state.idea_results
    reasoning = st.session_state.idea_reasoning

    st.markdown('<div class="result-section"></div>', unsafe_allow_html=True)
    st.subheader("📌 Recommended projects")

    if show_reasoning and reasoning is not None:
        with st.expander("Intermediate reasoning", expanded=False):
            st.write("Matched project type:", reasoning.get("matched_project_type"))
            st.write("Candidates retrieved:", reasoning.get("candidates_count"))
            st.write("Final projects:", reasoning.get("final_projects"))

            scores = reasoning.get("scores_table", [])
            if scores:
                df_scores = pd.DataFrame(scores)
                st.markdown("**Scores for recommended projects (after reranking):**")
                st.dataframe(df_scores, use_container_width=True, hide_index=True)

            sim_prev = reasoning.get("top_similarity_preview", [])
            if sim_prev:
                st.markdown("**Top candidates by similarity (before reranking):**")
                st.dataframe(pd.DataFrame(sim_prev), use_container_width=True, hide_index=True)

    if df_info.empty:
        st.warning("No projects found. Check your IRI base, TTL content, or project type mapping.")
    else:
        st.dataframe(
            df_info,
            column_config={"website": st.column_config.LinkColumn("Website")},
            use_container_width=True,
            hide_index=True,
        )

        df_expanded = st.session_state.idea_expanded_results
        df_edges = st.session_state.idea_edges

        if expand_more:
            st.subheader("Further recommendations")
            st.caption("Based on shared participants, resources, or target groups.")

            if df_expanded is None or df_expanded.empty:
                st.info("No additional related projects found from the seed projects.")
            else:
                st.dataframe(
                    df_expanded,
                    column_config={"website": st.column_config.LinkColumn("Website")},
                    use_container_width=True,
                    hide_index=True,
                )

                if show_reasoning and df_edges is not None:
                    with st.expander("Why these additional projects?", expanded=False):
                        st.dataframe(df_edges, use_container_width=True, hide_index=True)

        st.markdown("#### Downloads")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_bytes = df_info.to_csv(index=False).encode("utf-8")

        export_payload = {
            "timestamp": ts,
            "mode": "idea",
            "proposal": st.session_state.idea_proposal,
            "settings": {
                "ttl_path": ttl_path,
                "iri_base": iri_base,
                "LM_model": LM_model,
                "model_name": model_name,
                "top_k": top_k,
                "show_reasoning": show_reasoning,
                "expand_more": expand_more,
                "expand_k": expand_k,
            },
            "reasoning": reasoning,
            "results": df_info.to_dict(orient="records"),
            "further_recommendations": None if df_expanded is None else df_expanded.to_dict(orient="records"),
        }
        json_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="⬇️ Download recommendations (CSV)",
                data=csv_bytes,
                file_name=f"recommendations_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_idea_csv",
            )
        with c2:
            st.download_button(
                label="⬇️ Download run (JSON)",
                data=json_bytes,
                file_name=f"run_export_{ts}.json",
                mime="application/json",
                use_container_width=True,
                key="download_idea_json",
            )

# -----------------------------
# CASE 2 DISPLAY
# -----------------------------
if st.session_state.active_mode == "random" and st.session_state.random_results is not None:
    df_random = st.session_state.random_results
    selection = st.session_state.random_selection

    st.subheader("🎲 Random recommendations")
    st.caption("One random project per project class.")

    if not selection:
        st.warning("No project classes or projects found for random sampling. Check TTL content and class assertions.")
    else:
        if show_reasoning:
            with st.expander("Random selection details", expanded=False):
                st.json(selection)

        st.dataframe(
            df_random,
            column_config={"website": st.column_config.LinkColumn("Website")},
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("#### Downloads")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_bytes_random = df_random.to_csv(index=False).encode("utf-8")

        export_payload_random = {
            "timestamp": ts,
            "mode": "random",
            "settings": {
                "ttl_path": ttl_path,
                "iri_base": iri_base,
                "show_reasoning": show_reasoning,
            },
            "selection": selection,
            "results": df_random.to_dict(orient="records"),
        }
        json_bytes_random = json.dumps(
            export_payload_random,
            ensure_ascii=False,
            indent=2
        ).encode("utf-8")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                label="⬇️ Download random recommendations (CSV)",
                data=csv_bytes_random,
                file_name=f"random_recommendations_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_random_csv",
            )
        with c2:
            st.download_button(
                label="⬇️ Download random run (JSON)",
                data=json_bytes_random,
                file_name=f"random_run_export_{ts}.json",
                mime="application/json",
                use_container_width=True,
                key="download_random_json",
            )


    def check_password():
        if st.session_state.get("password_correct"):
            return True
        pw = st.text_input("Passwort", type="password")
        if pw:
            if hmac.compare_digest(pw, st.secrets["APP_PASSWORD"]):
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("Falsches Passwort")
        return False


    if not check_password():
        st.stop()