from rdflib import Graph, Namespace, RDF
import random
from sentence_transformers import SentenceTransformer, util
from app.city_matching.city_matching import compute_city_weight
from app.type_classification.project_type_matching import match_project_type


g = Graph()
g.parse("robin.ttl", format="turtle")
PREFIXES = """
PREFIX : <http://www.semanticweb.org/pyt/ontologies/2025/10/robin#>
PREFIX ecrm: <http://erlangen-crm.org/240307/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xml: <http://www.w3.org/XML/1998/namespace>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

IRI = "http://erlangen-crm.org/240307/"
ECRM = Namespace("http://erlangen-crm.org/240307/")

project_classes = []
for s in g.subjects(RDF.type, ECRM.E303_Project_Class):
    project = str(s).split("/")[-1]
    project_classes.append(project)
#print("✅ E7_Activity Projects extracted.",project_classes)

def retrieve_projects_from_one_class(project_class):
    query = f"""
        {PREFIXES}
        SELECT ?project
        WHERE {{
        ?project a ecrm:E7_Activity ;
                ecrm:P503_has_project_class ecrm:{project_class} ;
        }}
        """
    query_result = g.query(query)
    if len(query_result) == 0:
        print("No similar projects found")

    projects = []
    if len(query_result) != 0:
        # print("The similar projects are: ")
        # print("length of the query result is: ", len(query_result))
        for row in query_result:
            project = row[0].split("/")[-1]
            projects.append(project)
    print("the projects are: ",projects)
    return projects


def retrieve_all_projects(project_classes):
    class_len = len(project_classes)
    all_projects = {}
    for i in range(class_len):
        all_projects[project_classes[i]] = retrieve_projects_from_one_class(project_classes[i])
    return all_projects

all_projects = retrieve_all_projects(project_classes)
random_selection = {key: random.choice(value) for key, value in all_projects.items()}
print(random_selection)



def retrieve_projects_with_the_same_project_type(project_type):

    query = f"""
    {PREFIXES}
    SELECT ?project ?description
    WHERE {{
    ?project a ecrm:E7_Activity ;
            ecrm:P502_has_project_type ecrm:{project_type} ;
            ecrm:P3_has_note ?description .
    }}
    """
    query_result = g.query(query)
    if len(query_result) == 0:
        print("No similar projects found")

    projects = []
    descriptions = []
    if len(query_result) != 0:
        #print("The similar projects are: ")
        #print("length of the query result is: ", len(query_result))
        for row in query_result:
            project = row[0].split("/")[-1]
            projects.append(project)
            description = row[1].split("/")[-1]
            descriptions.append(description)
    return projects, descriptions




def retrieve_full_information_of_projects(projects):
    full_projects_information = []

    for proj in projects:
        project_iri = IRI + proj
        #print ("the projects ",project_iri)
        query = f"""
        {PREFIXES}
        SELECT ?project ?description ?project_type ?city ?website ?operator ?sponsor
        WHERE {{
        
        BIND(<{project_iri}> AS ?project)
        
        ?project a ecrm:E7_Activity ;
                   ecrm:P3_has_note ?description ;
                   ecrm:P502_has_project_type ?project_type ;
        OPTIONAL {{ ?project ecrm:P502_has_project_type ?project_type . }}
        OPTIONAL {{ ?project ecrm:P7_took_place_at ?city . }}
        OPTIONAL {{ ?project ecrm:P90_has_value ?website . }}
        
        }}
        """

        query_result = list(g.query(query))
        #print("the query results are: ", query_result)

        if len(query_result) == 0:
            #print(f"No information found for: {proj}")
            full_projects_information.append([])
            continue

        # usually 1 row, but keep generic
        for row in query_result:
            full_projects_information.append([
                str(row.project.split("/")[-1]),
                str(row.description.split("/")[-1]) if row.description else "",
                str(row.project_type.split("/")[-1]) if row.project_type else "",
                str(row.city.split("/")[-1]) if row.city else "",
                str(row.website.split("/")[-1]) if row.website else "",
            ])
    return full_projects_information



# compute the similarity between the project proposal and the projects descriptions, return a list of their similarity scores
def compute_similarity(project_propasal, projects_descriptions_pairs):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # Define the string to match
    descriptions = list(projects_descriptions_pairs.values())

    #print("the length of the descriptions is: ", len(descriptions))
    similarity_scores = []

    proposal_embedding = model.encode(project_propasal, convert_to_tensor=True)
    for desp in descriptions:
        # Compute embedding for the string
        project_embedding = model.encode(desp, convert_to_tensor=True)

        # Compute cosine-similarities using GPU
        cos_score = util.cos_sim(proposal_embedding, project_embedding)[0][0].cpu().item()
        similarity_scores.append(cos_score)
    #print("the similarity score is: ",similarity_scores)
    return similarity_scores


if __name__ == "__main__":

    project_proposal = "Die Initiierung eines öffentlichen Bücherschranks im Stadtteil soll die Lesekultur fördern und den Zugang " \
                            "zu Literatur für alle Bewohner erleichtern. Der Bücherschrank wird als zentraler Treffpunkt dienen, an dem " \
                            "Bücher kostenlos getauscht und entliehen werden können. Dieses Projekt stärkt nicht nur den sozialen Zusammenhalt " \
                            "im Stadtteil, sondern fördert auch den nachhaltigen Umgang mit Ressourcen, indem Bücher weitergegeben statt entsorgt werden. " \
                            "Von diesem Projekt profitieren insbesondere Familien, Kinder und Jugendliche, die durch den einfachen Zugang zu Büchern " \
                            "ihre Lesefähigkeiten verbessern können. Lokale Schulen und Bildungseinrichtungen können den Bücherschrank in ihre Programme integrieren, " \
                            "um das Interesse am Lesen zu fördern. Der Bücherschrank bietet zudem eine Plattform für lokale Autoren, ihre Werke einer breiteren Öffentlichkeit " \
                            "zugänglich zu machen. Für die Stadt entsteht ein Mehrwert durch die Förderung der Bildung und die Stärkung des Gemeinschaftsgefühls im Stadtteil. " \
                            "Der Bücherschrank kann als Vorzeigeprojekt für weitere Initiativen zur Förderung der Lesekultur dienen. " \
                            "Die geschätzten Kosten für das Projekt belaufen sich auf ca. 2.000 Euro. Ich bin eine Privatperson."


    intermediate_reasoning_step = []
    # step 1: prefiltering, extract the projects and their descriptions with the same project type
    project_type = match_project_type(project_proposal)
    intermediate_reasoning_step.append(project_type)
    #print(f"The idea from you are similar to {project_type}.")
    projects, descriptions = retrieve_projects_with_the_same_project_type(project_type)
    projects_descriptions_pairs = dict(zip(projects, descriptions))
    #print(f"I have find {len(projects_descriptions_pairs)} similar project with your idea.")

    # step 2: compute the similarity between the project proposal and the projects descriptions and only keep the top 5
    top_k = 5
    if top_k > len(projects_descriptions_pairs):
        top_k = len(projects_descriptions_pairs)

    similarity_scores = compute_similarity(project_proposal, projects_descriptions_pairs)
    sorted_projects = sorted(zip(similarity_scores,projects),reverse=True)
    top_k_projects = sorted_projects[:top_k]
    projects_to_recommend = [p for s, p in top_k_projects]


    # step 3: assign city weights to the final projects, if the city matches the project proposal, assign a higher weight 1.2, otherwise 1.0
    cities_weights = compute_city_weight(project_proposal,projects_to_recommend)
    # apply city weights to scores
    weighted_top_k = [(score * city_weight, project) for (score, project), city_weight in zip(top_k_projects, cities_weights)]

    # step 4: resort the projects considering weighted city scores
    weighted_top_k = sorted(weighted_top_k, reverse=True, key=lambda x: x[0])
    final_projects = []

    #print(f"According to your preference, the {top_k} similar projects are re-ranked and listed below.")
    for city_weight in weighted_top_k:
        city = city_weight[1]
        final_projects.append(city)
    #print("The final projects to recommend are: ",final_projects)

    full_information_of_projects = retrieve_full_information_of_projects(final_projects)
    for proj in full_information_of_projects:
        print("the full informations are: ",proj)








