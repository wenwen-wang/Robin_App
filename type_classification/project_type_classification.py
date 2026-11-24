from mistralai import Mistral
import os
from rdflib import Graph, Namespace, RDF
from rule_based_translator import translate_de_to_en
import json

if __name__ == "__main__":

    with open("project_type_example.json", "r") as f:
        project_type_example = json.load(f)

    # === 1. Load the local TTL file ===
    g = Graph()
    g.parse("robin.ttl", format="turtle")
    #print("✅ Knowledge graph loaded.")

    # === 2. Extract entities and text descriptions ===
    entities = []
    texts = []
    ECRM = Namespace("http://erlangen-crm.org/240307/")

    projects = []
    for s in g.subjects(RDF.type, ECRM.E7_Activity):
        project = str(s).split("/")[-1]
        projects.append(project)
    #print("✅ E7_Activity Projects extracted.", projects)

    project_descriptions = []
    for project_uri in g.subjects(RDF.type, ECRM.E7_Activity):
        project_id = str(project_uri).split("/")[-1]

        for note in g.objects(project_uri, ECRM.P3_has_note):
            project_descriptions.append(str(note))

    project_description_pair = dict(zip(projects, project_descriptions))
    #print(project_description_pair)

    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

    project_types = []
    for project, description in project_description_pair.items():
        project_description = description
        ProjectType = [
            'Community_cargo_bike',
            "Concept_development_workshop",
            "Corporate_social_engagement_platform",
            "Free_shop",
            "Idea_consulting",
            "Mobile_housing",
            "Neighbourhood_bazaar",
            "Neighbourhood_governance",
            "Open_makerspace",
            "Parklet",
            "Public_bookcase",
            "Urban_green_spaces",
            "Water_sensitive_urban_intervention"]

        system_prompt = "You are an project classifier, given a project description, please classify the project into three types it belongs to with the confidence score. " \
                        f"Possible Project_Type values: {ProjectType}. Check the example in project_type_examples in {project_type_example} and the reason why it belongs to a certain type. Sometimes, a project can belongs two types. Only retruen the project type with the higherest confidence score in string, no markdown, no explanation."

        project_description = translate_de_to_en(project_description,MISTRAL_API_KEY)
        # give explaination for why you think the project belongs to the type
        user_prompt = f"Project description is: {project_description}."

        client = Mistral(api_key=MISTRAL_API_KEY)
        response = client.chat.complete(
            model="mistral-small-latest",
            temperature=0.0,
            response_format={"type": "text"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        project_type = response.choices[0].message.content
        #print(f"{project} belongs to: ",project_type)
        print(project_type)
        #print(project)
        project_types.append(project_type)

