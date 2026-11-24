import os
from mistralai import Mistral
import rdflib
import json
from pathlib import Path

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def extract_user_input_city(project_proposal):

    system_prompt_city = "You are an geographical detector, give a project proposal, please identify the city where the project is located. " \
                         "The valid values for cities are [Nuremberg, Fürth, unknown]" \
                         "If no explicit location associated with a city is provided, the city should be labeled as “unknown” rather than inferred." \
                         "No markdown format" \
                         "" \
                         "return only the city name without explanation" \


    user_prompt_city = f"Project description is: {project_proposal}."

    client = Mistral(api_key=MISTRAL_API_KEY)
    response_city = client.chat.complete(
        model="mistral-small-latest",
        temperature=0.0,
        response_format={"type": "text"},
        messages=[
            {"role": "system", "content": system_prompt_city},
            {"role": "user", "content": user_prompt_city},
        ],
    )

    city = response_city.choices[0].message.content
    #print("user city is: ",city)
    return city


def extract_projects_city(projects):
    if len(projects) == 0:
        return ["unknown"]
    cities = []
    for proj in projects:

        city_query = f"""
            PREFIX : <http://www.semanticweb.org/pyt/ontologies/2025/10/robin#>
            PREFIX ecrm: <http://erlangen-crm.org/240307/> 
            PREFIX owl: <http://www.w3.org/2002/07/owl#> 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> 
            PREFIX xml: <http://www.w3.org/XML/1998/namespace> 
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> 
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> 

            SELECT ?city
            WHERE {{
            ecrm:{proj} a ecrm:E7_Activity ;
                    ecrm:P7_took_place_at ?city .
            }}
            """
        g = rdflib.Graph()
        g.parse("robin.ttl", format="turtle")
        result = g.query(city_query)
        for row in result:
            city = row[0].split("/")[-1]
            cities.append(city)
    #print("cities are ",cities)
    return cities

def normalize_city(city):
    if city == "unknown":
        return "unknown"
    #print("city is ",city)
    return city.strip().lower()

def match_cities(project_proposal, projects):

    user_city = extract_user_input_city(project_proposal)
    if user_city == "unknown":
        print("It seems that you don’t care about the cities where the projects are located.")
    else:
        print(f"Your idea shows that you are interested in the projects located in the city {user_city}.")
    user_city_norm = normalize_city(user_city)
    project_cities = extract_projects_city(projects)
    project_cities_norm = [normalize_city(c) for c in project_cities]

    return [int(user_city_norm == c) for c in project_cities_norm]

def compute_city_weight(project_proposal, projects):
    match_scores = match_cities(project_proposal, projects)
    weights = [1.2 if item == 1 else 1 for item in match_scores]
    return weights



