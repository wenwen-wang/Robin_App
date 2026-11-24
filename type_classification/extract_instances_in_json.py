from rdflib import Graph
from rdflib.namespace import RDFS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rdflib import Graph, Namespace, RDF
import json

# === 1. Load the local TTL file ===
g = Graph()
g.parse("robin.ttl", format="turtle")
print("✅ Knowledge graph loaded.")

# === 2. Extract entities and text descriptions ===
entities = []
texts = []
ECRM = Namespace("http://erlangen-crm.org/240307/")

Projects = []
for s in g.subjects(RDF.type, ECRM.E7_Activity):
    project = str(s).split("/")[-1]
    Projects.append(project)
print("✅ E7_Activity Projects extracted.",Projects)


Places = []
for s in g.subjects(RDF.type, ECRM.E53_Place):
    place = str(s).split("/")[-1]  # get the local name (after last '/')
    Places.append(place)

print("✅ E53_Place:", Places)
print(f"Total: {len(Places)} instances found.")

Groups = []
for s in g.subjects(RDF.type, ECRM.E74_Group):
    group = str(s).split("/")[-1]
    Groups.append(group)
print("✅ E74_Group:", Groups)
print(f"Total: {len(Groups)} instances found.")

Persons = []
for s in g.subjects(RDF.type, ECRM.E21_Person):
    person = str(s).split("/")[-1]
    Persons.append(person)
print("✅ E21_Person:", Persons)
print(f"Total: {len(Persons)} instances found.")

Roles = []
for s in g.subjects(RDF.type, ECRM.E300_Role):
    role = str(s).split("/")[-1]
    Roles.append(role)
print("✅ E300_Role:", Roles)
print(f"Total: {len(Roles)} instances found.")

Stages = []
for s in g.subjects(RDF.type, ECRM.E301_Stage_of_Project):
    stage = str(s).split("/")[-1]
    Stages.append(stage)
print("✅ E301_Stage_of_Project:", Stages)
print(f"Total: {len(Stages)} instances found.")

Project_types = []
for s in g.subjects(RDF.type, ECRM.E302_Project_Type):
    #if (s, RDF.type, ECRM.E300_Role) in g or (s, RDF.type, ECRM.E301_Stage_of_Project) in g:
        #continue
    project_type = str(s).split("/")[-1]
    Project_types.append(project_type)
print("✅ E302 project_Type:", Project_types)
print(f"Total: {len(Project_types)} instances found.")

Human_made_objects = []
for s in g.subjects(RDF.type, ECRM["E22_Human-Made_Object"]):
    object = str(s).split("/")[-1]
    Human_made_objects.append(object)
print("✅ E22_Human-Made_Object:", Human_made_objects)
print(f"Total: {len(Human_made_objects)} instances found.")



Usages = []
for s in g.subjects(RDF.type, ECRM.E29_Design_or_Procedure):
    usage = str(s).split("/")[-1]
    Usages.append(usage)
print("✅ E29_Design_or_Procedure:", Usages)
print(f"Total: {len(Usages)} instances found.")

output = {
    "E7_Activity": Projects,
    "E53_Place": Places,
    "E74_Group": Groups,
    "E21_Person": Persons,
    "E300_Role": Roles,
    "E301_Stage_of_Project": Stages,
    "E302_ProjectType": Project_types,
    "E22_Human-Made_Object": Human_made_objects,
    "E29_Design_or_Procedure": Usages
}

with open("robin_entities.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("💾 Saved all extracted entities to robin_entities.json")
