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

projects = []
for s in g.subjects(RDF.type, ECRM.E7_Activity):
    project = str(s).split("/")[-1]
    projects.append(project)
print("✅ E7_Activity Projects extracted.",projects)

project_descriptions  = []
for project_uri in g.subjects(RDF.type, ECRM.E7_Activity):
    project_id = str(project_uri).split("/")[-1]

    for note in g.objects(project_uri, ECRM.P3_has_note):
        project_descriptions.append(str(note))

output = []

for project, description in zip(projects, project_descriptions):
    output.append({
        project: description
    })

with open("projects_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("💾 Saved all extracted entities to robin_entities.json")


