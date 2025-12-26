docker pull mongodb-atlas-local
docker run --name text2kg_mongo -p 27018:27018 mongodb-atlas-local:latest
python3 create_wikidata_ontology_db.py
python3 create_ontological_triplets_db.py

# to not use the ontology from wikidata, run:
# python3 create_triplets_db.py
