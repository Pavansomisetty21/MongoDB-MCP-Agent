from typing import Any, Dict
from fastmcp import FastMCP
from pymongo import MongoClient
from pydantic import create_model

mcp = FastMCP("MongoDB Server")


client = MongoClient("mongodb://localhost:27017/")
db = client['sampledb']

type_map = {
    "string": (str, ...),
    "int": (int, ...),
    "float": (float, ...),
    "bool": (bool, ...),
    "list": (list, ...),
    "dict": (dict, ...),
    "date": (str, ...),
}


@mcp.tool
def list_all_collections():
    """
    List all collections in the MongoDB database.
    """
    collections = db.list_collection_names()
    if not collections:
        return "No collections found."
    return ", ".join(collections)

@mcp.tool
def drop_collection(collection_name: str) -> str:
    """
    Drop a collection from the MongoDB database.
    """
    if collection_name not in db.list_collection_names():
        return f"Collection '{collection_name}' does not exist."
    db.drop_collection(collection_name)
    db['schema'].delete_many({"collection_name": collection_name})
    return f"Collection '{collection_name}' dropped successfully."

@mcp.tool
def create_collection_with_schema(collection_name: str, schema: dict):
    """
    Create a new collection with a specified schema in the MongoDB database.
    """

    for field_name, field_type in schema.items():
        if field_type not in type_map:
            return f"Unsupported field type '{field_type}' for field '{field_name}'. Supported types are {list(type_map.keys())}."

    if collection_name in db.list_collection_names():
        return f"Collection '{collection_name}' already exists."
    
    db.create_collection(collection_name)

    db['schema'].insert_one({
        "collection_name": collection_name,
        "schema": schema
    })
    return f"Collection {collection_name} created with schema."

@mcp.tool
def insert_document(collection_name: str, document: dict) -> str:
    """
    Insert a document into a specified collection in the MongoDB database.
    """
    if collection_name not in db.list_collection_names():
        return f"Collection '{collection_name}' does not exist."
    
    schema_doc = db['schema'].find_one({"collection_name": collection_name})
    
    if not schema_doc:
        return f"Collection '{collection_name}' does not exist or has no schema defined so create a collection with schema first."
    
    schema = schema_doc['schema']
    model_fields: Dict[str, Any] = {}

    for field_name, field_type in schema.items():
        if field_type in type_map:
            model_fields[field_name] = type_map[field_type]
        else:
            return f"Unsupported field type '{field_type}' for field '{field_name}'. Supported types are {list(type_map.keys())}."
    
    DynamicModel = create_model(f"{collection_name}Model", **model_fields)

    try:
        validated_document = DynamicModel(**document)
        db[collection_name].insert_one(validated_document.dict())
        return f"Document inserted into collection '{collection_name}'."
    except Exception as e:
        return f"Error inserting document: {str(e)}"


@mcp.tool
def view_collection(collection_name: str ):
    """
    View all documents in a specified collection in the DB.
    """
    if collection_name not in db.list_collection_names():
        return f"Collection '{collection_name}' does not exist."
    
    documents = list(db[collection_name].find())
    if not documents:
        return f"No documents found in collection '{collection_name}'."
    
    return [doc for doc in documents]

@mcp.tool
def get_collection_schema(collection_name: str):
    """
    Get the schema of a specified collection in the DB.
    """
    schema_doc = db['schema'].find_one({"collection_name": collection_name})
    if not schema_doc:
        return f"Collection '{collection_name}' does not exist or has no schema defined."
    
    return schema_doc['schema']

@mcp.tool
def update_document(collection_name: str, document_id: str, update_fields: dict) -> str:
    """
    Update a document in a specified collection in the DB.
    """
    if collection_name not in db.list_collection_names():
        return f"Collection '{collection_name}' does not exist."
    
    result = db[collection_name].update_one({"_id": document_id}, {"$set": update_fields})
    
    if result.matched_count == 0:
        return f"No document found with ID '{document_id}' in collection '{collection_name}'."
    
    return f"Document with ID '{document_id}' updated successfully in collection '{collection_name}'."

@mcp.tool
def delete_document(collection_name: str, document_id: str) -> str:
    """
    Delete a document from a specified collection in the DB.
    """
    if collection_name not in db.list_collection_names():
        return f"Collection '{collection_name}' does not exist."
    
    result = db[collection_name].delete_one({"_id": document_id})
    
    if result.deleted_count == 0:
        return f"No document found with ID '{document_id}' in collection '{collection_name}'."
    
    return f"Document with ID '{document_id}' deleted successfully from collection '{collection_name}'."



if __name__ == '__main__':
    mcp.run()
