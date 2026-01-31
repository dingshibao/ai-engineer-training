from llama_index.graph_stores.neo4j import Neo4jGraphStore

class Neo4jManager:
    def __init__(self, uri: str, username: str, password: str, database: str):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database

    
    def get_graph_store(self):
        return Neo4jGraphStore(
            username=self.username,
            password=self.password,
            url=self.uri,
            database=self.database,
        )
