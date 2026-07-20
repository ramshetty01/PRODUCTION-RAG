def test_s3_storage_dependency_is_available():
    import boto3

    assert boto3.client


def test_qdrant_vector_dependency_is_available():
    from langchain_qdrant import QdrantVectorStore

    assert QdrantVectorStore
