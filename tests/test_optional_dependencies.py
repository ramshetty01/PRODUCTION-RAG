def test_s3_storage_dependency_is_available():
    import boto3

    assert boto3.client
