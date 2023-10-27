import os 
from google.cloud import storage 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'app/secret/cloud_storage_key.json'
storage_client = storage.Client() # create client instance

""" 
dir(storage_client)
[
    'SCOPE', '_SET_PROJECT', '__annotations__', '__class__', 
    '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', 
    '__format__', '__ge__', '__getattribute__', '__getstate__', 
    '__gt__', '__hash__', '__init__', '__init_subclass__', 
    '__le__', '__lt__', '__module__', '__ne__', '__new__', 
    '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', 
    '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 
    '_base_connection', '_batch_stack', '_bucket_arg_to_bucket', 
    '_client_cert_source', '_connection', '_credentials', 
    '_delete_resource', '_determine_default', '_get_resource', 
    '_http', '_http_internal', '_list_resource', '_patch_resource', 
    '_pop_batch', '_post_resource', '_push_batch', '_put_resource', 
    'batch', 'bucket', 'close', 'create_anonymous_client', 
    'create_bucket', 'create_hmac_key', 'current_batch', 
    'download_blob_to_file', 'from_service_account_info', 
    'from_service_account_json', 'generate_signed_post_policy_v4', 
    'get_bucket', 'get_hmac_key_metadata', 'get_service_account_email', 
    'list_blobs', 'list_buckets', 'list_hmac_keys', 'lookup_bucket', 
    'project'
]

"""



# create a new bucket
bucket_name = 'model_results_ss' # bucket name cannot contain spaces
bucket = storage_client.bucket(bucket_name=bucket_name)
bucket.location = 'US'
bucket.storage_class = 'COLDLINE' # Archive | Nearline | Standard
bucket = storage_client.create_bucket(bucket) # returns Bucket object
print(vars(bucket))

bucket.name
bucket._properties['selfLink']
bucket._properties['id']
bucket._properties['location']
bucket._properties['timeCreated']
bucket._properties['storageClass']
bucket._properties['timeCreated']
bucket._properties['updated']

# access bucket
my_bucket = storage_client.get_bucket(bucket_name)
print(vars(my_bucket))


# upload file to bucket
def upload_to_bucket(blob_name, file_path, bucket_name):
    '''
    Upload file to a bucket
    : blob_name  (str) - object name
    : file_path (str)
    : bucket_name (str)
    '''
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    return blob

# response = upload_to_bucket('blob_name', 'path.csv', bucket_name)
# response = upload_to_bucket('/docs/requirementABC', 'requirements.txt', bucket_name)



# download file from bucket blob name
def download_file_from_bucket(blog_name, file_path, bucket_name):
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blog_name)
    with open(file_path, 'wb') as f:
        storage_client.download_blob_to_file(blob, f)
    print('Saved')

# download_file_from_bucket('blob_name', r'download_location.csv', bucket_name)



# download file by uri of bucket
def download_file_uri(uri, file_path):
    with open(file_path, 'wb') as f:
        storage_client.download_blob_to_file(uri, f)
    print('Saved')

uri = 'gs://<uri>'
# download_file_uri(uri, r'download_location.csv')



# list Buckets
#list_buckets(max_results=None, page_token=None, prefix=None, projection='noAcl', fields=None, project=None, timeout=60, retry=<google.api_core.retry.Retry object>)

for bucket in storage_client.list_buckets(max_results=100):
    print(bucket)
