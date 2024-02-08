import requests
from requests_aws4auth import AWS4Auth

# Amplify.configure({
#   API: {
#     GraphQL: {
#       endpoint: 'https://ljrju4chcbeftbp6fjp7j4ke4q.appsync-api.us-west-1.amazonaws.com/graphql',
#       region: 'us-west-1',
#       defaultAuthMode: 'apiKey',
#       apiKey: 'da2-n2bao*********************'
#     }
#   }
# });

# Use AWS4Auth to sign a requests session
session = requests.Session()
session.auth = AWS4Auth(
    # An AWS 'ACCESS KEY' associated with an IAM user.
    'AKxxxxxxxxxxxxxxx2A',
    # The 'secret' that goes with the above access key.                    
    'kwWxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxgEm',    
    # The region you want to access.
    'us-west-1',
    # The service you want to access.
    'appsync'
)
# As found in AWS Appsync under Settings for your endpoint.
APPSYNC_API_ENDPOINT_URL = 'https://ljrju4chcbeftbp6fjp7j4ke4q.appsync-api.us-west-1.amazonaws.com/graphql'
# Use JSON format string for the query. It does not need reformatting.
query = """
    query foo {
        GetUserSettings (
           identity_id: "ap-southeast-2:8xxxxxxb-7xx4-4xx4-8xx0-exxxxxxx2"
        ){ 
           user_name, email, whatever 
}}"""
# Now we can simply post the request...
response = session.request(
    url=APPSYNC_API_ENDPOINT_URL,
    method='POST',
    json={'query': query}
)
print(response.text)