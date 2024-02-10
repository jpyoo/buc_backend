import decimal
import json
import logging
import os
import pprint
import time
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key, Attr

logger = logging.getLogger(__name__)
dynamodb = boto3.resource("dynamodb")

MAX_GET_SIZE = 100  # Amazon DynamoDB rejects a get batch larger than 100 items.

dynamodb = boto3.client("dynamodb")
TABLE_NAME = "User-zirl2v2iw5h7tnbs3wznskuvvm-bulkupcoac"
EXERCISE_TABLE_NAME = "Exercise-zirl2v2iw5h7tnbs3wznskuvvm-bulkupcoac"
PROTEIN_TABLE_NAME = "Protein-zirl2v2iw5h7tnbs3wznskuvvm-bulkupcoac"

# Creating the DynamoDB Client
dynamodb_client = boto3.client('dynamodb', region_name="us-west-1")

# Creating the DynamoDB Table Resource
dynamodb = boto3.resource('dynamodb', region_name="us-west-1")
table = dynamodb.Table(TABLE_NAME)
exercise_table = dynamodb.Table(EXERCISE_TABLE_NAME)
protein_table = dynamodb.Table(PROTEIN_TABLE_NAME)

response = exercise_table.scan(
    FilterExpression=Attr('userID').eq('user5')
)
items = response['Items']
print(items)

# need to know if we can retrieve item from table with a non-primary key
# response = dynamodb_client.get_item(
#     TableName=TABLE_NAME,
#     Key={
#         'id': {'S': 'd138d559-b08b-4495-878b-32059d1cab99'}
#     }
# )


# print(response['Item'])