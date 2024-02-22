import decimal
import json
import logging
import os
import pprint
import time
import boto3
from botocore.exceptions import ClientError
from boto3.dynamodb.conditions import Key, Attr
import psycopg2

logger = logging.getLogger(__name__)

MAX_GET_SIZE = 100  # Amazon DynamoDB rejects a get batch larger than 100 items.

aws_acct = 'zirl2v2iw5h7tnbs3wznskuvvm-bulkupcoac'
USER_TABLE_NAME = 'User-' + aws_acct
EXERCISE_TABLE_NAME = "Exercise-" + aws_acct
PROTEIN_TABLE_NAME = "Protein-" + aws_acct
SLEEP_TABLE_NAME = "Sleep-" + aws_acct

# Creating the DynamoDB Client
dynamodb_client = boto3.client('dynamodb', region_name="us-west-1")

# Creating the DynamoDB Table Resource
dynamodb = boto3.resource('dynamodb', region_name="us-west-1")

user_table = dynamodb.Table(USER_TABLE_NAME)
exercise_table = dynamodb.Table(EXERCISE_TABLE_NAME)
protein_table = dynamodb.Table(PROTEIN_TABLE_NAME)
sleep_table = dynamodb.Table(SLEEP_TABLE_NAME)


# # Filtering items in a table by user ID
# response = exercise_table.scan(
#     FilterExpression=Attr('userID').eq('user5')
# )
# items = response['Items']
# print(items)

# Get all items in a table
response = user_table.scan()
items = response['Items']
print(items)

attribute_names = set()  # Using a set to ensure uniqueness

for item in items:
    # Extract keys (attribute names) from the item
    for key in item.keys():
        attribute_names.add(key)

# Print the attribute names
print("Attribute Names:")
for attribute_name in attribute_names:
    print(attribute_name)


# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="bulkupcoach",
    user="bulkupcoach",
    password="password",
    host="localhost",
    port = '5432',
)

# Create a cursor
cur = conn.cursor()


# Fetch data from DynamoDB and insert into PostgreSQL tables
for table_name in [USER_TABLE_NAME]:
    response = dynamodb_client.scan(TableName=table_name)
    items = response['Items']
    for item in items:
        # Insert item into PostgreSQL table
        # Assuming the structure of DynamoDB items and PostgreSQL tables are compatible
        cur.execute(
            f"INSERT INTO graphapi_user (id, bmi) VALUES (%s, %s, ...)",
            (item['id'], item['BMI'])
        )