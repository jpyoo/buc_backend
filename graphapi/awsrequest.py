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
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

MAX_GET_SIZE = 100  # Amazon DynamoDB rejects a get batch larger than 100 items.
AWS_ACCT = 'zirl2v2iw5h7tnbs3wznskuvvm-bulkupcoac'

class DataAPI:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()

        # Define AWS DynamoDB client and tables
        self.dynamodb_client = boto3.client('dynamodb', region_name="us-west-1")
        self.USER_TABLE_NAME = 'User-' + AWS_ACCT
        self.EXERCISE_TABLE_NAME = "Exercise-" + AWS_ACCT
        self.PROTEIN_TABLE_NAME = "Protein-" + AWS_ACCT
        self.SLEEP_TABLE_NAME = "Sleep-" + AWS_ACCT

        # Define PostgreSQL connection parameters
        self.conn = psycopg2.connect(
            dbname="bulkupcoach",
            user="bulkupcoach",
            password="password",
            host="localhost",
            port="5432"
        )

    def setup_routes(self):
        self.app.add_url_rule('/get_items_by_user', 'get_items_by_user', self.get_items_by_user, methods=['GET'])
        self.app.add_url_rule('/print_all_items', 'print_all_items', self.print_all_items, methods=['GET'])
        self.app.add_url_rule('/import_user_from_aws', 'import_user_from_aws', self.import_user_from_aws, methods=['POST'])
        # Define other routes for importing exercises, proteins, and sleep data from AWS DynamoDB to PostgreSQL

    # Example:
    # requests.get(f'{base_url}/get_items_by_user?user_id=user1&table_name=User')
    def get_items_by_user(self):
        user_id = request.args.get('user_id')
        table_name = request.args.get('table_name') + '-' + AWS_ACCT
        try:
            response = self.dynamodb_client.query(
                TableName=table_name,
                KeyConditionExpression="userID = :uid",
                ExpressionAttributeValues={
                    ":uid": {"S": user_id}
                }
            )
            items = response['Items']
            return jsonify(items)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def get_weekly_items_by_user(self):
        user_id = request.args.get('user_id')
        table_name = request.args.get('table_name') + '-' + AWS_ACCT
        try:
            # Calculate the start and end dates of the current week
            today = datetime.today()
            start_of_week = today - timedelta(days=today.weekday())
            end_of_week = start_of_week + timedelta(days=6)

            # Convert start and end dates to DynamoDB compatible format
            start_of_week_str = start_of_week.strftime('%Y-%m-%d')
            end_of_week_str = end_of_week.strftime('%Y-%m-%d')

            response = self.dynamodb_client.query(
                TableName=table_name,
                KeyConditionExpression="userID = :uid AND #date BETWEEN :start_date AND :end_date",
                ExpressionAttributeNames={
                    "#date": "completedAt"
                },
                ExpressionAttributeValues={
                    ":uid": {"S": user_id},
                    ":start_date": {"S": start_of_week_str},
                    ":end_date": {"S": end_of_week_str}
                }
            )
            items = response['Items']
            return jsonify(items)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def print_all_items(self):
        table_name = request.args.get('table_name') + '-' + AWS_ACCT
        try:
            # Query all items from DynamoDB table
            response = self.dynamodb_client.scan(
                TableName=table_name
            )
            items = response['Items']
            return jsonify(items)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def import_user_from_aws(self):
        try:
            cur = self.conn.cursor()
            response = self.dynamodb_client.scan(
                TableName=self.USER_TABLE_NAME
            )
            items = response['Items']
            for item in items:
                user_id = item.get('id', {}).get('S', '')
                bmi = item.get('BMI', {}).get('N', '')
                
                # Check if the user already exists in the PostgreSQL database
                cur.execute("SELECT COUNT(*) FROM graphapi_user WHERE id = %s", (user_id,))
                user_count = cur.fetchone()[0]
                
                if user_count == 0:  # If user doesn't exist, insert it into the database
                    cur.execute(
                        "INSERT INTO graphapi_user (id, bmi) VALUES (%s, %s)",
                        (user_id, float(bmi))
                    )
                    self.conn.commit()
            return jsonify({"message": "Data imported successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Define other methods for importing exercises, proteins, and sleep data from AWS DynamoDB to PostgreSQL

    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    data_api = DataAPI()
    data_api.run()