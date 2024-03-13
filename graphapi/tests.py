from django.test import TestCase

# Create your tests here.
from awsrequest import DataAPI
from flask import Flask, request, jsonify
from django.http import JsonResponse

# Your test code
app = Flask(__name__)

def test_functionality():
    with app.app_context():
        # Instantiate the DataAPI class
        data_api = DataAPI()

        # Sample user_id and table_name
        user_id = "user2"
        table_name = "User"

        # Call the get_items_by_user method
        items = data_api.get_items_by_user(user_id, table_name)
        # json_data = JsonResponse(items, safe=False)
        # # Print the returned items
        print("Items Retrieved from data API:", items)
        p_user_id = "user4"
        p_table_name = "Protein"
        p_items = data_api.get_items_by_user(p_user_id, p_table_name)
        print("Protein Items Retrieved from data API:", p_items)

        # all_users = data_api.print_all_items(table_name)
        # print("All users:", all_users)

# Run the test code
if __name__ == "__main__":
    test_functionality()