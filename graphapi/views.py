from django.shortcuts import render

from django.http import JsonResponse
from django.views.decorators.http import require_GET, require_POST
from .awsrequest import DataAPI
from django.views.decorators.csrf import csrf_exempt

data_api = DataAPI()

@csrf_exempt
@require_GET
def get_items_by_user(request):
    if request.method == 'GET':
        user_id = request.GET.get('user_id')
        table_name = request.GET.get('table_name', '')
        items = data_api.get_items_by_user(user_id, table_name)
        return JsonResponse(items)

@csrf_exempt
@require_GET
def get_weekly_items_by_user(request):
    if request.method == 'GET':
        user_id = request.GET.get('user_id')
        table_name = request.GET.get('table_name', '')
        items = data_api.get_weekly_items_by_user(user_id, table_name)
        return JsonResponse(items)

@csrf_exempt
@require_GET
def print_all_items(request):
    if request.method == 'GET':
        table_name = request.GET.get('table_name')
        items = data_api.print_all_items(table_name)
        return JsonResponse(items)

@csrf_exempt
@require_POST
def import_user_from_aws(request):
    if request.method == 'POST':
        result = data_api.import_user_from_aws()
        return JsonResponse(result)
    
# Define other API endpoints for importing exercises, proteins, and sleep data from AWS DynamoDB to PostgreSQL
# Example:
# @csrf_exempt
# @require_POST
# def import_exercises_from_aws(request):
#     if request.method == 'POST':
#         result = data_api.import_exercises_from_aws()
#         return JsonResponse(result)
#
# @csrf_exempt
# @require_POST
# def import_proteins_from_aws(request):
#     if request.method == 'POST':
#         result = data_api.import_proteins_from_aws()
#         return JsonResponse(result)
#