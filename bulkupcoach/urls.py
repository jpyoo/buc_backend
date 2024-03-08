"""
URL configuration for bulkupcoach project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from graphapi import views as graphapi_views

urlpatterns = [
    path('admin/', admin.site.urls),

    path('get_items_by_user/', graphapi_views.get_items_by_user, name='get_items_by_user'),
    path('print_all_items/', graphapi_views.print_all_items, name='print_all_items'),
    path('import_user_from_aws/', graphapi_views.import_user_from_aws, name='import_user_from_aws'),
    path('get_weekly_items_by_user/', graphapi_views.get_weekly_items_by_user, name='get_weekly_items_by_user'),
    # Define other URL patterns for additional API endpoints
]


