from django.db import models

class Sleep(models.Model):
    id = models.UUIDField(primary_key=True)
    end_at = models.DateTimeField(null=True)
    start_at = models.DateTimeField(null=True)
    user_id = models.UUIDField(db_index=True)

class User(models.Model):
    id = models.UUIDField(primary_key=True)
    bmi = models.FloatField(null=True)
    
class Exercise(models.Model):
    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    completed_at = models.DateTimeField(null=True)
    target = models.CharField(max_length=255)
    user_id = models.UUIDField(db_index=True)

class Protein(models.Model):
    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    grams = models.FloatField()
    completed_at = models.DateTimeField(null=True)
    user_id = models.UUIDField(db_index=True)
