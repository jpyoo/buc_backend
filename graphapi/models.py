from django.db import models

class Sleep(models.Model):
    id = models.CharField(primary_key=True)
    end_at = models.DateTimeField(null=True)
    start_at = models.DateTimeField(null=True)
    user_id = models.CharField(db_index=True)

class User(models.Model):
    id = models.CharField(primary_key=True)
    bmi = models.FloatField(null=True)

class Exercise(models.Model):
    id = models.CharField(primary_key=True)
    name = models.CharField(max_length=255)
    weight_lb = models.IntegerField(null=True)
    reps = models.IntegerField(null=True)
    completed_at = models.DateTimeField(null=True)
    target = models.CharField(max_length=255)
    user_id = models.CharField(db_index=True)

class Protein(models.Model):
    id = models.CharField(primary_key=True)
    name = models.CharField(max_length=255)
    grams = models.FloatField()
    completed_at = models.DateTimeField(null=True)
    user_id = models.CharField(db_index=True)
