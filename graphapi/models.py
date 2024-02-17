from django.db import models

class Sleep(models.Model):
    id = models.UUIDField(primary_key=True)
    endAt = models.DateTimeField(null=True)
    startAt = models.DateTimeField(null=True)
    userID = models.UUIDField(db_index=True)

class User(models.Model):
    id = models.UUIDField(primary_key=True)
    BMI = models.FloatField(null=True)

class Exercise(models.Model):
    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    weight_lb = models.IntegerField(null=True)
    reps = models.IntegerField(null=True)
    completedAt = models.DateTimeField(null=True)
    target = models.CharField(max_length=255)
    userID = models.UUIDField(db_index=True)

class Protein(models.Model):
    id = models.UUIDField(primary_key=True)
    name = models.CharField(max_length=255)
    grams = models.FloatField()
    completedAt = models.DateTimeField(null=True)
    userID = models.UUIDField(db_index=True)
