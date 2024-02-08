from django.db import models

# Create your models here.
class Priority(models.TextChoices):
    LOW = 'LOW'
    NORMAL = 'NORMAL'
    HIGH = 'HIGH'

class Todo(models.Model):
    name = models.CharField(max_length=255)
    priority = models.CharField(max_length=10, choices=Priority.choices)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return self.name