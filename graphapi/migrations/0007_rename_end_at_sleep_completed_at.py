# Generated by Django 5.0.2 on 2024-03-08 09:02

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('graphapi', '0006_alter_exercise_id_alter_exercise_user_id_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='sleep',
            old_name='end_at',
            new_name='completed_at',
        ),
    ]
