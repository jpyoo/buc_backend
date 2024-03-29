# Generated by Django 5.0.2 on 2024-02-17 06:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('graphapi', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Exercise',
            fields=[
                ('id', models.UUIDField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('completed_at', models.DateTimeField(null=True)),
                ('target', models.CharField(max_length=255)),
                ('user_id', models.UUIDField(db_index=True)),
            ],
        ),
        migrations.CreateModel(
            name='Protein',
            fields=[
                ('id', models.UUIDField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=255)),
                ('grams', models.FloatField()),
                ('completed_at', models.DateTimeField(null=True)),
                ('user_id', models.UUIDField(db_index=True)),
            ],
        ),
        migrations.CreateModel(
            name='Sleep',
            fields=[
                ('id', models.UUIDField(primary_key=True, serialize=False)),
                ('end_at', models.DateTimeField(null=True)),
                ('start_at', models.DateTimeField(null=True)),
                ('user_id', models.UUIDField(db_index=True)),
            ],
        ),
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.UUIDField(primary_key=True, serialize=False)),
                ('bmi', models.FloatField(null=True)),
            ],
        ),
        migrations.DeleteModel(
            name='Todo',
        ),
    ]
