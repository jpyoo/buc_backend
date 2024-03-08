# Django backend application for the bulk up coach project

# Bulk Up Coach Backend Configuration
This Jupyter notebook will demonstrate the backend configuration of Bulk Up Coach's backend server and database configurations and their rationals. Each steps are sorted in the order of progress.
Configuring, Defining, Creating, and Updating features of the backend can be found in this document.

## Step 1: Configure the environment
To configure the environment, we need to create a new virtual environment that will function as our backend server.
Once the backend server and its database is constructed, we need to test connecting with our application's Amplify cloud server.
This step will connect the Amplify server and test data transactions between the backend and the front end server.

### 1.a AWS dynamo db connection
After investigating the AWS SDK for python, I found that the boto3 library is the most popular and well documented.
Tracking down the data storation system of Amplify, we could notice that dynamodb was used and its endpoint could be found in AWS console.


```python
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
from datetime import datetime, timedelta
from numpy import random

logger = logging.getLogger(__name__)

MAX_GET_SIZE = 100  # Amazon DynamoDB rejects a get batch larger than 100 items.

aws_acct = 'zirl2v2iw5h7tnbs3wznskuvvm-bulkupcoac'
USER_TABLE_NAME = 'User-' + aws_acct
EXERCISE_TABLE_NAME = "Exercise-" + aws_acct
PROTEIN_TABLE_NAME = "Protein-" + aws_acct
SLEEP_TABLE_NAME = "Sleep-" + aws_acct

# Creating the DynamoDB Client
dynamodb_client = boto3.client('dynamodb', region_name="us-west-1")

# Creating the DynamoDB Table Resource
dynamodb = boto3.resource('dynamodb', region_name="us-west-1")

user_table = dynamodb.Table(USER_TABLE_NAME)
exercise_table = dynamodb.Table(EXERCISE_TABLE_NAME)
protein_table = dynamodb.Table(PROTEIN_TABLE_NAME)
sleep_table = dynamodb.Table(SLEEP_TABLE_NAME)
```

### 1.b Connecting to backend database
After creating and configuring the backend database separately, we connect the database to allow syncronous data transactions.


```python
# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="bulkupcoach",
    user="bulkupcoach",
    password="password",
    host="localhost",
    port = '5432',
)

# Create a cursor
cur = conn.cursor()
```

### 1.c Testing data transfer
After connecting the two desired databases, we tested if data could be retrieve, inserted, and updated on both ends.

#### Getting items from AWS by user ID


```python
def get_items_by_user(table, user_id):
    """Get items in a table by user ID.

    :param table: The table to query.
    :param user_id: The user ID to search for.
    :return: The items found.
    """
    try:
        response = table.query(
            KeyConditionExpression=Key('userID').eq(user_id)
        )
    except ClientError as e:
        logger.exception("Couldn't get items by user ID.")
        raise
    return response['Items']
```

#### Getting All items from Users Table into Postgres DB


```python
def print_all_items(table):
    """Print all items in a table.

    :param table: The table to print.
    """
    try:
        response = table.scan()
        items = response['Items']
        for item in items:
            print(item)
    except ClientError as e:
        logger.exception("Couldn't get items by user ID.")
        raise
```

    [{'id': 'user4', 'BMI': Decimal('19')}, {'id': 'user5', 'BMI': Decimal('24')}, {'id': 'user2', 'BMI': Decimal('20')}, {'id': 'user', 'BMI': Decimal('21')}, {'id': 'user6', 'BMI': Decimal('18.5')}, {'id': 'user3', 'BMI': Decimal('21.5')}]


#### Printing attribute names to confirm seamless data transfer


```python
def print_attribute_names(items):
    attribute_names = set()  # Using a set to ensure uniqueness

    for item in items:
        # Extract keys (attribute names) from the item
        for key in item.keys():
            attribute_names.add(key)

    # Print the attribute names
    print("Attribute Names:")
    for attribute_name in attribute_names:
        print(attribute_name)
```

#### Get all users from AWS database, and insert them into Local DB


```python
# Fetch data from DynamoDB and insert into PostgreSQL tables
def import_user_from_aws():
    response = user_table.scan()
    items = response['Items']
    for item in items:
        user_id = item.get('id', '')  # Assuming 'id' is the primary key in DynamoDB
        bmi = item.get('BMI', '')  # Assuming 'BMI' is an attribute in DynamoDB
        # Insert item into PostgreSQL table
        # Assuming the structure of DynamoDB items and PostgreSQL tables are compatible
        cur.execute(
            f"INSERT INTO graphapi_user (id, bmi) VALUES (%s, %s)",
            (user_id['S'], float(bmi['N']))
        )
        conn.commit()
```

    user4 {'N': '19'} <class 'dict'>
    user5 {'N': '24'} <class 'dict'>
    user2 {'N': '20'} <class 'dict'>
    user {'N': '21'} <class 'dict'>
    user6 {'N': '18.5'} <class 'dict'>
    user3 {'N': '21.5'} <class 'dict'>


#### Get all exercises from AWS database, and insert them into Local DB


```python
def import_exercise_from_aws():
    e_response = exercise_table.scan()
    e_items = e_response['Items']
    for item in e_items:
        e_id = item.get('id', '')
        completedAt = item.get('completedAt', '')
        reps = item.get('reps', '')
        userID = item.get('userID', '')
        target = item.get('target', '')
        weight_lb = item.get('weight_lb', '')
        name = item.get('name', '')
        print(item)
        # Insert item into PostgreSQL table
        # Assuming the structure of DynamoDB items and PostgreSQL tables are compatible
        cur.execute(
            f"INSERT INTO graphapi_exercise (id, name, completed_at, target, user_id, weight_lb, reps) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (e_id, name, completedAt, target, userID, weight_lb, reps)
        )
        conn.commit()
```

    {'completedAt': '02/22/2024, 21:26:04', 'reps': Decimal('20'), 'userID': 'user4', 'target': 'back', 'weight_lb': Decimal('25'), 'id': 'ex4', 'name': 'latpulldown'}
    {'completedAt': '02/22/2024, 21:25:43', 'reps': Decimal('10'), 'userID': 'user1', 'target': 'back', 'weight_lb': Decimal('35'), 'id': 'ex1', 'name': 'latpulldown'}
    {'completedAt': '02/22/2024, 21:25:36', 'reps': Decimal('15'), 'userID': 'user1', 'target': 'back', 'weight_lb': Decimal('35'), 'id': 'ex3', 'name': 'latpulldown'}
    {'completedAt': '02/22/2024, 21:25:50', 'reps': Decimal('10'), 'userID': 'user1', 'target': 'back', 'weight_lb': Decimal('25'), 'id': 'ex2', 'name': 'latpulldown'}


Workgin on Protein table


```python
p_response = protein_table.scan()
p_items = p_response['Items']
print(p_items)
```

    [{'completedAt': '02/22/2024, 21:37:35', 'id': 'pr5', 'name': 'shake', 'grams': Decimal('10'), 'userID': 'user4'}, {'completedAt': '02/22/2024, 21:37:17', 'id': 'pr4', 'name': 'burger', 'grams': Decimal('25'), 'userID': 'user4'}]


#### Get all Proteins from AWS database, and insert them into Local DB


```python
def import_protein_from_aws():
    for item in p_items:
        p_response = protein_table.scan()
        p_items = p_response['Items']
        p_id = item.get('id', '')
        completedAt = item.get('completedAt', '')
        userID = item.get('userID', '')
        name = item.get('name', '')
        grams = item.get('grams', '')
        print(item)
        # Insert item into PostgreSQL table
        # Assuming the structure of DynamoDB items and PostgreSQL tables are compatible
        cur.execute(
            f"INSERT INTO graphapi_protein (id, name, completed_at, user_id, grams) VALUES (%s, %s, %s, %s, %s)",
            (p_id, name, completedAt, userID, grams)
        )
        conn.commit()
```

    {'completedAt': '02/22/2024, 21:37:35', 'id': 'pr5', 'name': 'shake', 'grams': Decimal('10'), 'userID': 'user4'}
    {'completedAt': '02/22/2024, 21:37:17', 'id': 'pr4', 'name': 'burger', 'grams': Decimal('25'), 'userID': 'user4'}
    {'id': 'first'}



    ---------------------------------------------------------------------------

    InvalidDatetimeFormat                     Traceback (most recent call last)

    Cell In[67], line 10
          7 print(item)
          8 # Insert item into PostgreSQL table
          9 # Assuming the structure of DynamoDB items and PostgreSQL tables are compatible
    ---> 10 cur.execute(
         11     f"INSERT INTO graphapi_protein (id, name, completed_at, user_id, grams) VALUES (%s, %s, %s, %s, %s)",
         12     (p_id, name, completedAt, userID, grams)
         13 )
         14 conn.commit()


    InvalidDatetimeFormat: invalid input syntax for type timestamp with time zone: ""
    LINE 1: ...ompleted_at, user_id, grams) VALUES ('first', '', '', '', ''...
                                                                 ^



Putting items to AWS dynamodb database


```python
def import_sleep_from_aws():
    s_response = sleep_table.scan()
    s_items = s_response['Items']
    for item in s_items:
        s_id = item.get('id', '')
        end_at = item.get('endAt', '')
        userID = item.get('userID', '')
        start_at = item.get('startAt', '')
        print(item)
        # Insert item into PostgreSQL table
        # Assuming the structure of DynamoDB items and PostgreSQL tables are compatible
        cur.execute(
            f"INSERT INTO graphapi_sleep (id, end_at, user_id, start_at) VALUES (%s, %s, %s, %s)",
            (s_id, end_at, userID, start_at)
        )
        conn.commit()
```




    {'ResponseMetadata': {'RequestId': 'GS4K41V8TRRD6JF65VMH70LGDJVV4KQNSO5AEMVJF66Q9ASUAAJG',
      'HTTPStatusCode': 200,
      'HTTPHeaders': {'server': 'Server',
       'date': 'Fri, 23 Feb 2024 05:26:04 GMT',
       'content-type': 'application/x-amz-json-1.0',
       'content-length': '2',
       'connection': 'keep-alive',
       'x-amzn-requestid': 'GS4K41V8TRRD6JF65VMH70LGDJVV4KQNSO5AEMVJF66Q9ASUAAJG',
       'x-amz-crc32': '2745614147'},
      'RetryAttempts': 0}}




```python
def put_user_item_aws(item):
    user_table.put_item(
        Item={
            'id': item['id'],
            'BMI': item['bmi'],
        }
    )

def put_exercise_item_aws(item):
    exercise_table.put_item(
        Item={
            'id': item['id'],
            'name': item['name'],
            'weight_lb': item['weight_lb'],
            'reps': item['reps'],
            'completedAt': item['completedAt'],
            'target': item['target'],
            'userID': item['userID'],
        }
    )

def put_protein_item_aws(item):
    protein_table.put_item(
        Item={
            'id': item['id'],
            'name': item['name'],
            'grams': item['grams'],
            'completedAt': item['completedAt'],
            'userID': item['userID'],
        }
    )

def put_sleep_item_aws(item):
    sleep_table.put_item(
        Item={
            'id': item['id'],
            'endAt': item['endAt'],
            'userID': item['userID'],
            'startAt': item['startAt'],
        }
    )
```

## Step 3: Creating Sample data
We have inserted 


```python
# # Insert 1000 sample data into tables
# for i in range(1, 1001):
#     # Generate random values with a normal distribution for BMI
#     user_bmi = random.normal(22.7, 4.2)
#     cur.execute(f"INSERT INTO graphapi_user (id, bmi) VALUES (%s, %s)", (i, user_bmi))
    
#     # Generate random values with a normal distribution for protein grams
#     protein_grams = max(0, random.normal(65, 7))
#     cur.execute(f"INSERT INTO graphapi_protein (id, user_id, name, grams, completed_at) VALUES (%s, %s, %s, %s, %s)",
#               (i, i, f"Protein {i}", protein_grams, datetime.now() - timedelta(days=i)))
    
#     # Generate random values with a normal distribution for exercise weight and reps
#     weight_lb = max(0, round(random.normal(50, 10)))
#     reps = max(1, round(random.normal(10, 2)))
#     cur.execute(f"INSERT INTO graphapi_exercise (id, user_id, name, weight_lb, reps, target, completed_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
#               (i, i, f"Exercise {i}", weight_lb, reps, "Muscle", datetime.now() - timedelta(days=i)))
    
#     # Generate random start and end times for sleep, differing by one day
#     start_time = datetime.now() - timedelta(days=i, hours=random.randint(4, 10))
#     end_time = start_time + timedelta(hours=random.randint(4, 10))
#     cur.execute(f"INSERT INTO graphapi_sleep (id, user_id, start_at, end_at) VALUES (%s, %s, %s, %s)",
#               (i, i, start_time, end_time))

# # Commit changes
# conn.commit()
```


```python
# Insert 100 sample for a user data into tables
TARGET_MUSCLE = ["Chest", "Back", "Legs", "Shoulders", "Arms"]
for i in range(1, 101):
    random_muscle = random.choice(TARGET_MUSCLE)
    
    # Generate random values with a normal distribution for exercise weight and reps
    weight_lb = max(0, round(random.normal(50, 10)))
    reps = max(1, round(random.normal(10, 2)))
    cur.execute(f"INSERT INTO graphapi_exercise (id, user_id, name, weight_lb, reps, target, completed_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
              (i+1200, 1, f"Exercise {i}", weight_lb, reps, random_muscle, datetime.now() - timedelta(days=i)))
    
# Commit changes
conn.commit()
```

## Step 4. Making Predictions

Using softmax, standard normal distribution, and Gaussian Naive Beyes algorithm, our machine learning models predict users' life habits and results.

If a result is off 2 sigma, they are considered harmful.

In other cases, the results produce a single scalar value.

### 4.1 Softmax Function
Given a vector of \( n \) real-valued scores \( z = (z_1, z_2, ..., z_n) \), the softmax function computes the probability distribution over \( n \) different classes.

The softmax function is defined as follows:

$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}} \quad \text{for} \quad i = 1, 2, ..., n$
where \( e \) is the base of the natural logarithm (Euler's number).

#### Properties
- The softmax function outputs a probability distribution, where each output is in the range \( [0, 1] \) and the sum of all outputs equals \( 1 \).
- The softmax function is differentiable, which makes it suitable for use in gradient-based optimization algorithms like stochastic gradient descent.


### 4.2 Gaussian Naive Bayes Algorithm

#### Introduction
Gaussian Naive Bayes (GNB) is a simple probabilistic classifier based on applying Bayes' theorem with the assumption of independence between features. It is particularly useful when dealing with continuous data, where the assumption of normal distribution (Gaussian distribution) for each feature is reasonable.

#### Algorithm Overview
Given a dataset with \( n \) features and \( m \) classes, the GNB algorithm involves the following steps:

1. **Data Preprocessing**: Calculate the mean and standard deviation of each feature for each class in the training dataset.
2. **Training**: Compute the prior probabilities of each class and the likelihoods of each feature given each class.
3. **Prediction**: For a new data point, calculate the posterior probability of each class using Bayes' theorem and select the class with the highest probability as the predicted class.

#### Assumptions
- **Independence**: GNB assumes that the features are conditionally independent given the class label.
- **Normal Distribution**: GNB assumes that the likelihood of the features given the class label follows a Gaussian (normal) distribution.

#### Mathematical Formulation
The probability density function (PDF) of the Gaussian distribution is given by:

$f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$

where x is the feature value, $\mu$ is the mean of the feature, and $\sigma^2$ is the variance of the feature.


### 4.3 Normal distribution heuristics
Since the actual user data can't be used with the initiative machine learning training process, we proceeded with a simple heuristic of rewarding activities within 1 sigma from the mean behavior.
As we can see from the plot below, values further than 1 sigma accounts for about 32% of data, which are off from normal behaviors.
Since excessive workout or nutrient intake are not recommended, our models' final scalar value will punish behaviors that are off from the mean.

#### Potential progress
As users input their actual data while using the application, we will be able to obtain real data and predicted updates.
When we have enough data to train machine learning models with actual target labels, we can enhance accuracy of our models using the real user data instead of this heuristic.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate data points for the normal distribution
x = np.linspace(-5, 5, 1000)
y = norm.pdf(x, 0, 1)  # Mean=0, Standard Deviation=1

# Plot the normal distribution
plt.plot(x, y, 'k-', linewidth=2, label='Normal Distribution')

# Find the area within 1 standard deviation
x_fill = np.linspace(-1, 1, 1000)
y_fill = norm.pdf(x_fill, 0, 1)
plt.fill_between(x_fill, y_fill, color='lightblue', alpha=0.5, label='Within 1 $\sigma$')

# Set labels and title
plt.xlabel('X')
plt.ylabel('Probability Density')
plt.title('Normal Distribution with Area within 1 $\sigma$')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
```


    
![png](output_30_0.png)
    



```python

```
