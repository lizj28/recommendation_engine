---
# Page settings
layout: default

title: Recommendation Engine
description: Let's build our recommendation engine.  We'll start by cleaning our data. Then we'll prepare our features.  Finally, we'll build our engine!

---

## Step 1 - Inspection

Let's read in our data from our csv and then inspect it.

```python
import pandas as pd
from typing import List
from utils.cleaning import lower_case_and_strip_spaces
from utils.cleaning import combine_genres_list
```


```python
movies_df: pd.DataFrame = pd.read_csv('input/all_movies.csv')
movies_df.sample(20)
```
![img.png](images/img.png)
```python
movies_df
```

```python
movies_df.shape[0]
```


## Step 2 - Cleaning

Now let's clean our input data. Some things we'll do here are:

1. Cleaning up strings
2. Removing unwanted rows
3. Checking our data for duplicates

```python
movies_cleaned_df = movies_df.copy()
movies_cleaned_df['genres'] = movies_cleaned_df['genres'].apply(lower_case_and_strip_spaces)
```

```python

```

```python


```

```python


```

```python


```

```python


```

```python


```