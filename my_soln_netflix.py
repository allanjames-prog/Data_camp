# Importing pandas and matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Read in the Netflix CSV as a DataFrame
netflix_df = pd.read_csv("netflix_data.csv")

# Start coding here! Use as many cells as you like
allshows_90s = netflix_df[(netflix_df['release_year'] >= 1990) & (netflix_df['release_year'] < 2000)] 
movies_90s = allshows_90s[allshows_90s['type'] == 'Movie']
plt.hist(movies_90s['duration'])
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of movies')
plt.title('Most frequent movie duration in the 1990s')
plt.grid(True)
plt.show()

# short action movies
action_movies_90s = movies_90s[movies_90s['genre'] == 'Action']
short_movie_count = 0
for lab, row in action_movies_90s.iterrows():
    if row['duration'] < 90:
        short_movie_count += 1
    else:
        short_movie_count = short_movie_count
print(f"Number of short action movies in the 90s : {short_movie_count}")
# most frequent movie duration in the 1990s
duration = 100
print(f"Most frequent movie duration in the 90s : {duration}")

