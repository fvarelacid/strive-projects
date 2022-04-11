import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

page = requests.get("https://www.imdb.com/list/ls091520106/?st_dt=&mode=detail&page=1&sort=user_rating,desc&ref_=ttls_ref_gnr&genres=Adventure")

soup = BeautifulSoup(page.content, 'html.parser')

all_data = soup.find_all('div', class_ = "lister-item-content")


# Lists for data Storage
movie_name = []
desc = []
release_date = []
director = []
rating = []
duration = []
genre = []
actors = []
filming_dates = []
gross = []


######## Fill lists with extracted data ########


for data in all_data:
    movie_name.append(data.find('a').get_text(separator=' '))
    desc.append(data.find('p', class_= "").get_text(separator=' ')[5:])
    release_date.append(int(data.find('span', class_="lister-item-year text-muted unbold").get_text(separator=' ').strip('()I ')))
    rating.append(data.find('span', class_ = 'ipl-rating-star__rating').get_text(separator=' '))
    duration.append(int(data.find('span', class_ = 'runtime').get_text(separator=' ')[:-4]))
    genre.append(data.find('span', class_ = 'genre').get_text(separator=' ')[1:-12].split(', '))



people = soup.find_all('p', class_ ='text-muted text-small')

for person in people[1::3]:
    person1= person.get_text(separator=' ')
    splittedPerson1 = person1.split('|')
    splittedPerson1a = splittedPerson1[0][16:]
    splittedPerson1b = splittedPerson1[1][15:]
    final_actors = splittedPerson1b.replace(' \n', '')
    final_director = splittedPerson1a.replace(' \n', '')
    director.append(final_director[:-1].split(' , '))
    actors.append(final_actors.split(' , '))
    


values = soup.find_all('p', class_ = 'text-muted text-small')

for value in values[2::3]:
    raw_text = value.get_text(separator=' ')
    values_splitted = raw_text.split('|')
    gross_v = values_splitted[1].strip('Gross: ')
    gross_f = gross_v.replace('$', '').replace('M', '')
    gross.append(float(gross_f.replace('\n ', '').replace(' \n', '')))


######## Create Pandas DataFrame ########

new_dir = []
for dir in director:
    dirTuple = tuple(dir)
    new_dir.append(dirTuple)

dataList = zip(movie_name, genre, desc, release_date, new_dir, actors, rating, duration, gross)
movieData = pd.DataFrame(dataList, columns = ['Title', 'Genre', 'Summary', 'Release Date', 'Directors', 'Actors', 'Ratings', 'Duration', 'Gross Income'])

movieData['Ratings'] = pd.to_numeric(movieData['Ratings'], downcast='float')


##### STREAMLIT #####

st.title('TOP Adventure Movies Data Analysis - Web Scraping from IMDB')

st.subheader('Took into consideration the top 100 adventure movies.')

#RATING vs DURATION

if st.button('Rating vs Duration'):

    st.write('The scatter plot below shows that longer movies were slightly more appreciated.')

    fig = plt.figure(figsize=(10, 10))
    plt.scatter(
        x = movieData['Ratings'], 
        y = movieData['Duration'], 
        s=2000,
        c="magenta", 
        alpha=0.6, 
        edgecolors="white", 
        linewidth=2)

    plt.xlabel('Ratings')
    plt.ylabel('Duration')

    st.pyplot(fig)

    


#RATING vs DIRECTOR

if st.button('Rating vs Director'):

    st.write('Directors like Quentin Tarantino, Christopher Nolan and Sergio Leone have better ratings.')
    fig = plt.figure(figsize=(10, 10))
    movieData.groupby('Directors')['Ratings'].mean().plot(kind='bar', cmap='RdYlBu')
    st.pyplot(fig)


#RELEASE YEAR vs BOX OFFICE

if st.button('Release Year vs Box Office'):

    st.write("The plot below shows us that the gross income of this genre of movies has been increasing over the years.")

    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(data=movieData, x="Release Date", y="Gross Income", legend=False, s = 2000, cmap="Accent", alpha=0.7, edgecolors="grey", linewidth=2)

    st.pyplot(fig)


#RATING vs RELEASE YEAR

if st.button('Rating vs Release Year'):

    st.write("According to the plot below there's no clear correlation between the rating and the release year.")

    fig = plt.figure(figsize=(10, 10))
    movieData.groupby('Release Date')['Ratings'].mean().plot(kind='bar')

    st.pyplot(fig)


#RATING vs BOX OFFICE

if st.button('Rating vs Box Office'):

    st.write("We can see a slight increase on the box office with higher ratings but its not significant enough to be considered a correlation, since the movies with more gross income are not the best rated ones.")

    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(data=movieData, x="Ratings", y="Gross Income", legend=False, s = 2000, cmap="Accent", alpha=0.7, edgecolors="grey", linewidth=2)

    st.pyplot(fig)
