import os
import pandas as pd
import numpy as np
import re
import time

'''
### use ###
import make_dict


# item info
movie_genre_dict, movie_year_dict, movie_director_dict, movie_writer_dict = make_dict.make_movie_info_dict()

# user info (train <= train set)
user_genre_dict, user_year_dict, user_director_dict, user_writer_dict = make_dict.make_user_info_dict(train)

'''


def preprocessing():
    path = '/opt/ml/input/data/train'
    titles = pd.read_csv(os.path.join(path,'titles.tsv'), sep="\t")
    genres = pd.read_csv(os.path.join(path,'genres.tsv'), sep="\t")
    years = pd.read_csv(os.path.join(path,'years.tsv'), sep="\t")
    directors = pd.read_csv(os.path.join(path, 'directors.tsv'), sep="\t")
    writers = pd.read_csv(os.path.join(path, 'writers.tsv'), sep="\t")

    genres = genres.groupby(['item']).agg({'genre':'unique'}).reset_index()
    directors = directors.groupby(['item']).agg({'director':'unique'}).reset_index()
    writers = writers.groupby(['item']).agg({'writer':'unique'}).reset_index()

    # data merge
    movie_info_df = pd.merge(titles, years, how='left', on='item')
    movie_info_df = pd.merge(movie_info_df, genres, how='left', on='item')
    movie_info_df = pd.merge(movie_info_df, directors, how='left', on='item')
    movie_info_df = pd.merge(movie_info_df, writers, how='left', on='item')

    # year 결측치 처리
    none_year_idx = movie_info_df[movie_info_df['year'].isnull()].index
    none_year_title = movie_info_df.iloc[none_year_idx].title.values
    none_year_movie_dict = dict([(i, int(re.findall('\((\d{4})\)',t)[0])) for i, t in zip(none_year_idx, none_year_title)])
    for i in none_year_idx:
        movie_info_df.iloc[i,np.where(movie_info_df.columns == 'year')[0]] = none_year_movie_dict[i]
    movie_info_df['year'] = movie_info_df['year'].astype('int')

    return movie_info_df

def make_movie_info_dict():
    movie_info_df = preprocessing()
    # make movie-genre dict
    movie_genre_dict = dict([(u,list(g)) for u,g in movie_info_df[['item','genre']].values])

    # make movie-year dict
    movie_year_dict = dict([(i,y) for i,y in movie_info_df[['item','year']].values])

    # make movie-director dict
    movie_director_dict = dict([(i,d) for i,d in movie_info_df[['item','director']].values if d is not np.NAN])

    # make movie-writer dict
    movie_writer_dict = dict([(i,w) for i,w in movie_info_df[['item','writer']].values if w is not np.NAN])

    return movie_genre_dict, movie_year_dict, movie_director_dict, movie_writer_dict

def make_user_info_dict(train: pd.DataFrame):
    data = train.copy()
    data['datetime'] = data['time'].apply(lambda x: time.strftime('%Y-%m-%d', time.localtime(x)))
    data['datetime'] = pd.to_datetime(data['datetime'])
    data['year'] = data['datetime'].dt.year

    movie_genre_dict, movie_year_dict, movie_director_dict, movie_writer_dict = make_movie_info_dict()

    # user-movie list data
    user_movie_data = data.groupby(['user']).agg({'item':'unique'}).reset_index()

    user_id_ls = user_movie_data.user.values

    # make user-genre dict
    user_movie_preference_ls = user_movie_data['item'].apply(lambda x :[j for i in x for j in movie_genre_dict[i]]).values
    user_genre_dict = dict([(u,p)for u,p in zip(user_id_ls, user_movie_preference_ls)])

    # make user-year dict (user last view year + std)
    user_year_data = data.groupby(['user']).agg({'year':'unique'}).reset_index()
    # user_year_dict = dict([(u, max(y)+1) if len(y)==1 else(u, max(y)+round(np.std(y))) for u,y in user_year_data.values])
    user_year_dict = dict([(u, max(y)) for u,y in user_year_data.values])

    # make user-director dict
    user_director_preference_ls = user_movie_data['item'].apply(lambda x :[j for i in x if i in movie_director_dict for j in movie_director_dict[i]]).values
    user_director_dict = dict([(u,d)for u,d in zip(user_id_ls, user_director_preference_ls)])

    # make user-writer dict
    user_writer_preference_ls = user_movie_data['item'].apply(lambda x :[j for i in x if i in movie_writer_dict for j in movie_writer_dict[i]]).values
    user_writer_dict = dict([(u,w)for u,w in zip(user_id_ls, user_writer_preference_ls)])

    return user_genre_dict, user_year_dict, user_director_dict, user_writer_dict