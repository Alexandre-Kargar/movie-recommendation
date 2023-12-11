# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 12:19:42 2023
Last update on Dec  10 2023

@author: Hosse
@author: Ken 
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
# ------------------------------------------------------------------------------------------------------------------

# Étape 1 : Charger les données
df_movies = pd.read_csv('./tmdb_5000_movies.csv')
df_ratings = pd.read_csv('./user_ratings.csv')  # Assurez-vous d'avoir le bon chemin vers le fichier
# ------------------------------------------------------------------------------------------------------------------

# Étape 2 : Préparer les données pour le filtrage collaboratif
movie_user_mat = df_ratings.pivot(index='movie_id', columns='user_id', values='rating').fillna(0)
hashmap = {movie: i for i, movie in enumerate(list(df_movies.set_index('id').loc[movie_user_mat.index].title))}
inverse_hashmap = {v: k for k, v in hashmap.items()}
# ------------------------------------------------------------------------------------------------------------------

# Créer un hashmap pour les utilisateurs
user_hashmap = {user: i for i, user in enumerate(df_ratings['user_id'].unique())}
inverse_user_hashmap = {v: k for k, v in user_hashmap.items()}

movie_user_mat_sparse = csr_matrix(movie_user_mat.values)

# Étape 3 : Calculer la matrice de similarité cosinus
cosine_sim = linear_kernel(movie_user_mat_sparse, movie_user_mat_sparse)

# ------------------------------------------------------------------------------------------------------------------
# Étape 4 : Créer une fonction pour obtenir des recommandations basées sur la similarité cosinus
def get_recommendations(title, cosine_sim=cosine_sim):
    # Obtenir l'index du film qui correspond au titre
    idx = hashmap[title]

    # Obtenir les scores de similarité pour tous les films avec ce film
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Trier les films en fonction des scores de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtenir les indices des films les plus similaires
    movie_indices = [i[0] for i in sim_scores[1:6]]

    # Retourner les cinq films les plus similaires
    return df_movies['title'].iloc[movie_indices]

# ------------------------------------------------------------------------------------------------------------------
def get_closest_titles(input_title, df_movies):
    # Trouver tous les films contenant le titre partiel
    matching_titles = df_movies[df_movies['title'].str.contains(input_title, case=False)]

    # S'il n'y a pas de correspondance
    if matching_titles.empty:
        print("\nDésolé, aucun film correspondant trouvé dans notre base de données.")
        return None

    # S'il y a une seule correspondance exacte
    if len(matching_titles) == 1 and matching_titles.iloc[0]['title'].lower() == input_title.lower():
        return matching_titles.iloc[0]['title']

    # S'il y a plusieurs correspondances
    print("\nAvez-vous voulu dire l'un de ces films ?")
    for i, title in enumerate(matching_titles['title']):
        print(f"{i + 1}. {title}")

    choice = int(input("\nChoisissez le numéro du film pour lequel vous voulez des recommandations : "))
    return matching_titles.iloc[choice - 1]['title']

def get_recommendations_for_closest_title(input_title, df_movies, cosine_sim):
    title = get_closest_titles(input_title, df_movies)
    if title:
        return get_recommendations(title, cosine_sim)
    else:
        return "Aucune recommandation disponible pour cette entrée."
    
# ------------------------------------------------------------------------------------------------------------------


def get_user_recommendations(user_id, cosine_sim=cosine_sim):
    # Obtenir l'index de l'utilisateur qui correspond à l'ID de l'utilisateur
    idx = user_hashmap[user_id]

    # Obtenir les scores de similarité pour tous les utilisateurs avec cet utilisateur
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Trier les utilisateurs en fonction des scores de similarité
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtenir les indices des utilisateurs les plus similaires
    user_indices = [i[0] for i in sim_scores if i[0] in inverse_user_hashmap][1:6]

    # Obtenir les notes des films que ces utilisateurs ont aimés
    similar_users_ratings = df_ratings[df_ratings['user_id'].isin(user_indices)]

    # Calculer la note moyenne pour chaque film
    movie_ratings = similar_users_ratings.groupby('movie_id').rating.mean()

    # Obtenir les ID des films ayant les notes moyennes les plus élevées
    top_movies_ids = movie_ratings.sort_values(ascending=False).index[:5]

    # Obtenir les titres de ces films
    top_movies = df_movies[df_movies['id'].isin(top_movies_ids)]['title']

    return top_movies

# ------------------------------------------------------------------------------------------------------------------

# Étape 5 : Obtenir des recommandations pour un film spécifique ou un utilisateur spécifique
rst = True
while rst :
    choice = input("\nVoulez-vous des recommandations basées sur le titre d'un film ou sur un ID d'utilisateur ? (film/utilisateur):\n")
    if choice.lower() == "film":
        nom = input("\nSaisir un nom de film: \n")
        nom = nom.capitalize()
        recommendations = get_recommendations_for_closest_title(nom, df_movies, cosine_sim)
        print(recommendations)
    elif choice.lower() == "utilisateur":
        user_id = int(input("\nSaisir un ID d'utilisateur (de 1 à 50): \n"))
        if user_id in user_hashmap:
            print("\nVoici quelques recommandations pour l'utilisateur", user_id)
            recommendations = get_user_recommendations(user_id)
            for title in recommendations:
                print(title)
        else :
            print("\nDésolé, cet utilisateur n'est pas dans notre base de données.")
    elif choice.lower() == "quit":
        rst = False
    else:
        print("\nChoix non valide. Veuillez choisir 'film' ou 'utilisateur'.")
        
reponse = input("\nVoulez-vous trouvez d'autres recommandations ? (oui/non):\n")
if reponse.lower() == "non":
    rst = False
    print("\nProgramme terminé.")

# ------------------------------------------------------------------------------------------------------------------
