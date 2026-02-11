executer le code : <br>
selon votre version  entrer dans le terminal: 
python3 main.py ou python main.py
une fois l'apprentissage terminé (relativement long la première fois) -> écrire les relations à tester
exemple : feu de bois 

Les différents fichiers : 
config.py = Constantes utilisées dans le code
data_loader = lecteur json et split du corpus
evaluate = fonctions d'évaluation du modèle (obsolète)
grasp_it = classe du modèle d'apprentissage
jdm_client = code appel API
signature = Classe des termes sous la forme {hyperonymes, types sémantiques, règles présentes}
main = parser de texte "A de B" et execution du code
