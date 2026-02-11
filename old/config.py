"""Configuration du projet A de B - Analyseur Semantique."""

# === API JeuxDeMots ===
JDM_API_BASE = "https://jdm-api.demo.lirmm.fr/v0"

# === Chemins ===
LEARN_DIR = "Learn/"
CACHE_DIR = "cache/"

# === IDs des types de relations JDM utilises pour la classification ===
# Mapping : nom du fichier corpus -> ID de la relation JDM correspondante
RELATION_TYPES = {
    "r_has_causatif": 42,       # Consequence (Co)
    "r_has_property-1": 153,    # Caracterisation (Ca) - r_has_prop dans JDM
    "r_objetmatiere": 50,       # Matiere/Composition (M) - r_object>mater
    "r_lieuorigine": 171,       # Origine (O) - r_lieu>origine
    "r_topic": 142,             # Topic (T) - r_has_topic
    "r_depict": 172,            # Depiction (D)
    "r_holo": 10,               # Holonymie (H)
    "r_lieu": 15,               # Lieu (L)
    "r_processus_agent": 70,    # Agent (A) - r_processus>agent
    "r_processus_patient": 76,  # Patient (P) - r_processus>patient
    "r_processus_instr": 80,    # Instrument (I) - r_processus>instr
    "r_own-1": 121,             # Possession (Po) - r_own
    "r_quantificateur": 58,     # Quantification (Q)
    "r_social_tie": 113,        # Lien social (LS) - r_has_social_tie_with
    "r_product_of": 54,         # Auteur/Createur (AC)
}

# Noms lisibles pour l'affichage
RELATION_LABELS = {
    "r_has_causatif": "Consequence (Co)",
    "r_has_property-1": "Caracterisation (Ca)",
    "r_objetmatiere": "Matiere (M)",
    "r_lieuorigine": "Origine (O)",
    "r_topic": "Topic (T)",
    "r_depict": "Depiction (D)",
    "r_holo": "Holonymie (H)",
    "r_lieu": "Lieu (L)",
    "r_processus_agent": "Agent (A)",
    "r_processus_patient": "Patient (P)",
    "r_processus_instr": "Instrument (I)",
    "r_own-1": "Possession (Po)",
    "r_quantificateur": "Quantification (Q)",
    "r_social_tie": "Lien social (LS)",
    "r_product_of": "Auteur/Createur (AC)",
}

# === IDs des relations JDM pour construire les signatures ===
R_ISA = 6           # Hyperonymes (generique)
R_HYPO = 8          # Hyponymes (specifique)
R_HAS_PART = 9      # Parties
R_HOLO = 10         # Tout (holonymie)
R_LIEU = 15         # Lieu
R_LIEU_1 = 28       # Lieu inverse
R_CARAC = 17        # Caracteristique
R_CARAC_1 = 23      # Caracteristique inverse
R_INFOPOT = 36      # Information potentielle (_INFO-SEM)
R_AGENT = 13        # Action > Agent
R_PATIENT = 14      # Action > Patient
R_INSTR = 16        # Action > Instrument
R_LIEU_ACTION = 30  # Lieu > Action
R_ACTION_LIEU = 31  # Action > Lieu

# Types de relations a inclure dans le bloc TRT des signatures
# Ce sont les relations dont la presence/absence est informative
TRT_RELATION_IDS = [
    R_ISA, R_HYPO, R_HAS_PART, R_HOLO, R_LIEU, R_LIEU_1,
    R_CARAC, R_CARAC_1, R_AGENT, R_PATIENT, R_INSTR,
    R_LIEU_ACTION, R_ACTION_LIEU,
    42,   # r_has_causatif
    50,   # r_object>mater
    54,   # r_product_of
    58,   # r_quantificateur
    70,   # r_processus>agent
    76,   # r_processus>patient
    80,   # r_processus>instr
    113,  # r_has_social_tie_with
    121,  # r_own
    122,  # r_own-1
    142,  # r_has_topic
    153,  # r_has_prop
    171,  # r_lieu>origine
    172,  # r_depict
]

# === Parametres GRASP-it ===
FUSION_THRESHOLD = 0.5  # Seuil de similarite cosinus pour fusionner deux regles
TRAIN_RATIO = 0.8       # Ratio train/test
RANDOM_SEED = 42
API_RATE_LIMIT = 0.05   # Secondes entre les appels API (50ms)
API_REQUEST_LIMIT = 200  # Nombre max de relations par requete
