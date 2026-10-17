import re
import json
import unicodedata
import pandas as pd
from nltk.stem.snowball import FrenchStemmer

# Installer si nécessaire :
# python -m pip install pandas openpyxl nltk

# 1. Charger le corpus
with open("corpus_B.json", encoding="utf-8") as f:
    corpus = json.load(f)

stemmer = FrenchStemmer()

# 2. Expressions régulières
email_re = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I
)
url_re = re.compile(r"https?://[^\s]+", re.I)

# Téléphone masqué seulement après "tel :" avec 10 chiffres
phone_re = re.compile(r"(?i)\btel\s*:\s*\d{10}\b")

# 3. Tokenizer : les codes, versions, montants et heures restent intacts
token_re = re.compile(
    r"EMAILTOKEN|PHONETOKEN|URLTOKEN|"
    r"\bE\d{3,}\b|"
    r"\b\d+(?:\.\d+)+\b|"
    r"\b\d{1,2}:\d{2}\b|"
    r"\b\d+(?:[.,]\d+)?\s*DH\b|"
    r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-zÀ-ÖØ-öø-ÿ]+)?|"
    r"\d+",
    re.I
)

protected_re = re.compile(
    r"^(EMAILTOKEN|PHONETOKEN|URLTOKEN|E\d{3,}|"
    r"\d+(?:\.\d+)+|\d{1,2}:\d{2}|"
    r"\d+(?:[.,]\d+)?\s*DH|\d+)$",
    re.I
)

# 4. Dictionnaire pédagogique de lemmatisation
lemma_map = {
    "erreurs": "erreur",
    "connexions": "connexion",
    "échouent": "échouer",
    "échoue": "échouer",
    "passe": "passer",
    "passent": "passer",
    "fonctionne": "fonctionner",
    "fonctionnent": "fonctionner",
    "supprimés": "supprimer",
    "supprimées": "supprimer",
    "fichiers": "fichier",
    "sont": "être",
    "est": "être",
    "les": "le",
    "des": "de",
    "messages": "message",
    "versions": "version",
    "contacts": "contact",
    "fermetures": "fermeture",
    "la": "le"
}


# 5. Normaliser les commentaires pour détecter les doublons
def normaliser_cle(texte):
    texte = unicodedata.normalize("NFKC", texte).lower()
    texte = texte.replace("’", "'")
    texte = re.sub(r"[^\w\s'.:]", " ", texte)
    return re.sub(r"\s+", " ", texte).strip().rstrip(".")


# 6. Masquer les informations sensibles
def masquer(texte):
    texte = email_re.sub("EMAILTOKEN", texte)
    texte = url_re.sub("URLTOKEN", texte)
    texte = phone_re.sub("tel : PHONETOKEN", texte)
    return texte


# 7. Identifier les doublons
cles = {}

for item in corpus:
    cle = normaliser_cle(item["text_raw"])
    cles.setdefault(cle, []).append(item["id"])


# 8. Préparer les résultats
lignes = []
details_tokens = []

# 9. Traiter chaque commentaire
for item in corpus:
    brut = item["text_raw"]
    identifiant = item["id"]

    cle = normaliser_cle(brut)
    masque = masquer(brut)

    # Extraire les mots et les tokens
    tokens = token_re.findall(masque)

    stems = []
    lemmes = []

    # Calculer les stems et les lemmes
    for token in tokens:
        if protected_re.fullmatch(token):
            stems.append(token)
            lemmes.append(token)
        else:
            mot = token.lower().replace("’", "'")
            stems.append(stemmer.stem(mot))
            lemmes.append(lemma_map.get(mot, mot))

    # 10. AJOUTER UN INDICE À CHAQUE MOT
    for indice, token in enumerate(tokens):
        details_tokens.append({
            "ID commentaire": identifiant,
            "Indice du mot": indice,
            "Mot (token)": token,
            "Stem": stems[indice],
            "Lemme": lemmes[indice]
        })

    # 11. Enregistrer les résultats du commentaire
    groupe = cles[cle]

    lignes.append({
        "ID": identifiant,
        "Texte brut": brut,
        "Texte masqué": masque,
        "Tokens": " | ".join(tokens),
        "Stems": " | ".join(stems),
        "Lemmes (approximatifs)": " | ".join(lemmes),
        "Clé de doublon avant masquage": cle,
        "Groupe de doublons": (
            " / ".join(groupe)
            if cle and len(groupe) > 1
            else "Aucun doublon"
        )
    })


# 12. Exporter les résultats généraux
df_resultats = pd.DataFrame(lignes)

df_resultats.to_excel(
    "correction_TP_Text_Mining_B.xlsx",
    index=False
)

df_resultats.to_csv(
    "correction_TP_Text_Mining_B.csv",
    index=False,
    encoding="utf-8-sig"
)


# 13. Exporter les mots avec leurs indices
df_indices = pd.DataFrame(
    details_tokens,
    columns=[
        "ID commentaire",
        "Indice du mot",
        "Mot (token)",
        "Stem",
        "Lemme"
    ]
)

df_indices.to_excel(
    "indices_mots_Text_Mining_B.xlsx",
    index=False
)

df_indices.to_csv(
    "indices_mots_Text_Mining_B.csv",
    index=False,
    encoding="utf-8-sig"
)


# 14. Afficher les résultats dans le terminal
print("Traitement terminé !")
print("Nombre de commentaires :", len(corpus))
print("Nombre total de tokens :", len(details_tokens))

print("\nAperçu des mots avec leurs indices :")
print(df_indices.head(30).to_string(index=False))

print("\nFichiers créés :")
print("- correction_TP_Text_Mining_B.xlsx")
print("- correction_TP_Text_Mining_B.csv")
print("- indices_mots_Text_Mining_B.xlsx")
print("- indices_mots_Text_Mining_B.csv")