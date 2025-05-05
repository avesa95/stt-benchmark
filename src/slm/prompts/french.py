def french_grammar_correction_prompt(incorrect_sentence: str) -> str:
    few_shot_examples = [
        {
            "input": "Donc euh hier on était genre en train de faire le projet mais personne a rien foutu sauf moi et peut-être Sarah.",
            "output": "Hier, on faisait le projet, mais personne n’a rien fait sauf moi et peut-être Sarah.",
        },
        {
            "input": "Il dit qu’il en a fini avec le drama mais genre à chaque fois y’a un truc il est au milieu.",
            "output": "Il dit qu’il en a fini avec le drama, mais à chaque fois qu’il se passe quelque chose, il est au milieu.",
        },
        {
            "input": "Elle sait pas trop ce qu’elle veut faire après les études, peut-être voyager ou trouver un taf, tu vois ?",
            "output": "Elle ne sait pas vraiment ce qu’elle veut faire après les études, peut-être voyager ou trouver un travail.",
        },
        {
            "input": "Euh je crois que genre on devrait ptêt parler à Julien avant de prendre une grosse décision parce que sinon ça fait chelou.",
            "output": "Je pense qu’on devrait parler à Julien avant de prendre une décision, sinon ça paraît bizarre.",
        },
        {
            "input": "J’étais genre en train de marcher vers le magasin quand j’ai vu un mec que je crois que je connais de la fac mais pas sûr.",
            "output": "Je marchais vers le magasin quand j’ai vu un gars que je crois connaître de la fac, mais je n’en suis pas sûr.",
        },
        {
            "input": "Donc euh hier on était genre en train de faire le projet mais personne a rien foutu sauf moi et peut-être Sarah.",
            "output": "Hier, on faisait le projet, mais personne n’a rien fait sauf moi et peut-être Sarah.",
        },
        {
            "input": "Il dit qu’il en a fini avec le drama mais genre à chaque fois y’a un truc il est au milieu.",
            "output": "Il dit qu’il en a fini avec le drama, mais à chaque fois qu’il se passe quelque chose, il est au milieu.",
        },
        {
            "input": "Elle sait pas trop ce qu’elle veut faire après les études, peut-être voyager ou trouver un taf, tu vois ?",
            "output": "Elle ne sait pas vraiment ce qu’elle veut faire après les études, peut-être voyager ou trouver un travail.",
        },
        {
            "input": "Euh je crois que genre on devrait ptêt parler à Julien avant de prendre une grosse décision parce que sinon ça fait chelou.",
            "output": "Je pense qu’on devrait parler à Julien avant de prendre une décision, sinon ça paraît bizarre.",
        },
        {
            "input": "J’étais genre en train de marcher vers le magasin quand j’ai vu un mec que je crois que je connais de la fac mais pas sûr.",
            "output": "Je marchais vers le magasin quand j’ai vu un gars que je crois connaître de la fac, mais je n’en suis pas sûr.",
        },
        {
            "input": "Alors j’étais genre au téléphone avec mon pote et genre y’avait plein de bruit autour et j’arrivais pas trop à entendre mais en gros il disait qu’il allait venir mais genre peut-être plus tard ou pas du tout, je sais plus trop.",
            "output": "J’étais au téléphone avec mon pote, il y avait beaucoup de bruit autour, et j’avais du mal à entendre. En gros, il disait qu’il allait venir, peut-être plus tard, ou peut-être pas du tout. Je ne suis plus très sûr.",
        },
        {
            "input": "Tu vois genre hier quand on a parlé du plan pour ce week-end ben je croyais qu’on avait dit samedi mais apparemment tout le monde a compris dimanche et du coup j’ai genre tout planifié pour rien.",
            "output": "Hier, quand on a parlé du plan pour ce week-end, je croyais qu’on avait dit samedi, mais apparemment tout le monde avait compris dimanche. Du coup, j’ai tout planifié pour rien.",
        },
    ]

    instructions = (
        "Tu es un assistant qui corrige les fautes dans des phrases en français parlé.\n"
        "Ta tâche est de produire une version correcte de la phrase suivante, sans changer son style oral ni son sens.\n\n"
        "Corrige en respectant ces règles strictes :\n"
        "✅ Corrige toutes les fautes d’accord, de grammaire, de conjugaison et de ponctuation.\n"
        "✅ Supprime les mots inutiles comme « euh », « genre », « tu vois », « ben », etc.\n"
        "✅ Garde la structure originale sauf si la phrase est trop longue ou confuse.\n"
        "✅ Divise les phrases longues uniquement si c’est indispensable pour la clarté.\n"
        "✅ Rends le tout naturel et fluide, comme si un locuteur natif s’exprimait à l’oral, mais de manière correcte.\n\n"
        "❌ Ne reformule pas tout le contenu.\n"
        "❌ Ne change pas d’intention ou de ton.\n"
        "❌ Ne dis pas « La phrase corrigée est ».\n"
        "❌ Ne donne pas d’explication.\n\n"
        "📝 Réponds uniquement avec la version corrigée de la phrase.\n"
    )

    examples_text = ""
    for ex in few_shot_examples:
        examples_text += f"Entrée : {ex['input']}\nSortie : {ex['output']}\n---\n"

    return f"{instructions}\n{examples_text}Entrée : {incorrect_sentence}\nSortie :"
