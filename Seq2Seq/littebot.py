# -*- litte_bot -*-
# -*- A.Pappa-*-
# -*- coding: utf-8 -*-

import tkinter
from tkinter import *
import random
import re
import pandas as pd
import tensorflow as tf
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import pickle
from training import textPreprocess
from training import transformer
from training import CustomSchedule
from training import accuracy, loss_function
from colorama import init
from colorama import Fore, Back
init()


your_name = input(f"{Back.BLUE}\nVeuillez indiquer votre nom: {Back.RESET}")
bot_name = "Litte_Bot"

print(f"{Back.BLUE}\n{bot_name} Presque pret...{Back.RESET}")

strategy = tf.distribute.get_strategy()

# longueur max de la phrase
MAX_LENGTH = 120

# parametres du tf.data.Dataset
BATCH_SIZE = int(64 * strategy.num_replicas_in_sync)
BUFFER_SIZE = 20000

# parametres du model Transformer
NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
UNITS = 2048
DROPOUT = 0.1

EPOCHS = 300

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# Definir le token de debut et de fin pour indiquer le debut et la fin d'une phrase
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# la taille du vocabulaire (+ le token de debut et de fin)
VOCAB_SIZE = tokenizer.vocab_size + 2



# fonction qui permet d'évaluer l'input (la phrase) grace au model pré-trainé
def evaluate(sentence, model):
  sentence = textPreprocess(sentence)
  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)
  output = tf.expand_dims(START_TOKEN, 0)

  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)
    # selectionne le dernier mot de la sequence input
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    # retourne le resultat si l'ID prédit est égale au token de fin
    if tf.equal(predicted_id, END_TOKEN[0]):
      break
    # concaténation de l'ID prédit avec l'output, le tout donné comme input au décodeur
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)



# fonction qui permet de prédire la phrase output (la réponse du bot) selon l'input et le model pré-trainé
def predict(sentence,model):
  prediction = evaluate(sentence,model)
  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])
  return predicted_sentence.lstrip()




learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

model = transformer(
      vocab_size=VOCAB_SIZE,
      num_layers=NUM_LAYERS,
      units=UNITS,
      d_model=D_MODEL,
      num_heads=NUM_HEADS,
      dropout=DROPOUT)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
model.load_weights('saved_weights68k-940ep.h5')


# contexte de base, ou d'application spécifique du bot
debut = re.compile("(.*(salut|hey|hello|bonjour)(.*))")
debut_ans = ["bien le bonjour", "salutation", "bienvenue"]

cava = re.compile("(.*(ca va|ça va|comment vas-tu|comment ça va|comment allez-vous|vas bien)(.*))")
cava_ans = ["à merveille, et vous", "on ne peut mieux, merci et vous", "je vais bien, c'est fort gentil de votre part, et vous", "je vous adresse mes meilleures salutations"]

quiestu = re.compile("(.*(qui es-tu|t'es qui|qui êtes-vous|vous êtes qui|comment tu t'appelles|quel est votre nom)(.*))")
quiestu_ans = ["je suis un personnage de Molière", "je suis un homme honnête et homme d'honneur et je ne lâche aucun mot qui ne sorte du coeur", "vous pouvez m'appeler Dom Juan", "je suis votre humble serviteur"]

qui_don = re.compile("(.*(qui est Don Juan|c'est qui Don Juan|connaissez-vous Don Juan|comment est Don Juan|Dom Juan)(.*))")
qui_don_ans = ["Dom Juan est l'archetype du séducteur", "il est le personnage principal de Festin de Pierre", "ah ! mon personnage préféré", "sans prétention... il est un homme libre", "Mon raisonnement est qu’il y a quelque chose d’admirable dans l’homme, quoi que vous puissiez dire, que tous les savants ne sauraient expliquer", "il se plaît à se promener de liens en liens, et n’aime guère à demeurer en place"]

tue = re.compile("(.*(qui tue Don Juan|mort Don Juan)(.*))")
tue_ans = ["Don Juan est mort de la main du Commandeur", "Don Juan est injustement assassiné par la justice divine"]

piece = re.compile("(.*(quel lieu|où se passe|date de la pièce)(.*))")
piece_ans = ["Quoi ? dans ces lieux champêtres, parmi ces arbres et ces rochers, on trouve des personnes faites comme vous êtes ?", "il a paru en 1682"]

theatre = re.compile("(.*(théâtre|de théâtre|dramaturge|aimez-vous le théâtre|dramaturgie)(.*))")
theatre_ans = ["Ah! le théâtre ! c'est ma vie !", "le poète doit chercher à plaire", "je voudrais bien savoir si la grande règle de toutes les règles n'est pas de plaire, et si une pièce de théâtre qui a attrapé son but n'a pas suivi un bon chemin ?"]

moliere = re.compile("(.*(qui est Molière|connaissez-vous Molière|parlez-moi de Molière|pourquoi Molière|pseudonyme Molière|Molière)(.*))")
moliere_ans = ["Jean-Baptiste Poquelin, dit Molière, comédien et dramaturge!", "Molière, né à Paris en 1622", "Molière a fondé l'Illustre Théâtre", "Molière est l'homme de théâtre complet par excellence !", "Molière est un pseudonyme, peut-être en mémoire du célèbre écrivain François de Molière d’Essertines", "Molière a pratiqué tous les genres de comédie", "Molière s’éteint le 17 février 1673 à l’issue d’une représentation du Malade Imaginaire"]

 
faire = re.compile("(.*(que fais-tu|que faites-vous)(.*))")
faire_ans = ["oh moi ! je converse avec vous ... :\n\
              - je peux vous parler de théâtre ...\n\
              - ou bien nous pourrions simplement discuter de tout et de rien..."]


poeme = re.compile("(.*(rime|poeme|poème|poète|rimes)(.*))")
poeme_ans = [
"L'amour charme\n\
Ceux qu'il desarme;\n\
L'Amour charme,\n\
Cedons luy tous.\n\
Nostre peine Seroit vaine\n\
De vouloir resister à ses coups\n\
Quelque chaîne Qu'un Amant prenne,\n\
La liberté n'a rien qui soit si doux. ",

"Veux-tu que je te dise ? une atteinte secrète \n\
Ne laisse point mon âme en une bonne assiette :  \n\
Oui, quoi qu'à mon amour tu puisses repartir,  \n\
Il craint d'être la dupe, à ne te point mentir : \n\
Qu'en faveur d'un rival ta foi ne se corrompe,  \n\
Ou du moins, qu'avec moi, toi-même on ne te trompe."
]

blague = re.compile("(.*(blague|rire|rigoler|humour)(.*))")
blague_ans = ["Alors que le coiffeur royal demande au roi comment celui-ci souhaite qu'on lui coupe les cheveux, le roi répond: 'En silence'"
          , "Voulant apprendre à son âne de ne pas manger, son propriétaire décide de ne pas le nourrir. Après que l'animal soit mort de faim, l'homme se dit tristement: 'Quelle perte! Il meurt au moment où il avait compris!'"
          , "- Que faisiez vous avant de vous marier ? Demande-t-on un jour à Marcel Jouhandeau.\n\
          - Avant ? Je faisais tout ce que je voulais !"]

fin = re.compile("(.*(aurevoir|bye|bye bye|à plus|au revoir|ciao|salut)(.*))")
fin_ans = ["au revoir", "à bientot", "adieu", "ce fût un réel plaisir de converser avec vous", "portez-vous bien"]



# création de l'interface grace au tkinter
def send(event):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)


    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, f"{your_name}: " + msg + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12 ))
        if re.search(debut,msg.lower()):
        	res = random.choice(debut_ans) + " " + your_name + " !"
        elif re.search(cava,msg.lower()):
          res = random.choice(cava_ans) + " " + your_name + " ?"	
        elif re.search(quiestu,msg.lower()):
          res = random.choice(quiestu_ans)
        elif re.search(qui_don,msg.lower()):
         	res = random.choice(qui_don_ans)
        elif re.search(tue,msg.lower()):
        	res = random.choice(tue_ans)
        elif re.search(piece,msg.lower()):
          	res = random.choice(piece_ans)
        elif re.search(theatre,msg.lower()):
          	res = random.choice(theatre_ans)
        elif re.search(moliere,msg.lower()):
          	res = random.choice(moliere_ans)
        elif re.search(faire,msg.lower()):
        	res = random.choice(faire_ans) + " " + your_name + " ! "
        elif re.search(blague,msg.lower()):
          	res = random.choice(blague_ans)
        elif re.search(poeme,msg.lower()):
        	res = random.choice(poeme_ans)
        elif re.search(fin,msg.lower()):
          	res = random.choice(fin_ans) + " " + your_name + " !"
        else:
           	res = predict(msg, model)
        
        ChatBox.insert(END, f"{bot_name}: " + res + '\n\n')
            
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)

def EntryBoxEraser(event):
  EntryBox.delete("0.0",END)


root = Tk()
root.title("Converser avec Litte_Bot !")
root.geometry("650x550")
root.resizable(width=FALSE, height=FALSE)
root.bind("<Return>", send)

# Creation de la fenetre de chat
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial", wrap="word")
ChatBox.insert(END, "\n\n      LitteBot peu prendre un peu de temps pour vous répondre,\n\t\tmerci de patienter :)\n\n\n")
ChatBox.config(state=DISABLED)

# lier la barre de défillement à la fenetre de chat
scrollbar = Scrollbar(root, command=ChatBox.yview)
ChatBox['yscrollcommand'] = scrollbar.set

#Creation du bouton "envoyer"
SendButton = Button(root, font=("Verdana",12,'bold'), text="Envoyer", width="12", height=5,
                    bd=0, bg="#1FB92B", activebackground="#35DA42",fg='#000000',
                    command= send )
SendButton.bind("<Button-1>", send)

# creation de la fenetre d'écriture des messages
EntryBox = Text(root, bd=0, bg="white",width="29", height="6", font="Arial", wrap="word")
EntryBox.insert(END, " Veuillez écrire ici...")
EntryBox.bind("<Button-1>", EntryBoxEraser)


# emplacement des composant dans la fenetre principale "root"
scrollbar.place(x=576,y=6, height=486)
ChatBox.place(x=6,y=6, height=432, width=560)
EntryBox.place(x=6, y=443, height=50, width=394)
SendButton.place(x=406, y=443, height=50)

root.mainloop()
