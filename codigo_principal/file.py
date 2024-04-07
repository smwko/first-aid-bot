import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from gtts import gTTS
import joblib

nltk.download('omw')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import numpy as np
import pandas as pd

from googletrans import Translator
from gensim.models import KeyedVectors

import speech_recognition as sr
from pydub import AudioSegment

import telebot
from telebot import types
import json

def ogg_to_wav(input_file, output_file):
    audio = AudioSegment.from_ogg(input_file)
    audio.export(output_file, format="wav")

def traducir(palabra):
    translator = Translator()
    translation = translator.translate(palabra, src='es', dest='en')
    return translation.text

def encontrar_relacion(palabra1, palabra2):
    global model
    if palabra1 == palabra2:
        return 1
    # Comprobar si las palabras están en el vocabulario del modelo
    if palabra1 not in model.index_to_key or palabra2 not in model.index_to_key:
        relaciones = []
        for syn1 in wn.synsets(palabra1):
            for syn2 in wn.synsets(palabra2):
                similitud = syn1.path_similarity(syn2)
                if similitud is not None:
                    relaciones.append((syn1, syn2, similitud))
        if relaciones:
            df = pd.DataFrame(relaciones, columns=['syn1', 'syn2', 'similitud'])
            return (df['similitud'].max()+df['similitud'].mean())/2
        else:
            return 0
    
    return model.similarity(palabra1, palabra2)

def procesar_entrada(texto):
    tokens = word_tokenize(texto)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    return tokens

with open('dicc.json', 'r', encoding='utf-8') as file:
    dicc = json.load(file)

with open('dicc_responses.json', 'r', encoding='utf-8') as file:
    actua = json.load(file)

with open('dicc_traduccion.json', 'r', encoding='utf-8') as file:
    traduccion = json.load(file)
for x in list(traduccion.keys()):
    traduccion[traduccion[x]] = x

model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

rfm = joblib.load('modelo_random_forest.pkl')

# Función para crear predicción de una frase
def scan(frase):
    global dicc,rfm, actua
    tokens = list(set(procesar_entrada(frase)))
    print(tokens)

    claves,result = [x for x in dicc],[]
    columns_df = ["key","vector","model","comb"]

    for key in claves:
        dataset = procesar_entrada(key)+dicc[key]

        lista = [max([encontrar_relacion(x, y) for y in tokens]) for x in dataset]
        mean = np.mean(lista)
        n = (mean+np.max(lista))*mean
        lista += [mean for _ in range(11-len(lista))]

        prob = rfm.predict_proba([lista])[0][1]

        result.append([key,n,prob,prob+n])

    df = pd.DataFrame(result,columns=columns_df)
    multi = 1/sum(df['comb'])
    df['comb'] = df['comb'].apply(lambda x: x*multi)
    df.sort_values(by="comb",inplace=True,ascending=False)

    threshold = 1.2
    if df['vector'].sum()<threshold:
        return []
    
    valor_max_comb = df['comb'].max()
    valor_max_vector = df['vector'].max()
    return list(df[(df['comb']>valor_max_comb*0.8) | (df['vector']>valor_max_vector*0.8)]['key'].values)

bot = telebot.TeleBot('6716509249:AAG-RKkIaMoxvVNLQ6OXsCUaaiOLQFpIXVE')
print("Bot iniciado...")

def response(prediccion, message):
    global actua, traduccion, bot
    
    if len(prediccion) in [0, 1, 2]:
        if len(prediccion) == 0:
            actuacion =  "Lo siento, no he podido entender lo que has dicho, ¿podrías ser más preciso?"
        elif len(prediccion) == 1:
            actuacion = f"{traduccion[prediccion[0]].upper()}\n\n{actua[prediccion[0]]}"
        else:
            actuacion = f"{traduccion[prediccion[0]].upper()}\n\n{actua[prediccion[0]]}\n\n\n{traduccion[prediccion[1]].upper()}\n\n{actua[prediccion[1]]}"
        prediccion = ["otro"]
    else:
        prediccion.append("otro")
        actuacion = "Marca la situación más factible:"
    tts = gTTS(text=actuacion, lang='es')
    tts.save("saludo.mp3")
    # Respondemos al usuario
    #bot.reply_to(message, actuacion)
    audio = open('saludo.mp3', 'rb')
    keyboard = crear_teclado([traduccion[x] for x in prediccion],prediccion)
    bot.send_message(message.chat.id, actuacion, reply_markup=keyboard)
    bot.send_voice(message.chat.id, audio)
    audio.close()

@bot.message_handler(func=lambda msg: True)
def echo_all(message):
    if message.text == "/start":
        bot.send_message(message.chat.id, "¡Hola! Soy un bot que te ayudará en situaciones de primeros auxilios.\nCabe destacar que este chat no será nunca sustitutivo de los servicios médicos profesionales.\n¿En qué puedo ayudarte?")
    else:
        prediccion = scan(traducir(message.text))
        print(prediccion)
        response(prediccion, message)

def crear_teclado(textos,opciones):
    keyboard = types.InlineKeyboardMarkup()
    for texto,opcion in zip(textos,opciones):
        button = types.InlineKeyboardButton(text=texto.upper(), callback_data=opcion)
        keyboard.add(button)
    return keyboard


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    # Descargar el archivo de voz
    voice = message.voice
    file_id = voice.file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    # Creamos archivo wav
    with open("audio.ogg", 'wb') as new_file:
        new_file.write(downloaded_file)
    ogg_to_wav("audio.ogg", "audio.wav")
    # Analizamos el audio
    reconocedor = sr.Recognizer()
    with sr.AudioFile("audio.wav") as audio:
        audio_data = reconocedor.record(audio)
    texto_transcrito = reconocedor.recognize_google(audio_data, language='es-ES')
    # Traducimos el audio
    frase = traducir(texto_transcrito)
    # Procesamos la frase para obtener la predicción
    prediccion = scan(frase)
    print(prediccion)
    response(prediccion, message)
        

@bot.callback_query_handler(func=lambda call: True)
def callback_handler(call):
    # Obtenemos la opción seleccionada por el usuario
    opcion_seleccionada = call.data

    # ID del chat donde se recibió el mensaje original
    chat_id = call.message.chat.id

    if opcion_seleccionada == "otro":
        keyboard = crear_teclado([traduccion[x] for x in list(actua.keys())],list(actua.keys()))
        bot.send_message(chat_id, "Marca la situación más factible:", reply_markup=keyboard)
    else:
        actuacion = actua[opcion_seleccionada]
        bot.send_message(chat_id, f"{traduccion[opcion_seleccionada].upper()}\n\n{actuacion}")
        tts = gTTS(text=actuacion, lang='es')
        tts.save("saludo.mp3")
        # Respondemos al usuario
        audio = open('saludo.mp3', 'rb')
        bot.send_voice(chat_id, audio)
        audio.close()
    
     
bot.infinity_polling()