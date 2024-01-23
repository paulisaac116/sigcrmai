from flask import Flask, request
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import os

app = Flask(__name__)

@app.route('/index')
def index():
    return "Hello, You Human!"

@app.route('/api/answer-question', methods=['POST'])
def openai():

    body = request.get_json()

    if 'question' not in body:
        return 'No question provided', 400
    
    if 'companyId' not in body:
        return 'No companyId provided', 400
    
    if 'apiKey' not in body:
        return 'No apiKey provided', 400
    
    if 'mainDomain' not in body:
        return 'No mainDomain provided', 400
    
    if 'apiUrl' not in body:
        return 'No apiUrl provided', 400
    
    question = body['question']
    company_id = body['companyId']
    company_id = str(company_id)
    api_key = body['apiKey']
    main_domain = body['mainDomain']
    api_url = body['apiUrl']
    conversation_history = body['chatHistory']

    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()

    api_url = main_domain + "/" + api_url + "/" + company_id
    positions = []
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        positions = data['data']
    else:
        return "Error: ", response.status_code
    
    # # create the 'concat_feature' column
    df_positions = pd.DataFrame(positions)
    df_positions['concat_feature'] = "Nombre de la Posición: " + df_positions['positionName'] + ' | ' + "Descripción: " + df_positions['positionDescription']

    # Function to process embeddings in batches
    def generate_embeddings_in_batches(texts, batch_size=100):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_batch = client.embeddings.create(model='text-embedding-ada-002', input=batch, encoding_format='float').data
            all_embeddings.extend([eb.embedding for eb in embeddings_batch])
        return all_embeddings

    # Apply batch processing for embeddings
    df_positions['embedding'] = generate_embeddings_in_batches(df_positions['concat_feature'].tolist())

    def get_df_similares(question, df):
        # Generate embedding for the question
        question_embedding = client.embeddings.create(model="text-embedding-ada-002", input=question, encoding_format="float").data[0].embedding
        question_embedding_arr = np.array(question_embedding).reshape(1, -1)

        # Vectorize embeddings in the DataFrame
        df_embeddings = np.vstack(df['embedding'].values)

        # Calculate similarities using vectorized operations
        df['similarities'] = cosine_similarity(df_embeddings, question_embedding_arr).flatten()

        return df.sort_values("similarities", ascending=False)

    def get_response(question, df_similars):
        client = OpenAI()

        bot_messages = [
          {"role": "system", "content": f"""Asistente es un chatbot virtual amable de una empresa encargada en ofrecer los servicios de expertos y expertas en desarrollo de sofware, desarrollo de aplicaciones web y desarrollo de aplicaciones móviles. Asistente ayuda a los usuarios a brindar información sobre los servicios o cargos de la empresa, y a agendar un turno para ser atendido con un trabajador.     
               
               Contexto:
               - El usuario hace una pregunta sobre los servicios de la empresa o solicitando información sobre un cargo disponible
               - El usuario puede preguntar por un cargo disponible o por un horario de atención
               - El usuario puede solicitar realizar el agendamiento de un turno en base al HORARIO DE AGENDAMIENTO DISPONIBLE de un trabajador
               
               INSTRUCCIONES:
                - Si no existe un CARGO DEL TRABAJADOR que coincida con los cargos disponibles, no inventar
                - Si el usuario pregunta por un CARGO DEL TRABAJADOR que si existe responder con la información de HORARIO DE AGENDAMIENTO DISPONIBLE
                - Si el usuario pregunta por un horario de atención, por una fecha u hora primero preguntar para qué cargo desea obtener información
                - Si no existe un HORARIO DE AGENDAMIENTO DISPONIBLE para el CARGO DEL TRABAJADOR, responder que no existe
                - Si el usuario no solicita información, preguntar en qué necesita ayuda sobre los servicios de la empresa 
        """}]

        for message in conversation_history:
            bot_messages.append(message)
        
        bot_messages.append({"role": "user", "content": question})

        completion = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = bot_messages,
        )
        return completion.choices[0].message.content
    
    df_similars = get_df_similares(question, df_positions)
    answer = get_response(question, df_similars)
    # answer = get_response(question, [])

    response = {
        'answer': answer,
        'positionName': df_similars.iloc[0]['positionName'],
        'positionDescription': df_similars.iloc[0]['positionDescription']
    }

    return response, 200

@app.route('/api/create-embedding', methods=['POST'])
def create_embedding():
    
    body = request.get_json()

    if 'companyId' not in body:
        return 'No companyId provided', 400
    
    if 'apiKey' not in body:
        return 'No apiKey provided', 400
    
    if 'mainDomain' not in body:
        return 'No mainDomain provided', 400
    
    if 'apiUrl' not in body:
        return 'No apiUrl provided', 400

    company_id = str(body['companyId'])
    api_key = body['apiKey']
    main_domain = body['mainDomain']
    api_url = body['apiUrl']
    full_url = main_domain + "/" + api_url + "/" + company_id

    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()
    
    positions = []
    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
        positions = data['data']
    else:
        return "Error: ", response.status_code
    
    # create the 'concat_feature' column
    df_concat_feature = pd.DataFrame(positions)
    df_concat_feature['concat_feature'] = "Cargo: " + df_concat_feature['positionName'] + ' | ' + "Descripción: " + df_concat_feature['positionDescription']
    # create the 'embedding' column
    df_concat_feature['embedding'] = df_concat_feature['concat_feature'].apply(lambda x: client.embeddings.create(model='text-embedding-ada-002', input=x, encoding_format='float').data[0].embedding)

    df_contat_feature_json = df_concat_feature.to_dict(orient='records')

    return df_contat_feature_json, 200


if __name__ == "_main_":
    app.run(host='0.0.0.0', port=5000)