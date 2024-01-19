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
    return "Hello, Worldfdsafs!"

@app.route('/api/openai', methods=['POST'])
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
    
    # create the 'concat_feature' column
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

    # Rest of your functions remain the same
    def get_prompt(question, df_similars):
      prompt = f"""
                CONTEXTO:

                Soy un asistente virtual de una empresa encargada en ofrecer los servicios de expertos y expertas en desarrollo de sofware, desarrollo de aplicaciones web y desarrollo de aplicaciones móviles.
                Ayudo a las personas a encontrar el cargo que mejor cumpla con las habilidades y conocimientos solicitados. Un cliente me ha hecho la siguiente pregunta:

                PERSONALIDAD:

                Útil, alegre, experto en recomendaciones

                CARGO DISPONIBLE:

                - Nombre del cargo: {df_similars.iloc[0]['positionName']}
                - Descripción: {df_similars.iloc[0]['positionDescription']}

                INSTRUCCIÓN:
                - Solo responde en base al CARGO DISPONIBLE
                - Si no contamos con un cargo adecuado a las necesidades el cliente, no inventes nada

                CONVERSACIÓN:

                Cliente: {question}
                Bot:"""
      return prompt

    def get_response(prompt):
      client = OpenAI()

      completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=[
              {"role": "system", "content": "Tú eres un asistente útil"},
              {"role": "user", "content": prompt}
          ]
      )
      return completion.choices[0].message.content
    
    df_similars = get_df_similares(question, df_positions)
    prompt = get_prompt(question, df_similars)
    answer = get_response(prompt)

    response = {
        'answer': answer,
        'positionName': df_similars.iloc[0]['positionName'],
        'positionDescription': df_similars.iloc[0]['positionDescription']
    }

    return response, 200


@app.route('/api/answer-question', methods=['POST'])
def answer_question():

    
    body = request.get_json()
    question = body['question']
    api_key = body['apiKey']
    
    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()

    # get the embeddings of the CRM db
    full_url = "https://api.sigcrmai.com/api/v1/position"
    positions = []
    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
        positions = data['data']
    else:
        return "Error: ", response.status_code

    def get_df_similares(question, df_embeddings):
        question_embedding = client.embeddings.create(model="text-embedding-ada-002",
                                                    input=question,
                                                    encoding_format="float").data[0].embedding


        question_embedding_arr = np.array(question_embedding).reshape(1, -1)
        df_embeddings['embedding'] = df_embeddings['embedding'].apply(lambda x: np.array(x).reshape(1, -1))

        # Buscamos los productos más similares a la pregunta
        # se genera un cosine_similarity (score) a todos los productos respecto a la pregunta realizada anteriormente
        df_embeddings["similarities"] = df_embeddings['embedding'].apply(lambda x: cosine_similarity(x, question_embedding_arr))
        df_embeddings = df_embeddings.sort_values("similarities", ascending=False)

        return df_embeddings
    
    # crear el prompt
    def get_promtp(question, df_similars):
        prompt = f"""
                    CONTEXTO:

                    Soy un asistente virtual de una empresa encargada en ofrecer los servicios de expertos y expertas en desarrollo de sofware, desarrollo de aplicaciones web y desarrollo de aplicaciones móviles.
                    Ayudo a las personas a encontrar el cargo que mejor cumpla con las habilidades y conocimientos solicitados. Un cliente me ha hecho la siguiente pregunta:

                    PERSONALIDAD:

                    Útil, alegre, experto en recomendaciones

                    CARGO DISPONIBLE:

                    - Nombre del cargo: {df_similars.iloc[0]['positionName']}
                    - Descripción: {df_similars.iloc[0]['positionDescription']}

                    INSTRUCCIÓN:
                    - Solo responde en base al CARGO DISPONIBLE
                    - Si no contamos con un cargo adecuado a las necesidades el cliente, no inventes nada

                    CONVERSACIÓN:

                    Cliente: {question}
                    Bot:"""
        return prompt




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