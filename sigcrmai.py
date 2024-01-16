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
    
    question = body['question']
    companyId = body['companyId']
    companyId = str(companyId)
    ngrok = body['ngrok']

    os.environ['OPENAI_API_KEY'] = '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'
    client = OpenAI()

    api_url = ngrok + "/api/services/" + companyId
    response_list = []
    response = requests.get(api_url)

    if response.status_code == 200:
        data = response.json()
        response_list = data['data']
    else:
        return "Error: ", response.status_code
    
    # create the 'concat_feature' column
    df_concat_feature = pd.DataFrame(response_list)
    df_concat_feature['concat_feature'] = "Cargo: " + df_concat_feature['positionName'] + ' | ' + "Descripción: " + df_concat_feature['positionDescription']

    # create the 'embedding' column
    df_concat_feature['embedding'] = df_concat_feature['concat_feature'].apply(lambda x: client.embeddings.create(model='text-embedding-ada-002', input=x, encoding_format='float').data[0].embedding)


    df_embeddings = df_concat_feature[['positionName', 'positionDescription', 'concat_feature', 'embedding' ]].copy()

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
    
    df_similars = get_df_similares(question, df_embeddings)
    prompt = get_promtp(question, df_similars)
    answer = get_response(prompt)

    response = {
        'answer': answer,
        'positionName': df_similars.iloc[0]['positionName'],
        'positionDescription': df_similars.iloc[0]['positionDescription']
    }

    return response, 200


if __name__ == "_main_":
    app.run(host='0.0.0.0', port=5000)