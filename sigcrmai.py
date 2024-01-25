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
    bot_instructions = body['instructions']
    instructions = []

    for instruction in bot_instructions:
        instructions.append(instruction['instruccion'])

    company_context = body['companyContext'][0]

    os.environ['OPENAI_API_KEY'] = api_key
    client = OpenAI()

    api_url = main_domain + "/" + api_url + "/" + company_id
    # api_url = "https://sigcrm.pro/api/services/18"
    # print(api_url)
    # return api_url, 200
    api_response = []
    response = requests.get(api_url)

    if response.status_code == 200:
        api_response = response.json()
        api_response = api_response['data']
    else:
        return "Error: ", response.status_code
    
    if len(api_response) == 0:
        response = {
            'answer': 'Lo sentimos. No existen las configuracinones necesatias para generar un agendamiento. Por favor, contacte con el administrador de la empresa.',
            'scheduleId': ''
        }
        return response, 200
    
    api_response_by_services = []

    for employee in api_response:
        for service in employee['services']:
            combined_entry = {**employee, **service}
            combined_entry.pop('services', None)
            api_response_by_services.append(combined_entry)

    df_api_response = pd.DataFrame(api_response_by_services)
    df_api_response(['employeeId', 'positionId'], axis=1, inplace=True)

    positions = df_api_response['positionName'].unique().tolist()
    services = df_api_response['serviceName'].unique().tolist()
  
    # # create the 'concat_feature' column
    df_api_response['concat_feature'] = "Nombre del Trabajador: " + df_api_response['employeeNames']  + "|" + "Apellidos del Trabajador: " + df_api_response['employeeLastNames'] + "|" +  "Nombre del Cargo del Trabajador: " + df_api_response['positionName'] + ' | ' + "Descripción del Cargo del Trabajador: " + df_api_response['positionDescription'] + "|" + "Día de la semana del horario de atención del trabajador: " + df_api_response['day'] +  "|" + "Hora de inicio del horario de atención del trabajador: " + df_api_response['startTime'] + "|" + "Hora de finalización del horario de atención del trabajador: " + df_api_response['endTime'] + "|" + "Servicio del trabajador: " + df_api_response['serviceName']

    # Function to process embeddings in batches
    def generate_embeddings_in_batches(texts, batch_size=100):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings_batch = client.embeddings.create(model='text-embedding-ada-002', input=batch, encoding_format='float').data
            all_embeddings.extend([eb.embedding for eb in embeddings_batch])
        return all_embeddings

    # Apply batch processing for embeddings
    df_api_response['embedding'] = generate_embeddings_in_batches(df_api_response['concat_feature'].tolist())

    def get_df_similares(question, df):
        # Generate embedding for the question
        question_embedding = client.embeddings.create(model="text-embedding-ada-002", input=question, encoding_format="float").data[0].embedding
        question_embedding_arr = np.array(question_embedding).reshape(1, -1)

        # Vectorize embeddings in the DataFrame
        df_embeddings = np.vstack(df['embedding'].values)

        # Calculate similarities using vectorized operations
        df['similarities'] = cosine_similarity(df_embeddings, question_embedding_arr).flatten()

        return df.sort_values("similarities", ascending=False)


    def get_response(question, df_similars, instructions, company_context, positions, services):
        client = OpenAI()
        joined_instructions = "\n".join(instructions)
        joined_positions = "\n".join(positions)
        joined_services = "\n".join(services)

        bot_messages = [
          {"role": "system", "content": f"""
           CONTEXTO:
           Asistente ayuda a los usuarios a realizar el agendamiento de un turno para ser atendido con un trabajador
           Asistente es un chatbot virtual amable encargado de brindar información de los cargos, servicios y horarios de atención de los trabajadores en la empresa {company_context['companyName']}.

           SERVICIO DE LA EMPRESA:
           {company_context['companyActivity']}

           DESCRIPCIÓN DE LA EMPRESA:
           {company_context['companyDescription']}
               
            INSTRUCCIONES:
           {joined_instructions}

            HORARIO DE AGENDAMIENTO DISPONIBLE:
            - NOMBRE DEL TRABAJADOR: {df_similars.iloc[0]['employeeNames'] + " " + df_similars.iloc[0]['employeeLastNames']}
            - HORARIO DE INICIO DEL TRABAJADOR: {df_similars.iloc[0]['startTime']}
            - HORARIO DE FINALIZACIÓN DEL TRABAJADOR: {df_similars.iloc[0]['endTime']}
            - DÍA DE LA SEMANA DEL HORARIO DEL TRABAJADOR: {df_similars.iloc[0]['day']}

            CARGO MÁS COINCIDENTE:
            - NOMBRE DEL CARGO: {df_similars.iloc[0]['positionName']}
            - DESCRIPCIÓN: {df_similars.iloc[0]['positionDescription']}

            SERVICIO MÁS COINDIDENTE:
            - NOMBRE DEL SERVICIO: {df_similars.iloc[0]['serviceName']}
            - DESCRIPCIÓN: {df_similars.iloc[0]['serviceDescription']}
            - COSTO DEL SERVICIO EN DÓLARES: {df_similars.iloc[0]['serviceCost']}

            CARGOS DISPONIBLES:
            {joined_positions}

            SERVICIOS DISPONIBLES:
            {joined_services}
        """}]

        for message in conversation_history:
            bot_messages.append(message)
        
        bot_messages.append({"role": "user", "content": question})

        completion = client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = bot_messages,
        )
        return completion.choices[0].message.content
    
    df_similars = get_df_similares(question, df_api_response)
    answer = get_response(question, df_similars, instructions, company_context, positions, services)

    schedule_id = int(df_similars.iloc[0]['scheduleId'])
    service_id = int(df_similars.iloc[0]['serviceId'])
    response = {
        'answer': answer,
        'scheduleId': schedule_id,
        'serviceId': service_id
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
    
    api_response = []
    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
        api_response = data['data']
    else:
        return "Error: ", response.status_code
    
    # create the 'concat_feature' column
    df_concat_feature = pd.DataFrame(api_response)
    df_concat_feature['concat_feature'] = "Cargo: " + df_concat_feature['positionName'] + ' | ' + "Descripción: " + df_concat_feature['positionDescription']
    # create the 'embedding' column
    df_concat_feature['embedding'] = df_concat_feature['concat_feature'].apply(lambda x: client.embeddings.create(model='text-embedding-ada-002', input=x, encoding_format='float').data[0].embedding)

    df_contat_feature_json = df_concat_feature.to_dict(orient='records')

    return df_contat_feature_json, 200


if __name__ == "_main_":
    app.run(host='0.0.0.0', port=5000)