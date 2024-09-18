import requests
import yaml
import sys
import fire
import os
import json
import time
import pandas as pd
import random

def generate_unique_filename(output_path, model_name, extension='.csv'):
    base_name = os.path.join(output_path, model_name)
    name_file = base_name + extension
    counter = 1
    while os.path.exists(name_file):
        name_file = f"{base_name}_{counter}{extension}"
        counter += 1
    return name_file

def get_all_models(url, headers):
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def deploy_model(url, headers, model_id):
    deploy_url = f'{url}/{model_id}/deploys'
    response = requests.post(deploy_url, headers=headers)
    response.raise_for_status()

def drop_model(url, headers, model_id):
    drop_url = f'{url}/{model_id}/deploys'
    response = requests.delete(drop_url, headers=headers)
    response.raise_for_status()

def preprocess_result(output_path, model_path, model_checkpoint, delete_result_after_preprocess):
    model_path = model_path.replace('/','__')
    all_files = os.listdir(output_path + model_path)
    result_file = [file for file in all_files if 'results' == file.split('_')[0]][0]

    
    with open(output_path + model_path + '/' + result_file, 'r') as f:
        results = json.load(f)
        json_dict = []
        for task in results['results']:
            json_dict.append({'name': task, 'acc': results['results'][task].get("acc,none", None), 'acc_std': results['results'][task].get('acc_stderr,none', None)})
        
        df = pd.DataFrame(json_dict)
        df['acc'] = df['acc'].map('{:.4f}'.format) + ' +- ' + df['acc_std'].map('{:.4f}'.format)
        df = df.set_index('name')['acc'].to_frame().T.reset_index(drop=True)
        df.index.name = None
        df['model_checkpoint'] = model_checkpoint
        os.makedirs(output_path + 'temp', exist_ok=True)
        df.to_csv(output_path + 'temp/' + str(random.randint(10000, 999999)) + '.csv', index=False)

    if delete_result_after_preprocess:
        os.system(f'rm -r {output_path + model_path}')

def eval_model(settings_path, index=None):
    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)['lm_eval_settings']
        autorization_token = settings['autorization_token']
        url = settings['url']
        model_checkpoint = settings['model_checkpoint']
        if index is not None:
            model_checkpoint = model_checkpoint[index]
        model_mode = settings['model']
        tasks = settings['tasks']
        model_name = settings['model_name']
        output_path = settings['output_path']
        num_concurrent = settings['num_concurrent']
        max_retries = settings['max_retries']
        tokenized_requests = settings['tokenized_requests']
        drop_model_bool = False
        delete_result_after_preprocess = settings['delete_result_after_preprocess']

    # CHECK IF MODEL EXISTS AND IS IN PRODUCTION
    payload = {}
    headers = {'User-Agent': 'insomnia/9.3.3', 'Authorization': f'Bearer {autorization_token}'}

    #all_models = requests.request("GET", url, headers=headers, data=payload).json()
    all_models = get_all_models(url, headers)
    for model in all_models.get('items', []):
        if model_checkpoint == model.get('checkpoint_path'):
            
            deploy_status = model.get('deploy', {}).get('status', {}).get('status')
            if deploy_status == 'DEPLOYED':
                print("O modelo já existe e está em produção. Iniciando avaliação...")
                model_path = model['deploy']['path']
                base_url = model['deploy']['url'] + '/v1/completions'
                openai_api_key = model['deploy']['token']
                model_id = model['id']
                break
            elif deploy_status == 'DROPPED':
                print("O modelo já existe, mas não está em produção. Iniciando deploy do modelo...")
                model_id = model['id']
                deploy_model(url, headers, model_id)
                drop_model_bool = True
                break
    else:
        # UPLOAD MODEL
        print("O modelo não existe. Iniciando upload do modelo...")
        payload = {"name": model_name, "checkpoint_path": model_checkpoint}
        headers = {'Content-Type': 'application/json', 'User-Agent': 'insomnia/9.3.3', 'Authorization': f'Bearer {autorization_token}'}
        model_id = requests.post(url, headers=headers, json=payload).json()['id']

        # DEPLOY MODEL
        print("Iniciando deploy do modelo...")
        deploy_model(url, headers, model_id)
        drop_model_bool = True

    # EVALUATE MODEL
    for attempt in range(5):
        time.sleep(10)
        all_models = get_all_models(url, headers)
        model_found = False
        for model in all_models['items']:
            if model_checkpoint == model['checkpoint_path'] and model['deploy']['status']['status'] == 'DEPLOYED':
                print("Iniciando avaliação...")
                model_path = model['deploy']['path']
                base_url = model['deploy']['url'] + '/v1/completions'
                openai_api_key = model['deploy']['token']

                os.system(f'OPENAI_API_KEY={openai_api_key} lm_eval --model {model_mode} --tasks {tasks} --model_args model={model_path},base_url={base_url},num_concurrent={num_concurrent},max_retries={max_retries},tokenized_requests={tokenized_requests} --output_path {output_path} --log_samples')
                model_found = True
                break
        if not model_found:
            print(f"Status do deploy é '{model['deploy']['status']['status']}'. Aguardando o modelo ser implantado... (tentativa {attempt +1})")
        else:
            print("Avaliação concluída.")
            break
    else:
        print("Falha ao implantar o modelo dentro do tempo esperado.")

    # DELETING MODEL
    if drop_model_bool:
        print("Iniciando remoção do modelo...")
        drop_model(url, headers, model_id)

    preprocess_result(output_path, model_path, model_checkpoint, delete_result_after_preprocess)

def main(settings_path):

    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)['lm_eval_settings']
        model_checkpoint = settings['model_checkpoint']
    
    if isinstance(model_checkpoint, list):
        for index in range(len(model_checkpoint)):
            eval_model(settings_path, index=index)
    else:
        eval_model(settings_path)

    all_files = os.listdir(settings['output_path'] + 'temp')
    df_concat = []
    for file in all_files:
        df = pd.read_csv(settings['output_path'] + 'temp/' + file)
        df_concat.append(df)

    df_concat = pd.concat(df_concat)
    name_file = generate_unique_filename(settings['output_path'], settings['model_name'])
    df_concat.to_csv(name_file, index=False)

    os.system(f'rm -r {settings["output_path"] + "temp"}')

    
if __name__ == '__main__':
    fire.Fire(main)

# python scripts/eval_model.py --settings_path scripts/settings.yaml

















# OPENAI_API_KEY=2902dbce-0c66-4ca2-9df3-35d0aeb9ca61 lm_eval --model local-completions --tasks minhas_vantagens_human --model_args model=/cortex/inference-orchestrator/models/bd8e5dd9-4b0a-434f-91fd-91f2f674177b,base_url=http://149.130.218.35:8102/v1/completions,num_concurrent=20,max_retries=3,tokenized_requests=False --output_path results --log_samples



# docker inspect -f '{{json .Args}}' 5c1e