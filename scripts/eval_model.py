import requests
import yaml
import sys
import fire
import os
import json
import time
import pandas as pd
import random
import ast
import numpy as np

def try_to_float(x):
    try:
        return float(x)
    except:
        return np.nan

def generate_unique_filename(output_path, task, extension='.csv'):
    base_name = os.path.join(output_path, task)
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

def preprocess_result(settings, model_path, model_checkpoint, output_path, metric, metric_filter):

    delete_result_after_preprocess = settings['delete_result_after_preprocess']
    categoric_column = settings['categoric_column']
    output_path_save = settings['output_path']
    

    model_path = model_path.replace('/','__')
    all_files = os.listdir(output_path + model_path)
    result_file = [file for file in all_files if 'results' == file.split('_')[0]][0]
    others_files = [file for file in all_files if 'results' != file.split('_')[0]]

    for file in others_files:
        with open(output_path + model_path + '/' + file, 'r') as f:
            lines = f.readlines()
            lines = [json.loads(line) for line in lines]
            df = pd.DataFrame(lines)
            df_clean = df[['target', 'filtered_resps', metric]]
            df_clean.loc[:, 'question'] = df['doc'].apply(lambda x: x['question'])
            df_clean.loc[:, 'model_checkpoint'] = model_checkpoint  
            for category in categoric_column:
                if category in df['doc'].iloc[0].keys():
                    df_clean.loc[:, 'category'] = df['doc'].apply(lambda x: x[category])
            
            os.makedirs(output_path_save + 'temp_df', exist_ok=True)
            df_clean.to_csv(output_path_save + 'temp_df/' + str(random.randint(10000, 999999)) + '.csv', index=False)

    
    with open(output_path + model_path + '/' + result_file, 'r') as f:
        results = json.load(f)
        json_dict = []
        for task in results['results']:
            json_dict.append({'name': task, metric: results['results'][task].get(f"{metric},{metric_filter}", None), f'{metric}_std': results['results'][task].get(f'{metric}_stderr,{metric_filter}', None)})
        
        df = pd.DataFrame(json_dict)
        df['category'] = df['category'].map('{:.4f}'.format) + ' +- ' + df[f'{metric}_std'].map('{:.4f}'.format)
        df = df.set_index('name')['category'].to_frame().T.reset_index(drop=True)
        df.index.name = None
        df['model_checkpoint'] = model_checkpoint
        os.makedirs(output_path_save + 'temp', exist_ok=True)
        df.to_csv(output_path_save + 'temp/' + str(random.randint(10000, 999999)) + '.csv', index=False)

    if delete_result_after_preprocess:
        os.system(f'rm -r {output_path + model_path}')

def postprocess_result(df):
    target_type = df['target'].apply(lambda x: type(x)).iloc[0]
    
    unique_category = df['category'].unique()
    unique_model_checkpoint = df['model_checkpoint'].unique()

    json_lines = []
    for category in unique_category:
        for model in unique_model_checkpoint:
            df_temp = df[(df['category'] == category) & (df['model_checkpoint'] == model)]
            if target_type == float:
                df_temp.loc[:, 'filtered_resps'] = df_temp['filtered_resps'].apply(lambda x: try_to_float(ast.literal_eval(x)[0]))
                df_temp.loc[:, 'diff'] = df_temp['target'] - df_temp['filtered_resps']
                mean_absolute_error = df_temp['diff'].apply(lambda x: abs(x)).mean()
                mean_squared_error = df_temp['diff'].apply(lambda x: x**2).mean()
                mean_absolute_error_std = df_temp['diff'].apply(lambda x: abs(x)).std()
                mean_squared_error_std = df_temp['diff'].apply(lambda x: x**2).std()
                accuracy = df_temp['exact_match'].mean()
                json_lines.append({'category': category, 'model_checkpoint': model, 'mean_absolute_error': mean_absolute_error, 'mean_squared_error': mean_squared_error, 'mean_absolute_error_std': mean_absolute_error_std, 'mean_squared_error_std': mean_squared_error_std, 'accuracy': accuracy})
            elif target_type == str:
                accuracy = df_temp['exact_match'].mean()
                json_lines.append({'category': category, 'model_checkpoint': model, 'accuracy': accuracy})

    return pd.DataFrame(json_lines)

def eval_model(settings_path, checkpoint, task, metric, metric_filter):
    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)['lm_eval_settings']
        autorization_token = settings['autorization_token']
        url = settings['url']
        model_mode = settings['model']
        model_name = settings['model_name']
        output_path = settings['output_path'] + task + '/'
        num_concurrent = settings['num_concurrent']
        max_retries = settings['max_retries']
        tokenized_requests = settings['tokenized_requests']
        drop_model_bool = False
        

    # CHECK IF MODEL EXISTS AND IS IN PRODUCTION
    payload = {}
    headers = {'User-Agent': 'insomnia/9.3.3', 'Authorization': f'Bearer {autorization_token}'}

    #all_models = requests.request("GET", url, headers=headers, data=payload).json()
    all_models = get_all_models(url, headers)
    for model in all_models.get('items', []):
        if checkpoint == model.get('checkpoint_path'):
            
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
        payload = {"name": model_name, "checkpoint_path": checkpoint}
        headers = {'Content-Type': 'application/json', 'User-Agent': 'insomnia/9.3.3', 'Authorization': f'Bearer {autorization_token}'}
        model_id = requests.post(url, headers=headers, json=payload).json()['id']

        # DEPLOY MODEL
        print("Iniciando deploy do modelo...")
        deploy_model(url, headers, model_id)
        drop_model_bool = True

    # EVALUATE MODEL
    for attempt in range(5):
        time.sleep(20)
        all_models = get_all_models(url, headers)
        model_found = False
        for model in all_models['items']:
            if checkpoint == model['checkpoint_path'] and model['deploy']['status']['status'] == 'DEPLOYED':
                print("Iniciando avaliação...")
                model_path = model['deploy']['path']
                base_url = model['deploy']['url'] + '/v1/completions'
                openai_api_key = model['deploy']['token']

                os.system(f'OPENAI_API_KEY={openai_api_key} lm_eval --model {model_mode} --tasks {task} --model_args model={model_path},base_url={base_url},num_concurrent={num_concurrent},max_retries={max_retries},tokenized_requests={tokenized_requests} --output_path {output_path} --log_samples')
                model_found = True
                break
        if not model_found:
            print(f"Status do deploy é '{model['deploy']['status']['status']}'. Aguardando o modelo ser implantado... (tentativa {attempt +1})")
        else:
            print("Avaliação concluída.")
            break
    else:
        print("Falha ao implantar o modelo dentro do tempo esperado.")
        return None

    # DELETING MODEL
    if drop_model_bool:
        print("Iniciando remoção do modelo...")
        drop_model(url, headers, model_id)

    preprocess_result(settings, model_path, checkpoint, output_path, metric, metric_filter)

def main(settings_path):
    pd.options.mode.chained_assignment = None 

    with open(settings_path, 'r') as stream:
        settings = yaml.safe_load(stream)['lm_eval_settings']
        model_checkpoint = settings['model_checkpoint']
        tasks = settings['tasks']
        metrics = settings['metrics']
        metrics_filter = settings['metrics_filter']
    
    for index in range(len(tasks)):
        task = tasks[index]
        metric = metrics[index]
        metric_filter = metrics_filter[index]
        for checkpoint in model_checkpoint:
            eval_model(settings_path, checkpoint, task, metric, metric_filter)
    

        all_files = os.listdir(settings['output_path'] + 'temp')
        df_concat = pd.concat([pd.read_csv(settings['output_path'] + 'temp/' + file) for file in all_files])
        all_files_raw = os.listdir(settings['output_path'] + 'temp_df')
        df_concat_raw = pd.concat([pd.read_csv(settings['output_path'] + 'temp_df/' + file) for file in all_files_raw])
        df_concat_raw = df_concat_raw.rename({'flux': 'category'}, axis=1)

        df_metrics = postprocess_result(df_concat_raw)

        name_file = generate_unique_filename(settings['output_path'], task)
        df_concat.to_csv(name_file, index=False)
        df_concat_raw.to_csv(name_file.replace('.csv', '_raw.csv'), index=False)
        df_metrics.to_csv(name_file.replace('.csv', '_metrics.csv'), index=False)


        os.system(f'rm -r {settings["output_path"] + "temp"}')
        os.system(f'rm -r {settings["output_path"] + "temp_df"}')

    
if __name__ == '__main__':
    fire.Fire(main)



# python scripts/eval_model.py --settings_path scripts/settings.yaml

















# OPENAI_API_KEY=2902dbce-0c66-4ca2-9df3-35d0aeb9ca61 lm_eval --model local-completions --tasks minhas_vantagens_human --model_args model=/cortex/inference-orchestrator/models/bd8e5dd9-4b0a-434f-91fd-91f2f674177b,base_url=http://149.130.218.35:8102/v1/completions,num_concurrent=20,max_retries=3,tokenized_requests=False --output_path results --log_samples



# docker inspect -f '{{json .Args}}' 5c1e