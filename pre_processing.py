import pandas as pd
import os
import numpy as np
import json
import time
import pickle
from imageio import imread
from PIL import Image
import matplotlib.pyplot as plt

def process_json(json_path):
    image_age_list = []
    image_gender_list = []
    image_emotion_list = []
    image_id_list = []
    
    bad_json_list = []
    bad_json_count = 0
    
    # for each json file in the json_path, load the json file and extract the attributes
    for file in sorted(os.listdir(json_path)):
        image_id = file.split('.')[0]
        
        if json.load(open(json_path+file)) == []:
            bad_json_count += 1
            bad_json_list.append(file)
            continue
            
        attribute_data = json.load(open(json_path+file))[0]["faceAttributes"]
        
        image_id_list.append(image_id)
        image_gender_list.append(attribute_data['gender'])
        image_age_list.append(attribute_data['age'])
        
        # in the emotiion attribute data set only the highest value to 1 and all others to 0
        emotion_dict = attribute_data['emotion']
        # max_emotion = max(emotion_dict, key=emotion_dict.get)
        # for key in emotion_dict:
        #     if key == max_emotion:
        #         emotion_dict[key] = 1
        #     else:
        #         emotion_dict[key] = 0
        
        image_emotion_list.append(attribute_data['emotion'])
        
    pd.DataFrame(bad_json_list).to_csv('bad_json_list.csv')
    print(f'empty json count: {bad_json_count}')
    
    # create a dictionary with the extracted attributes
    json_dict = {
        'image id': image_id_list,
        'age': image_age_list,
        'gender': image_gender_list
        }
    
    # for each entry in emotion list, get the key and add it to the json_dict
    emotion_keys = set([key for emotion in image_emotion_list for key in emotion.keys()])
    
    # create a list of lists for each emotion key
    for key in sorted(emotion_keys):
        json_dict[key] = [emotion[key] for emotion in image_emotion_list]
        
    attribute_df = pd.DataFrame(json_dict)
    attribute_df.sort_values(by='image id', inplace=True)
    attribute_df.set_index('image id', inplace=True)
    attribute_df['gender'] = attribute_df['gender'].map({'male': 0.0, 'female': 1.0})
    
    # plot the sum of each of the keys in the emotion dictionary
    emotion_df = pd.DataFrame(image_emotion_list)
    print(emotion_df.sum())
    
    return attribute_df

def process_image(image_path): 
    image_url = image_path.split(os.getcwd())[1]
    processed_image = Image.open(image_path)
    # convert image to array
    image_vector = np.asarray(processed_image)            
    # flatten the image array to a single dimension
    image_vector = image_vector
    return image_vector, image_url


def pre_processing(dataset_path, json_path):
    # Load the images from the dataset_path
    # convert images to arrays
    # combine image ids with image vectors and add to a list
    # convert list to dataframe
    image_vector_list = []
    image_id_list = []
    image_url_list = []
    image_file_path_list = []
    dataset_dir_list = os.listdir(dataset_path)
    
    for dataset in sorted(dataset_dir_list):
        img_list = os.listdir(dataset_path+'/'+dataset)
        print(f'loading the images from {dataset}')
        images = sorted(os.listdir(dataset_path+'/'+dataset))
        for image in images:
            image_id_list.append(image.split('.')[0])
            image_vector, image_url = process_image(dataset_path+dataset+'/'+image)
            image_url_list.append(image_url)
            image_vector_list.append(image_vector)

            
    # convert list of image vectors to dataframe and add image id
    print(image_file_path_list)
    image_dict = { 'image id': image_id_list, 'image_path': image_url_list }
    image_dataframe = pd.DataFrame(image_dict)
    image_dataframe.sort_values(by='image id', inplace=True)
    image_dataframe.set_index('image id', inplace=True)
    return image_dataframe
    
def compile_final_dataset(images_dataframe, json_path):
    # load the json files from the json_path and extract relevant attributes into a dataframe
    attribute_df = process_json(json_path)
    
    # add attribute dataframe to image dataframe
    final_dataframe = images_dataframe.merge(attribute_df, on='image id')

    # return the processed image dataset
    return final_dataframe

def main(type, resolution='128x128'):
    if type == 'sample':
        image_filepath = os.getcwd() + '/sample/sample_images/'
        json_filepath = os.getcwd() + '/sample/sample_json/'
    elif type == 'ffhq_dataset':
        image_filepath = os.getcwd() + '/ffhq_dataset/thumbnails128x128/'
        json_filepath = os.getcwd() + '/ffhq_dataset/ffhq-features-dataset-master/json/'
    else:
        raise ValueError('Invalid type, please specify either sample or ffhq_dataset.')
    
    if not os.path.exists(f'processed_image_data_set_{resolution}.pkl'):
        print('Processing image data set...')
        vectorized_images = pre_processing(image_filepath, json_filepath)
        processed_image_data_set = compile_final_dataset(vectorized_images, json_filepath)
    else:
        processed_image_data_set = pickle.load(open(f'processed_image_data_set_{resolution}.pkl', 'rb'))    
    
    processed_image_data_set.dropna(inplace=True)
    processed_image_data_set.to_json(f'processed_image_data_set_{resolution}.json')
    processed_image_data_set.to_csv(f'{type}/processed_image_data_set_{resolution}.csv')
        
    # save the processed image dataset to a pickle file
    # pickle.dump(processed_image_data_set, open(f'processed_image_data_set_{resolution}.pkl', 'wb'))
    # processed_image_data_set = []
    # load the processed image dataset from the pickle file (this is just proof that the pickle file works)
    # processed_image_data_set = pickle.load(open(f'processed_image_data_set_{resolution}.pkl', 'rb'))
    #print(f"image dimensions: {processed_image_data_set.loc['00000', 'image vector'].shape}")
    #print(f"dataset shape: {processed_image_data_set.shape}")
    print(processed_image_data_set.columns)
    
    if type == 'sample':
        print(processed_image_data_set)

    
if __name__ == '__main__':
    #main('sample', 'sample')
    main('ffhq_dataset', '128x128')