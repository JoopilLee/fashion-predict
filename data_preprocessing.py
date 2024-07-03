from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np
import pickle
import argparse
import random
import torch

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_and_preprocess_data(train_path, test_path, text_embed_path, image_embed_path):
    set_seed(42)

    with open(text_embed_path, 'rb') as f:
        data = pickle.load(f)
    df_text = pd.DataFrame(data) # text embedding
    df = pd.read_csv(image_embed_path, index_col='Unnamed: 0').transpose() # image embedding

    sizes_columns = ['44size', '55size', '66size', '77size', '88size', '99size', 'FFsize']
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    test['size_ratio'] = test[['44','55','66','77','88','99','FF']].apply(lambda row: list(round(row*100,2)), axis=1)
    train['size_ratio'] = train[['44','55','66','77','88','99','FF']].apply(lambda row: list(round(row*100,2)), axis=1)
    test[['44','55','66','77','88','99','FF']] = test[['44','55','66','77','88','99','FF']].apply(lambda row: round(row*100,2), axis=1)
    train[['44','55','66','77','88','99','FF']] = train[['44','55','66','77','88','99','FF']].apply(lambda row: round(row*100,2), axis=1)

    def one_hot_encode(lst):
        return [1 if value > 0 else 0 for value in lst]
    # size one-hot encoding 적용 # 
    test['one_hot'] = test['size_ratio'].apply(one_hot_encode) 
    train['one_hot'] = train['size_ratio'].apply(one_hot_encode)

    train = train[train['품번'].isin(list(df.columns))].reset_index().drop(['index','Unnamed: 0'],axis=1)
    test = test[test['품번'].isin(list(df.columns))].reset_index().drop(['index','Unnamed: 0'],axis=1)
    columns_name = ['품번','품명','소분류','season','판매수량','44size','55size','66size','77size','88size','99size','FFsize','size_ratio','one_hot']
    train.columns = columns_name
    test.columns = columns_name

    data1 = df.loc[:,test.품번.tolist()].transpose()
    data2 = test.set_index('품번')
    data3 = train.set_index('품번')
    data4 = df.loc[:,train.품번.tolist()].transpose()
    data5 = df_text.loc[:,test.품번.tolist()].transpose()
    data6 = df_text.loc[:,train.품번.tolist()].transpose()

    text_numbers = [str(i) for i in range(512,1024)]
    image_numbers = [str(i) for i in range(512)]
    merge_data = data3.join(data4, how='inner')
    merge_data_test = data2.join(data1, how='inner')
    merge_data_all = merge_data.join(data6, how = 'inner').reset_index()
    merge_data_all_test = merge_data_test.join(data5, how = 'inner').reset_index()

    ss = ['품번','품명','소분류','season','판매수량','44size','55size','66size','77size','88size','99size','FFsize','size_ratio','one_hot']
    numbers = [str(i) for i in range(1024)]
    ss.extend(numbers)

    merge_data_all.columns = ss
    merge_data_all_test.columns = ss

    return merge_data_all, merge_data_all_test, sizes_columns, text_numbers, image_numbers, train, test

def create_dataloaders(merge_data_all, merge_data_all_test, sizes_columns, text_numbers, image_numbers, batch_size):
    text_embeds_train = np.array(merge_data_all[text_numbers]).astype(np.float32) # text data
    image_embeds_train = np.array(merge_data_all[image_numbers]).astype(np.float32) # image data
    one_hot_train = np.array(merge_data_all['one_hot'].tolist()).astype(np.float32) # one-hot information 
    labels_train = np.array(merge_data_all[sizes_columns]).astype(np.float32)# train labels 

    text_embeds_test = np.array(merge_data_all_test[text_numbers]).astype(np.float32)
    image_embeds_test = np.array(merge_data_all_test[image_numbers]).astype(np.float32)
    one_hot_test = np.array(merge_data_all_test['one_hot'].tolist()).astype(np.float32)
    labels_test = np.array(merge_data_all_test[sizes_columns]).astype(np.float32)

    class FashionDataset(Dataset):
        def __init__(self, text_embeds, image_embeds, labels, one_hot_labels):
            self.text_embeds = text_embeds
            self.image_embeds = image_embeds
            self.labels = labels
            self.one_hot_labels = one_hot_labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.text_embeds[idx], self.image_embeds[idx], self.labels[idx], self.one_hot_labels[idx]

    full_dataset = FashionDataset(
        torch.tensor(text_embeds_train), 
        torch.tensor(image_embeds_train), 
        torch.tensor(labels_train), 
        torch.tensor(one_hot_train))

    train_size = int(0.8 * len(full_dataset)) # train 8
    valid_size = len(full_dataset) - train_size # valid 2 
    train_dataset, valid_dataset = random_split(full_dataset, [train_size, valid_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader, text_embeds_test, image_embeds_test, one_hot_test, labels_test