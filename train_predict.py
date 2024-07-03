import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import argparse
from model import SalesPredictionModel # Only FFN Model
# from model_plus import SalesPredictionModel # FFN + FFN
from data_preprocessing import create_dataloaders, load_and_preprocess_data, set_seed

def train_model(model, dataloaders, criterion, one_hot_criterion, optimizer, args):
    train_loss_history = []
    valid_loss_history = []

    best_model_wts = model.state_dict()
    best_loss = float('inf')

    for epoch in range(args.num_epochs):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_one_hot_loss = 0.0

            for text_embeds, image_embeds, labels, one_hot_labels in dataloaders[phase]:
                text_embeds, image_embeds, labels, one_hot_labels = text_embeds.cuda(), image_embeds.cuda(), labels.cuda(), one_hot_labels.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    ratio_outputs = model(text_embeds, image_embeds, one_hot_labels)
                    one_hot_outputs = (ratio_outputs > 0).float() # 예측값을 one-hot 변환 
                    ratio_loss = criterion(ratio_outputs, labels) # 비율 예측 loss
                    one_hot_loss = one_hot_criterion(one_hot_outputs, one_hot_labels) # one-hot loss 
                    total_loss = (args.ratio_weight * ratio_loss) +  (args.one_hot_weight * one_hot_loss)

                    if phase == 'train':
                        total_loss.backward()
                        optimizer.step()

                running_loss += ratio_loss.item() * text_embeds.size(0)
                running_one_hot_loss += one_hot_loss.item() * text_embeds.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_one_hot_loss = running_one_hot_loss / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                valid_loss_history.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f}, One-Hot Loss: {epoch_one_hot_loss:.4f}')

            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        print(f'Epoch {epoch}/{args.num_epochs - 1}')
        print('-' * 10)

    model.load_state_dict(best_model_wts)
    return model, train_loss_history, valid_loss_history

def train(args):
    set_seed(7)
    merge_data_all, merge_data_all_test, sizes_columns, text_numbers, image_numbers, train_before, test_before = load_and_preprocess_data(args.train_path, args.test_path, args.text_embed_path, args.image_embed_path)
    train_dataloader, valid_dataloader, text_embeds_test, image_embeds_test, one_hot_test, labels_test = create_dataloaders(merge_data_all, merge_data_all_test, sizes_columns, text_numbers, image_numbers, args.batch_size)

    dataloaders = {'train': train_dataloader, 'valid': valid_dataloader}
    
    # gt가 0인 값에 보다 weight를 추가하여 loss 계산 
    def weighted_mse_loss(predictions, targets, zero_weight=args.loss_weight):
        weight = torch.ones_like(targets)
        weight[targets == 0] = zero_weight
        return (weight * (predictions - targets) ** 2).mean()
    
    model = SalesPredictionModel(embed_dim=512).cuda()
    criterion = nn.MSELoss()
    one_hot_criterion = weighted_mse_loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    trained_model, train_loss_history, valid_loss_history = train_model(model, dataloaders, criterion, one_hot_criterion, optimizer, args)

    #### train - valid loss plot #### 
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(valid_loss_history, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(args.plot_save_path)
    
    #### save model #### 
    torch.save(trained_model.state_dict(), args.model_save_path)
    return trained_model, text_embeds_test, image_embeds_test, one_hot_test, labels_test, test_before

def metrics(actual, predicted):
    mse = np.sum((np.array(actual) - np.array(predicted)) ** 2)
    rmse = np.sqrt(mse)
    mae = np.sum(np.abs(np.array(actual) - np.array(predicted)))
    return mse, rmse, mae

def predict(model, text_embeds_test, image_embeds_test, one_hot_test, labels_test, test_before, args):
    model.eval()
    with torch.no_grad():
        text_embeds_test = torch.tensor(text_embeds_test, dtype=torch.float32).cuda()
        image_embeds_test = torch.tensor(image_embeds_test, dtype=torch.float32).cuda()
        one_hot_test = torch.tensor(one_hot_test, dtype=torch.float32).cuda()
        ratio_predictions = model(text_embeds_test, image_embeds_test, one_hot_test)
    ratio_predictions = ratio_predictions.cpu().numpy()

    MSE, RMSE, MAE = [], [], []
    for i in range(len(labels_test)):
        rmse = metrics(labels_test[i], ratio_predictions[i])[1]
        mae = metrics(labels_test[i], ratio_predictions[i])[2]
        mse = metrics(labels_test[i], ratio_predictions[i])[0]
        MSE.append(mse)
        RMSE.append(rmse)
        MAE.append(mae)
    cnt = sum(1 for r in ratio_predictions if round(r[0],2) == 0)
    
    # 예측 결과 .csv 저장 
    results_df = pd.DataFrame(ratio_predictions) 
    results_df.columns = ['44','55','66','77','88','99','FF']
    test_before['predictions'] = results_df[['44','55','66','77','88','99','FF']].apply(lambda row: list(round(row,2)), axis=1)
    test_before['MSE'] = MSE 
    test_before['MAE'] = MAE
    test_before.to_csv(args.results_save_path)

    print(f'MSE: {np.mean(MSE)}')
    print(f'RMSE: {np.sqrt(np.mean(MSE))}')
    print(f'MAE: {np.mean(MAE)}')
    print('0으로 예측한 갯수: ', cnt) 
    
    with open(args.results_metrics_path, 'w') as f:
        f.write(f'MSE: {np.mean(MSE)}\n')
        f.write(f'RMSE: {np.sqrt(np.mean(MSE))}\n')
        f.write(f'MAE: {np.mean(MAE)}\n')
        f.write(f'Count of predictions as 0: {cnt}\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='./data/12week_train.csv')
    parser.add_argument('--test_path', type=str, default='./data/12week_test.csv')
    parser.add_argument('--text_embed_path', type=str, default='./data/fclip_text_embedding.pickle')
    parser.add_argument('--image_embed_path', type=str, default='./data/image_embeddings.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--ratio_weight', type=float, default=1)
    parser.add_argument('--one_hot_weight', type=float, default=5)
    parser.add_argument('--loss_weight', type=float, default=23.0)
    parser.add_argument('--results_metrics_path', type=str, default='./results/metrics.txt')
    parser.add_argument('--results_save_path', type=str, default='./results/ratio_predictions.csv')
    parser.add_argument('--plot_save_path', type=str, default='./results/loss_plot.png')
    parser.add_argument('--model_save_path', type=str, default='./train/trained_model.pth')
    args = parser.parse_args()

    trained_model, text_embeds_test, image_embeds_test, one_hot_test, labels_test, test_before = train(args)
    predict(trained_model, text_embeds_test, image_embeds_test, one_hot_test, labels_test, test_before, args)

if __name__ == "__main__":
    main()