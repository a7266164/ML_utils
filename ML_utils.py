def MLP_evaluate_tabular_data(datapath, savepath):
    import seaborn as sns
    import json
    import torch
    import torch.nn as nn
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
    from torch.utils.data import DataLoader, TensorDataset
    import matplotlib.pyplot as plt
    
    '''
    Build a MLP model to quickly analyze tabular data
    
    Args:
        datapath: str, 輸入資料路徑
        savepath: str, 儲存模型與測試結果的路徑
        
    Example:
        from ML_utils import MLP_evaluate_tabular_data
    
        MLP_evaluate_tabular_data('./data/data.csv', './analysis_results')
    '''

    
    class MLP(nn.Module):
        '''
        model structure:
        feature_dim -> 2*feature_dim -> 2*feature_dim -> feature_dim -> num_of_classes
        '''
        def __init__(self, feature_dim, num_of_classes, dropout_rate):
            super(MLP, self).__init__()
            # 定義模型架構
            self.layer1 = nn.Linear(feature_dim, 2*feature_dim)
            self.layer2 = nn.Linear(2*feature_dim, 3*feature_dim)
            self.layer3 = nn.Linear(3*feature_dim, feature_dim)
            self.layer4 = nn.Linear(feature_dim, num_of_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout_rate)

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.relu(self.layer3(x))
            x = self.dropout(x)
            x = self.layer4(x)
            return x
        
    def preprocess_tabular_data(datapath):
        data = pd.read_csv(datapath)
        x = data.drop('label', axis=1)
        y = data['label']
        
        # 如果y的類別不是從0開始，重新編碼
        if y.min() != 0:
            y = y - y.min()
        
        # Label Encoding
        if (x.dtypes == 'object').any():
            x = pd.get_dummies(x)

        # 切割訓練集、測試集
        x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=None, stratify=y)

        # 數值歸一化
        scaler = MinMaxScaler()
        x_temp = scaler.fit_transform(x_temp)
        x_test = scaler.transform(x_test)
        
        # 切割訓練集、驗證集
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=None, stratify=y_temp)
        
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        
        return x_train, y_train, x_val, y_val, x_test, y_test
        
    def build_data_loader(x_train, y_train, x_val, y_val, x_test, y_test):
        
        # 轉換為張量
        X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # 創建數據加載器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=18, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)
        
        return train_loader, val_loader, test_loader
            
    def fit_model(model, train_loader, val_loader, criterion, optimizer, epochs, train_losses, val_losses, train_accs, val_accs, patience=10, min_delta=0.001):
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()  # 確保模型在訓練模式
            total_train_loss = 0
            total_val_loss = 0
            total_train_correct = 0
            total_val_correct = 0
            total_train_samples = 0
            total_val_samples = 0
            
            for data, target in train_loader:
                optimizer.zero_grad()
                output_train = model(data)
                train_loss = criterion(output_train, target)
                train_loss.backward()
                optimizer.step()
                
                total_train_loss += train_loss.item() * data.size(0)
                total_train_correct += (output_train.argmax(1) == target).sum().item()
                total_train_samples += data.size(0)
            
            model.eval()  # 設定模型到評估模式
            with torch.no_grad():
                for data, target in val_loader:
                    output_val = model(data)
                    val_loss = criterion(output_val, target)
                    
                    total_val_loss += val_loss.item() * data.size(0)
                    total_val_correct += (output_val.argmax(1) == target).sum().item()
                    total_val_samples += data.size(0)
            
            avg_train_loss = total_train_loss / total_train_samples
            avg_val_loss = total_val_loss / total_val_samples
            train_accuracy = total_train_correct / total_train_samples
            val_accuracy = total_val_correct / total_val_samples
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
            
            # 檢查是否有改善
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()
                patience_counter = 0  # 重置耐心計數器
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
                
        if best_model is not None:
            model.load_state_dict(best_model)
            
    def plot_loss_acc(train_losses, val_losses, train_accs, val_accs, savepath):
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.legend()
        plt.savefig(savepath + '/loss.png')
        plt.close()
        
        plt.plot(train_accs, label='Training Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.legend()
        plt.savefig(savepath + '/accuracy.png')
        plt.close()

    def save_test_results(model, mcc, acc, precision, recall, f1, auc, cm, savepath):
        results = {
            'MCC': mcc,
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'AUC': auc,
        }
        
        with open(savepath + '/test_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(savepath + '/confusion_matrix.png')
        plt.close()
        
        torch.save(model.state_dict(), savepath + '/model.pth')
        
    # 1. data preprocessing
    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_tabular_data(datapath)
    train_loader, val_loader, _ = build_data_loader(x_train, y_train, x_val, y_val, x_test, y_test)
    
    # 2. build model
    model = MLP(x_train.shape[1], len(y_train.unique()), 0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 100
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    fit_model(model, train_loader, val_loader, criterion, optimizer, epochs, train_losses, val_losses, train_accs, val_accs)
    
    # 3. evaluate model
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(x_test).float())
        test_labels_pred = torch.max(test_outputs, 1)[1].numpy()
    mcc = matthews_corrcoef(y_test, test_labels_pred)
    acc = accuracy_score(y_test, test_labels_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_labels_pred, average='macro')
    cm = confusion_matrix(y_test, test_labels_pred)
    if len(y_test.unique()) == 2:
        auc = roc_auc_score(y_test, test_outputs.numpy()[:,1])
    else:
        auc = None

    # 4. save process and results
    plot_loss_acc(train_losses, val_losses, train_accs, val_accs, savepath)
    save_test_results(model, mcc, acc, precision, recall, f1, auc, cm, savepath)
    
    print('Model and test results saved successfully.')
    
def AutoEncoder_imputing_missed_data():
    pass