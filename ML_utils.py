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
    import shap
    
    '''
    Build a MLP model to quickly analyze tabular data
    
    Args:
        datapath: str, path to the data file
        savepath: str, path to save the analysis results
        
    Example:
        import os
        from ML_utils import MLP_evaluate_tabular_data

        data_path = './data/example.csv'
        save_path = './analysis_result/analysis_results'
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    
        MLP_evaluate_tabular_data(data_path, save_path)
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
        
        # recoding the label
        if y.min() != 0:
            y = y - y.min()
        
        # One-hot encoding
        if (x.dtypes == 'object').any():
            x = pd.get_dummies(x)

        feature_names = x.columns

        x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=None, stratify=y)

        scaler = MinMaxScaler()
        x_temp = scaler.fit_transform(x_temp)
        x_test = scaler.transform(x_test)
        
        x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.2, random_state=None, stratify=y_temp)
        
        y_train = y_train.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # class weights
        class_counts = y_train.value_counts()
        total_samples = len(y_train)
        weights = total_samples / class_counts
        weights = weights.sort_index()  # 確保權重是按類別索引排序
        weights = torch.tensor(weights.values, dtype=torch.float32)

        return x_train, y_train, x_val, y_val, x_test, y_test, weights, feature_names

    def build_data_loader(x_train, y_train, x_val, y_val, x_test, y_test):
    
        X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=18, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=18, shuffle=False)
        
        return train_loader, val_loader, test_loader
            
    def fit_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience=10, min_delta=0.001, savepath='./'):
        
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
        
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0
            total_train_correct = 0
            total_val_correct = 0
            total_train_samples = 0
            total_val_samples = 0
            
            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()
                output_train = model(data)
                train_loss = criterion(output_train, target)
                train_loss.backward()
                optimizer.step()
                
                total_train_loss += train_loss.item() * data.size(0)
                total_train_correct += (output_train.argmax(1) == target).sum().item()
                total_train_samples += data.size(0)
            
            model.eval()
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
            
            # Early Stopping
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                best_model = model.state_dict()
                patience_counter = 0 
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
                
        if best_model is not None:
            model.load_state_dict(best_model)
            
        plot_loss_acc(train_losses, val_losses, train_accs, val_accs, savepath)
            
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
    
    def save_feature_importance(model, x_train, x_val, x_test, feature_names):
        background = torch.tensor(x_train, dtype=torch.float32)
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(torch.tensor(x_val, dtype=torch.float32))
        shap.summary_plot(shap_values, x_test, feature_names=feature_names, show = False)
        plt.savefig(savepath + '/feature_importance.png')
        plt.close()
    
    # 1. data preprocessing
    x_train, y_train, x_val, y_val, x_test, y_test, weights, feature_names = preprocess_tabular_data(datapath)
    train_loader, val_loader, _ = build_data_loader(x_train, y_train, x_val, y_val, x_test, y_test)
    
    # 2. build model
    model = MLP(x_train.shape[1], len(y_train.unique()), 0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=weights.float())
    
    epochs = 500
    patience = 20
    min_delta = 0.001
    fit_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, min_delta, savepath)
    
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
    save_feature_importance(model, x_train, x_val, x_test, feature_names)
    save_test_results(model, mcc, acc, precision, recall, f1, auc, cm, savepath)
    
    print('Model and test results saved successfully.')
    
def AutoEncoder_imputing_missed_data(missing_data_path, model_path, save_path, num_feats, cat_feats, keep_feats, label, column_order):

    '''
    Using AutoEncoder to impute missing data.

    Parameters
    ----------

    missing_data_path : str
        The path of the missing data.
    model_path : str
        The path of the model.
    save_path : str
        The path to save the imputed data.
    num_feats : list of str
        The list of numerical features.
    cat_feats : list of str
        The list of categorical features.
    keep_feats : list of str
        The list of features that are not discarded in the model training.
    label : list of str
        The list of labels.
    column_order : list of str
        The order of columns.

    Examples
    --------

    missing_data_path = "./Dataset/Data.csv"
    model_path = "./Model/"
    save_path = "./Dataset/Imputed_Data.csv/"

    num_feats = ['num_feat1', 'num_feat2', 'num_feat3', 'num_feat4', 'num_feat5', 'num_feat6', 'num_feat7', 'num_feat8', 'num_feat9', 'num_feat10']

    cat_feats = ['cat_feat1', 'cat_feat2', 'cat_feat3', 'cat_feat4', 'cat_feat5', 'cat_feat6', 'cat_feat7', 'cat_feat8', 'cat_feat9', 'cat_feat10']

    keep_feats = ['num_feat1', 'num_feat5', 'num_feat6', 'cat_feat1', 'cat_feat3', 'cat_feat4']

    label = ['Label1', 'Label2', 'Label3']

    column_order = ['num_feat1', 'num_feat2', 'num_feat3', 'num_feat4', 'num_feat5', 'num_feat6', 'num_feat7', 'num_feat8', 'num_feat9', 'num_feat10', 
                    'cat_feat1', 'cat_feat2', 'cat_feat3', 'cat_feat4', 'cat_feat5', 'cat_feat6', 'cat_feat7', 'cat_feat8', 'cat_feat9', 'cat_feat10', 
                    'Label1', 'Label2', 'Label3']


    AutoEncoder_imputing_missed_data(missing_data_path, model_path, save_path, num_feats, cat_feats, keep_feats, label, column_order)
    '''

    #--------------------------------------utils--------------------------------------#

    import os
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.utils.data
    import matplotlib.pyplot as plt

    # 自定義Dropout
    class CustomDropout(nn.Module):
        def __init__(self, p=0.5, keep_indices=None):
            super(CustomDropout, self).__init__()
            self.p = p
            self.keep_indices = keep_indices

        def forward(self, x):
            if self.training:  # 只在訓練模式下應用dropout
                mask = torch.ones_like(x).bernoulli_(1 - self.p)
                if self.keep_indices is not None:
                    mask[:, self.keep_indices] = 1  # 確保指定的特徵不被丟棄
                return x * mask
            return x

    # Autoencoder
    class Autoencoder(nn.Module):
        def __init__(self, dim, theta, keep_indices=None):
            super(Autoencoder, self).__init__()
            self.dim = dim
            self.keep_indices = keep_indices
            
            self.drop_out = CustomDropout(p=0.3, keep_indices=self.keep_indices)
            
            self.encoder = nn.Sequential(
                nn.Linear(dim+theta*0, dim+theta*1),
                nn.Tanh(),
                nn.Linear(dim+theta*1, dim+theta*2),
                nn.Tanh(),
                nn.Linear(dim+theta*2, dim+theta*3)
            )
                
            self.decoder = nn.Sequential(
                nn.Linear(dim+theta*3, dim+theta*2),
                nn.Tanh(),
                nn.Linear(dim+theta*2, dim+theta*1),
                nn.Tanh(),
                nn.Linear(dim+theta*1, dim+theta*0)
            )
            
        def forward(self, x):
            x = x.view(-1, self.dim)
            x_missed = self.drop_out(x)
            
            z = self.encoder(x_missed)
            out = self.decoder(z)
            
            out = out.view(-1, self.dim)
            
            return out

    def check_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    # 切分訓練、驗證和測試集
    def split_data(data, val_size, label):
        '''切分方式
        沒有任何缺失值的個案作為訓練集
        訓練集中切分出一部分作為驗證集
        任何有缺失值的個案將被劃入缺失集
        '''    
        
        """
        參數:
        data : DataFrame
            包含所有數據的DataFrame。
        val_size : float
            驗證集的比例。

        返回:
        DataFrame
            訓練集。
        DataFrame
            驗證集。
        DataFrame
            測試集。
        """
        
        train_data = data.dropna().copy()
        train_data.reset_index(drop=True, inplace=True)
        shuffled_indices = np.random.permutation(train_data.index)
        val_set_size = int(len(train_data) * val_size)
        val_data = train_data.iloc[shuffled_indices[:val_set_size]].copy()
        train_data = train_data.iloc[shuffled_indices[val_set_size:]].copy()
        missed_data = data[data.isna().any(axis=1)].copy()

        train_data.reset_index(drop=True, inplace=True)
        val_data.reset_index(drop=True, inplace=True)
        missed_data.reset_index(drop=True, inplace=True)

        train_data_label = train_data[label]
        val_data_label = val_data[label]
        missed_data_label = missed_data[label]

        train_data = train_data.drop(columns=label)
        val_data = val_data.drop(columns=label)
        missed_data = missed_data.drop(columns=label)

        return train_data, val_data, missed_data, train_data_label, val_data_label, missed_data_label

    # 前處理 - 數值特徵
    def preprocess_num_features(data, num_feats):
        
        # # 將特定數值特徵限制在一個最大值
        # def cap_feature(value, cap):
        #     return min(value, cap)
        
        '''
        Example:
        data['num_feat1'] = data['num_feat1'].apply(lambda x: cap_feature(x, 36))
        data['num_feat2'] = data['num_feat2'].apply(lambda x: cap_feature(x, 8))

        '''
        # 如果數值特徵有缺失值，填補為0
        data[num_feats] = data[num_feats].fillna(0)  

        return data

    # 前處理 - 標準化數值特徵
    def normalize_data(train_data, val_data, missed_data, num_feats):
        scaler = MinMaxScaler()
        scaler.fit(train_data[num_feats])
        train_data.loc[:, num_feats] = scaler.transform(train_data[num_feats])
        val_data.loc[:, num_feats] = scaler.transform(val_data[num_feats])
        # 對缺失數據進行標準化時，要避開0(數值為0代表缺失)
        for idx, col in enumerate(num_feats):
            min_ = scaler.data_min_[idx]
            range_ = scaler.data_range_[idx]
            mask = missed_data[col] != 0
            missed_data.loc[mask, col] = (missed_data.loc[mask, col] - min_) / range_

        return train_data, val_data, missed_data, scaler

    # 逆轉換數據
    def reverse_data(data, cat_feats, num_feats, scaler):
        """
        將類別、數值特徵轉換回原始形式。

        參數:
        data : DataFrame
            原始的DataFrame。
        cat_feats : list of str
            包含所有類別特徵名稱的列表。
        num_feats : list of str
            包含所有數值特徵名稱的列表。
        scaler : Scaler object
            用於逆標準化數值特徵的標準化器對象。

        返回:
        DataFrame
            逆轉換完類別特徵和數值特徵的DataFrame。
        """
        # 首先對數值特徵進行逆標準化
        data_inverse = data.copy()
        data_inverse[num_feats] = scaler.inverse_transform(data[num_feats])

        # 再將One-Hot Encoding過的類別特徵恢復到原始類別
        for feature in cat_feats:
            columns = [col for col in data_inverse.columns if col.startswith(feature)]
            data_inverse[feature] = data_inverse[columns].idxmax(axis=1).apply(lambda x: x.split('_')[-1])

        # 最後移除所有One-Hot Encoding的欄位
        columns_to_remove = [col for col in data_inverse.columns if col.rsplit('_', 1)[-1].isdigit()]
        data_inverse = data_inverse.drop(columns=columns_to_remove)

        return data_inverse

    # 記錄缺失位置
    def find_missing_mask(data, num_feats):
        '''確認missing data哪些欄位缺失特徵
        (1) 首先判別是數值還是類別特徵
        (2) 如果是數值特徵，確認該欄是否為0，若為0則為缺失
        (3) 如果是類別特徵，確認One-Hot Encoding後的相關特徵欄是否全為0，若全為0則為缺失
        (4) 將缺失的部份記錄下來，並使用imputed_data對該部份進行填補
        '''

        """
        參數:
        data : DataFrame
            包含所有數據的DataFrame。
        num_feats : list of str
            包含所有數值特徵名稱的列表。

        返回:
        dict
            包含缺失值位置的字典。
        """
        missing_mask = {}

        for column in data.columns:
            if column in num_feats:  # 數值特徵
                missing_mask[column] = (data[column] == 0)
            else:  # 類別特徵
                base_name = column.rsplit('_', 1)[0]  # 提取類別特徵的名稱（去除後綴數字和底線）
                related_columns = [col for col in data.columns if col.startswith(base_name + '_')]  # 獲取所有以該名稱開頭的特徵欄位
                is_missing = np.array(data[related_columns].sum(axis=1) == 0)
                for col in related_columns:
                    missing_mask[col] = is_missing

        return missing_mask

    # 拼接數據
    def concat_data(train_data_inverse, val_data_inverse, combined_data_inverse, train_data_label, val_data_label, missed_data_label):
        train_data_inverse = pd.concat([train_data_inverse, train_data_label], axis=1)
        val_data_inverse = pd.concat([val_data_inverse, val_data_label], axis=1)
        combined_data_inverse = pd.concat([combined_data_inverse, missed_data_label], axis=1)
        combined_data_inverse = pd.concat([combined_data_inverse, train_data_inverse], axis=0)
        combined_data_inverse = pd.concat([combined_data_inverse, val_data_inverse], axis=0)

        return combined_data_inverse

    # 儲存MSE變化
    def save_mse(cost_list, save_path, type):
        plt.figure()
        plt.plot(cost_list)
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('MSE Change During ' + type)
        plt.savefig(os.path.join(save_path, type + '_MSE.png'))  # 儲存圖片
        plt.close()

        return

    # 檢查目錄
    check_dir(save_path)
    check_dir(model_path)

    theta = 10 # encoder維度
    num_epochs = 10000 # 訓練次數

    # 驗證集比例
    val_size = 0.2

    # 批次大小
    batch_size  = 512

    #--------------------------------------setting--------------------------------------#

    # 讀取數據
    data = pd.read_csv(missing_data_path)
    data = data[num_feats + cat_feats + label]

    # 切分訓練、驗證和缺失集
    train_data, val_data, missed_data, train_data_label, val_data_label, missed_data_label = split_data(data, val_size, label)

    # 前處理 - 數值特徵
    train_data = preprocess_num_features(train_data, num_feats)
    val_data = preprocess_num_features(val_data, num_feats)
    missed_data = preprocess_num_features(missed_data, num_feats)

    # 對所有數值特徵做標準化
    train_data, val_data, missed_data, scaler = normalize_data(train_data, val_data, missed_data, num_feats)

    # 前處理 - 類別特徵
    # One-Hot Encoding
    train_data = pd.get_dummies(pd.DataFrame(train_data), columns=cat_feats)
    val_data = pd.get_dummies(pd.DataFrame(val_data), columns=cat_feats)
    missed_data = pd.get_dummies(pd.DataFrame(missed_data), columns=cat_feats)

    # 將類別特徵的One-Hot Encoding欄位名稱做修正，去除.0
    for col in train_data.columns:
        if col.endswith('.0'):
            train_data.rename(columns={col: col[:-2]}, inplace=True)
            val_data.rename(columns={col: col[:-2]}, inplace=True)
            missed_data.rename(columns={col: col[:-2]}, inplace=True)

    # 將 DataFrame 轉換成 NumPy 數組
    train_data_np = train_data.to_numpy()
    val_data_np = val_data.to_numpy()
    missed_data_np = missed_data.to_numpy()

    # 將 NumPy 數組轉換成 PyTorch 張量
    train_data_tensor = torch.from_numpy(train_data_np).float()
    val_data_tensor = torch.from_numpy(val_data_np).float()
    missed_data_tensor = torch.from_numpy(missed_data_np).float()

    # 創建 DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_data_tensor, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data_tensor, batch_size=batch_size, shuffle=False)

    keep_indices = []
    for feature in keep_feats:
        if feature in num_feats:
            keep_indices.append(num_feats.index(feature))
        else:
            base_name = feature.rsplit('_', 1)[0]
            related_columns = [col for col in missed_data.columns if col.startswith(base_name + '_')]
            keep_indices.extend([missed_data.columns.get_loc(col) for col in related_columns])

    #--------------------------------------------1. 資料前處理--------------------------------------------#

    #--------------------------------------------2. 載入/訓練模型--------------------------------------------#

    cols = train_data.shape[1]

    # 先判斷是否有訓練好的模型，有的話直接載入，沒有的話重新訓練
    if os.path.exists(model_path + 'model.pth'):
        model = Autoencoder(dim=cols, theta=theta, keep_indices=keep_indices)
        model.load_state_dict(torch.load(model_path + 'model.pth', map_location='cpu'))
        print("Model Loaded!")
    else:
        model = Autoencoder(dim=cols, theta=theta, keep_indices=keep_indices)
        loss = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), momentum=0.99, lr=0.002, nesterov=True)
        cost_list_train = []
        cost_list_val = []
        early_stop = False

        for epoch in range(num_epochs):
            
            total_batch = len(train_data) // batch_size
            
            # 訓練模型
            model.train()
            for i, batch_data in enumerate(train_loader):
                reconst_data = model(batch_data)
                cost = loss(reconst_data, batch_data)
                
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                        
                if (i+1) % (total_batch//2) == 0:
                    print('Epoch [%d/%d], Iter [%d/%d], Train Loss: %.6f'
                        % (epoch+1, num_epochs, i+1, total_batch, cost.item()))
                
                # early stopping: MSE
                if cost.item() < 4e-02:
                    early_stop = True
                    break
                
                cost_list_train.append(cost.item())

            # 驗證模型
            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                total_val_samples = 0
                for val_data in val_loader:
                    reconst_data = model(val_data)
                    val_loss = loss(reconst_data, val_data)
                    total_val_loss += val_loss.item() * len(val_data)
                    total_val_samples += len(val_data)
                
                avg_val_loss = total_val_loss / total_val_samples
                cost_list_val.append(avg_val_loss)
                print('Epoch [%d/%d], Validation Loss: %.6f'
                    % (epoch+1, num_epochs, avg_val_loss))
            
            if early_stop:
                break

        # 繪製MSE變化並儲存
        save_mse(cost_list_train, save_path, 'Training')
        save_mse(cost_list_val, save_path, 'Validation')
        # 儲存模型
        torch.save(model.state_dict(), model_path + 'model.pth')
        print("Learning Finished!")

    #--------------------------------------------2. 載入/訓練模型--------------------------------------------#

    #--------------------------------------------3. 填補缺失數據--------------------------------------------#
    # 使用訓練好的模型對缺失個案進行填補
    model.eval()
    imputed_data = model(missed_data_tensor)
    imputed_data = imputed_data.cpu().detach().numpy()

    # 將填補後的數據保存到新文件
    imputed_data = pd.DataFrame(imputed_data, columns=missed_data.columns)

    # 記錄缺失位置
    missing_mask = find_missing_mask(missed_data, num_feats)

    # 將缺失位置的字典轉為DataFrame
    missing_mask_df = pd.DataFrame(missing_mask)

    # 創建一個新的DataFrame來合併原始數據和填補後的數據
    combined_data = missed_data.copy()
    combined_data = combined_data.mask(missing_mask_df, imputed_data)

    #--------------------------------------------3. 填補缺失數據--------------------------------------------#

    #--------------------------------------------4. 儲存數據--------------------------------------------#

    # 將數據轉換回原始形式
    train_data_transforemd_df = pd.DataFrame(train_data)
    val_data_transforemd_df = pd.DataFrame(val_data)
    train_data_inverse = reverse_data(train_data_transforemd_df, cat_feats, num_feats, scaler)
    val_data_inverse = reverse_data(val_data_transforemd_df, cat_feats, num_feats, scaler)
    combined_data_inverse = reverse_data(combined_data, cat_feats, num_feats, scaler)

    # 拼接資料、標籤
    combined_data_inverse = concat_data(train_data_inverse, val_data_inverse, combined_data_inverse, train_data_label, val_data_label, missed_data_label)

    # 對combined_data_inverse的欄位名稱進行排序
    combined_data_inverse = combined_data_inverse[column_order]

    # 將轉換回原始維度的數據保存到新文件
    combined_data_inverse.to_csv(save_path + 'combined_data_inverse.csv', index=False)

def Masked_AutoEncoder_augmenting_figures():
    pass
