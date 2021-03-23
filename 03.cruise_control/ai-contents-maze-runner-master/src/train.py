import numpy as np
import pandas as pd
from tensorflow import keras
# from sklearn.externals import joblib
import joblib

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

class Model(object):
    def __init__(self):
        pass
        
    def baseline_model(self, input_d):
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten
        
        model = keras.Sequence()
        model.add(Dense(64, input_dim=input_d, kernel_initializer='normal', activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(5,activation='softmax'))
        
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        return model
    
    
    def train(self):
        print('-----------------------------')
        print('학습시킬 모델을 고르세요')
        print('1. Random Forest')
        print('2. Support Vector machine(SVM)')
        print('3. Deep Neural Network')
            
        model_choice = int(input('모델을 선택하세요:'))
        
        data_path = "../data/data_new4.csv"
        print('학습할 데이터는 {}'.format(data_path))
        df = pd.read_csv(data_path)
        df['ir_1-2'] = df['ir1'] - df['ir2']
        df['ir_1/2'] = df['ir1'] / df['ir2']

        df = df[df !=0]
        df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0)

        df = df.loc[:, ['ir1', 'ir2', 'degree', 'ir_1-2', 'ir_1/2']]
        
        
        # Original Data
        df_inputs = df.loc[:,['ir1', 'ir2', 'ir_1-2', 'ir_1/2']]
        df_outputs = df.loc[:,['degree']]

        # Normalized Data
        # df_inputs = df_n.loc[:,['ir_L', 'ir_R']]
        # df_outputs = df_n.loc[:,['degree']]
        # df_inputs = df_n.loc[:,['ir_L', 'ir_R', 'ir_1-2', 'ir_1/2']]
        # df_outputs = df_n.loc[:,['degree']]

        inputs = np.array(df_inputs)
        outputs = np.array(df_outputs)
  
        
        num_data = len(inputs)
        TRAIN_SPLIT = int(0.6 * num_data)
        TEST_SPLIT = int(0.2 * num_data + TRAIN_SPLIT)
        inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
        outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])
       
        if model_choice==1:
            from sklearn.ensemble import RandomForestClassifier 
              
            rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
            rf.fit(inputs_train, outputs_train.ravel())
              
            file_name = '../model/test1.pkl' 
            joblib.dump(rf, file_name)
            print('Randomforest train 완료')
            print('모델 위치 :{}'.format(file_name))
        
        elif model_choice==2:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
            
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
            clf.fit(inputs_train, outputs_train.ravel())
              
            file_name = '../model/test1.pkl' 
            joblib.dump(clf, file_name)
            print('SVM train 완료')
            print('모델 위치 :{}'.format(file_name))
        
        elif model_choice==3:
            
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten

            model = Sequential()
            model.add(Dense(64, input_dim=4, kernel_initializer='normal', activation='relu'))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(1))


            
            model.compile(loss='mean_squared_error', optimizer='adam')
            
            EPOCHS = 50
            #model = self.baseline_model(4)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
            
            history = model.fit(inputs_train,outputs_train.ravel(), epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
            model_path = '../model/' + 'model.h5'
            model.save(model_path)
            
            print('DNN train 완료')
            print('모델 위치 :{}'.format(model_path))
        
         
        # 객체를 pickled binary file 형태로 저장한다 
        
        
  
        
def main():
    
    pass

    


if __name__ == "__main__":
    main()        
        
        
        
        
