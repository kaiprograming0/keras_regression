# Kerasとは
ディープラーニング用のライブラリで、tensorflowよりも簡単に実装することが可能です。  
音声認識、画像判定、自然言語処理など様々なところで使用されます。  

回帰分析でkerasを用いて簡単なモデルを作りました。
### kerasモデル作成
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([layers.Dense(4, activation='relu'),
                          layers.Dense(4, activation='relu'),
                          layers.Dense(1)])
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
```
### 学習曲線の表示
```python
all_mae_histories = []

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=16, verbose=0)
mae_history = history.history['val_mae']
all_mae_histories.append(mae_history)

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
```
