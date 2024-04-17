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
model.fit(X_train, y_train, epochs=100, verbose=0)
```

