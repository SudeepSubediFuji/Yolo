# Yoloの調査プロジェクト
<p>
本プロジェクトはYoloモデルの調査や学習のため作成しました。
目的：個々の顔を認証するようなプロジェクト。
</p>

## 本プロジェクトのPythonスクリプトの説明
* app.py　→　単純なYoloモデルを読み込んで、StreamLitウェブビューでカメラを表示して、カメラから画像を移った場合、物体や生物を検知して表示する
* liveCam　→ ライブウェブカメラで検知する
* livecamTest →　ライブカメラで人間の顔を認証して、名前を表示するソースコード（作成中）

app	->photo capture -> create bbox and detect object 
liveCam/liveCam_ObjDetection->  live web cam view and detection
livecamTest-> identifying person and displaying name from save image by checking and if not found register new 

#### 要件パッケージ：

```python
python -m venv ./仮想環境名
source ./仮想環境名/bin/activate
pip install ultralytics streamlit pillow numpy matplotlib transformers opencv-python Pillow av face_recognition
```

### 実行する方法

```python
## 同期に仮想環境の位置の場所
## 自分のソースコード
cd ./project/
streamlit run プロジェクト.py
```



