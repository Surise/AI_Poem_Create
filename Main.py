import re
import numpy as np
import tensorflow as tf
from collections import Counter
from surise.surise import *
from flask import Flask,redirect,request,render_template,url_for
from pypinyin import lazy_pinyin, Style
def Prosodic_obtain(word):
    style = Style.FINALS
    return(lazy_pinyin(word, style=style))
def predict(model, token_ids):
    _probas = model.predict([token_ids, ],verbose=0)[0, -1, 3:]
    p_args = _probas.argsort()[-100:][::-1]
    p = _probas[p_args]
    p = p / sum(p)
    target_index = np.random.choice(len(p), p=p)
    token_ids.append(p_args[target_index] + 3 )
    return token_ids
def Surise(token_ids,model,handletoken):
    i=0
    while len(token_ids) < 36:
        token_ids =predict(model, token_ids)
        if token_ids[-1] in [9,15] and token_ids[-2] in [9,15]:
            while token_ids[-1] in [9,15] and token_ids[-2] in [9,15]:
                TAR = predict(model, token_ids)
            token_ids=TAR
        if token_ids[-1] == handletoken.end_id:
            break
    return token_ids
def tokenttt(mode):
    global handletoken
    if mode=="七绝":
        poem_path = './data/qijue-all.txt'
    if mode=="五律":
        poem_path = './data/wulv-all.txt'
    if mode=="五绝":
        poem_path = './data/wujue-all.txt'
    else:
        poem_path = './data/poems.txt'
    maxlenth = 64
    Error_word = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
    poetry = []
    with open(poem_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  
    for line in lines:
        fields = re.split(r"[:：]", line)
        if len(fields) != 2:
            continue
        content = fields[1]
        if len(content) > maxlenth - 2:
            continue
        if any(word in content for word in Error_word):
            continue
        poetry.append(content.replace('\n', ''))
    Minimum_of = 8
    counter = Counter()
    for line in poetry:
        counter.update(line)
    tokens = [token for token, count in counter.items() if count >= Minimum_of]
    tokens = ["[PAD]", "[NONE]", "[START]", "[END]"] + tokens
    handletoken = Handle_token(tokens)
app=Flask(__name__)
@app.route('/poem',methods=["GET"])#http://127.0.0.1:5000/poem?s=尘
def poem():
    begin_char=request.args.get("s")#获取首句参数
    mode=request.args.get("y")#获取首句参数
    result="还未生成"
    print(mode)
    print(begin_char)
    if (begin_char=="" and mode=="") or(begin_char==None and mode==None):
        print("No get")
        begin_char="请输入您想生成诗歌的第一句:"
        mode="请输入您想选择的模型版本（内置:poem，五律，五绝，七绝):"
        info={ 's' :begin_char,"y":mode,'result':result}
        return render_template("Poem.html",**info)
    tokenttt(mode)
    if mode=="七绝":
        model = tf.keras.models.load_model('./model/qijue9.keras')
    if mode=="五律":
        model = tf.keras.models.load_model('./model/wulv9.keras')
    if mode=="五绝":
        model = tf.keras.models.load_model('./model/wujue9.keras')
    if mode=="poem":
        model = tf.keras.models.load_model('./model/poems.keras')
    token_ids = handletoken.encode(begin_char)[:-1]
    token_ids=Surise(token_ids,model,handletoken)
    result="".join(handletoken.decode(token_ids))
    print(result)
    info={ 's' :begin_char,"y":mode,'result':result}
    return render_template("Poem.html",**info)
@app.route('/home')#http://127.0.0.1:2333/home
def home():
    return render_template("Home.html")
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=2333)

