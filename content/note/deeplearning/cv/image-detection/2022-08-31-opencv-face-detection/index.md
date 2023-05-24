---
title: OpenCV 人脸检测
author: 王哲峰
date: '2022-08-31'
slug: opencv-face-detection
categories:
  - computer vision
tags:
  - article
---

<style>
details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
}
summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
}
details[open] {
    padding: .5em;
}
details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
}
img {
    pointer-events: none;
}
</style>

<details><summary>目录</summary><p>

- [目标简介](#目标简介)
- [项目结构](#项目结构)
  - [Flask App](#flask-app)
  - [Web 页面](#web-页面)
    - [index.html](#indexhtml)
    - [stop.html](#stophtml)
- [运行 Flask App](#运行-flask-app)
- [参考](#参考)
</p></details><p></p>

# 目标简介

目标: 使用 Flask API 部署 OpenCV App 进行人脸检测
实现技术: 

* Flask
    - Flask 是一个广泛使用的微型 Web 框架，用于在 Python 中构建 API。
      它是一个简单而强大的 Web 框架，旨在快速轻松地启动，并能够扩展到复杂的应用程序
* OpenCV
    - OpenCV 是一个 Python 库，旨在解决计算机视觉问题。它用于各种应用，例如人脸检测、
      视频捕获、跟踪移动对象和对象披露
    - Haarcascade 算法，是一种对象检测算法，用于识别图像或实时视频中的人脸。该算法使用边缘或线检测特征

![img](images/haarcascade.jpeg)

# 项目结构

![img](images/opencv_flask_face.png)

## Flask App

```python
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

def capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)
    while True:
        # read the camera frame
        success, frame = camera.read()
        detector = cv2.CascadeClassifier("Haarcascades/haarcascade_frontalface_default.xml")
        faces = detector.detectMultiScale(frame, 1, 2, 6)
        # draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield(b"--frame\r\n"
              b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods = ["POST"])
def start():
    return render_template("index.html")

@app.route("/stop", methods = ["POST"])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template("stop.html")

@app.route("./video_capture")
def video_capture():
    return Response(capture_by_frames(), mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug = True, use_reloader = False, port = 8000)
```

## Web 页面

### index.html

```html
<!DOCTYPE html>
<html>
 <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
 <title>Dharmaraj - Face Detection</title>
<style>
h2
{
padding-bottom:20px;
font-weight: 600;
font-size: 3.2em
}
img {
    pointer-events: none;
}
</style>
  <body>
    <div class="container"><center><h2>Face Detection</h2></center>
      <div class="col-lg-offset-2 col-lg-8">
        <center><form  class="form-inline" action = "/stop" method = "post" enctype="multipart/form-data">          
          <input type = "submit" class="btn btn-danger btn-md btn-block" value="Stop">
             </form></center>
                <center><form  class="form-inline" action = "/start" method = "post" enctype="multipart/form-data">          
          <input type = "submit" class="btn btn-success btn-md btn-block" value="Start">
             </form></center><br></div>
                    <div class="col-lg-offset-2 col-lg-8">
         <img src="{{ url_for('video_capture') }}" width="100%">
        </div></div>
    </body>
</html>
```

### stop.html

```html
<!DOCTYPE html>
<html>
 <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
 <title>Dharmaraj - Face Detection</title>
<style>
h2
{
padding-bottom:20px;
font-weight: 600;
font-size: 3.2em
}
img {
    pointer-events: none;
}
</style>
  <body>
    <div class="container">
    <center><h2>Face Detection</h2></center>
            <div class="col-lg-offset-2 col-lg-8">
                  <center><form  class="form-inline" action = "/stop" method = "post" enctype="multipart/form-data">          
          <input type = "submit" class="btn btn-danger btn-md btn-block" value="Stop">
                       </form></center>
                <center><form  class="form-inline" action = "/start" method = "post" enctype="multipart/form-data">          
          <input type = "submit" class="btn btn-success btn-md btn-block" value="Start">
             </form></center><br>                
            </div></div>
    </body>
</html>
```

# 运行 Flask App

```bash
$ flask app
```

- http://127.0.0.1:8000/


# 参考

* [Opencv-face-detection-deployment-using-flask-API](https://github.com/DharmarajPi/Opencv-face-detection-deployment-using-flask-API)

