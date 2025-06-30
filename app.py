import os
import cv2
import numpy as np
import gc
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from datetime import datetime
from polarini_focusini import detect_infocus_mask

app = Flask(__name__)

# Используем временную директорию Render
UPLOAD_FOLDER = '/tmp/uploads'
RESULT_FOLDER = '/tmp/results'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB limit

# Создаем папки если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Генерируем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        
        # Обработка изображения
        img = cv2.imread(upload_path)
        if img is None:
            return "Ошибка чтения изображения", 400
        
        # Уменьшаем изображение для ускорения обработки
        max_size = 1024
        h, w = img.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        
        # Генерируем маску фокуса
        mask = detect_infocus_mask(
            img,
            limit_with_circles_around_focus_points=True,
            ignore_cuda=True,
            verbose=False
        )
        
        # Сохраняем результаты
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(app.config['RESULT_FOLDER'], mask_filename)
        cv2.imwrite(mask_path, mask * 255)
        
        # Создаем визуализацию
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        result_img = np.where(mask[:, :, np.newaxis], img, img_gray)
        result_filename = f"result_{filename}"
        result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
        cv2.imwrite(result_path, result_img)
        
        # Очистка памяти
        del img, mask, result_img
        gc.collect()
        
        return redirect(url_for('show_result', 
                               original=filename, 
                               mask=mask_filename, 
                               result=result_filename))

@app.route('/result')
def show_result():
    original = request.args.get('original')
    mask = request.args.get('mask')
    result = request.args.get('result')
    
    return render_template('result.html', 
                          original=original,
                          mask=mask,
                          result=result)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
