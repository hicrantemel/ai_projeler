from flask import Flask, request, jsonify, send_from_directory
import util
import os

# Flask uygulamasını başlatıyoruz
app = Flask(__name__, static_folder='../UI')

# Ana sayfa
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'app.html')

# Statik dosyaları doğrudan sunuyoruz (CSS, JS, Images, vs.)
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

# Sınıflandırma işlemi
@app.route('/classify_image', methods=['POST'])
def classify_image():
    image_data = request.form.get('image_data')
    if not image_data:
        return jsonify({"error": "Image data not found"}), 400
    
    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# Uygulamayı başlat
if __name__ == "__main__":
    print("Yüz Tanıma Başlıyor.")
    util.load_saved_artifacts()
    app.run(port=5000)
