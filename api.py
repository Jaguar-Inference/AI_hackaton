# api.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tempfile
import os
import traceback

import main as main_module

app = Flask(__name__)
CORS(app)


@app.route('/process', methods=['POST'])
def process_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            save_path = os.path.join(tmpdir, file.filename)
            file.save(save_path)

            # вызываем main.py для обработки
            report_path = main_module.handle_uploaded_file(save_path)

            # проверяем параметр send_file
            send_inline = request.form.get('send_file', '1') in ('1', 'true', 'True')

            if send_inline:
                # возвращаем файл прямо на фронт
                return send_file(
                    report_path,
                    as_attachment=True,
                    download_name=os.path.basename(report_path),
                    mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # иначе возвращаем JSON с info о файле
            return jsonify({
                'report_path': report_path,
                'report_filename': os.path.basename(report_path)
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'AI API running',
        'routes': ['/process (POST)', '/report?path=... (GET)']
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
