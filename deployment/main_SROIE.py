from flask import Flask, request, jsonify
from deployment.inference_SROIE import inference_pipe
from deployment.module_load import inference_init


(
    MODEL,
    OCR_URL,
    TOKENIZER,
    DEVICE,
    NUM_CLASSES,
    PARSE_MODE,
) = inference_init(dir_config="deployment/config/inference_SROIE.yaml")


app = Flask(__name__)


@app.route("/core", methods=["POST"])
def KIE_System():
    if request.method == "POST":
        file = request.files["file"]
        image_bytes = file.read()
        result = inference_pipe(
            MODEL,
            OCR_URL,
            TOKENIZER,
            DEVICE,
            NUM_CLASSES,
            image_bytes=image_bytes,
            parse_mode=PARSE_MODE,
        )
        return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=11451, debug=False)
