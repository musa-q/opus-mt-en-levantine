from flask import Blueprint, request
from translator import translator

bp = Blueprint('home', __name__)

@bp.route('/ping')
def ping():
    return 'pong'

@bp.route('/translate', methods=['POST'])
def translate():
    texts = request.get_json()['texts']
    return translator.translate_batch(texts)