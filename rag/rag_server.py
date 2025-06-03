import time
import logging

from flask import (
    jsonify,
    request,
    Blueprint,
)

bp = Blueprint('rag', __name__, url_prefix='/')


@bp.route('/chat_completion', methods=['POST', 'GET'])
def chat_completion():
    return 'test output'

    logging.info('chat_completion')

    req = request.json

    logging.info(f"chat_completion: req = {req}")

    time.sleep(2)
    response = {
        "code": "1",
        "message": "success",
        "data": "Just echo to user input"
    }
    return jsonify(response)
