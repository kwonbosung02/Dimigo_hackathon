from flask import Flask
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)
global element


class CreateUser(Resource):
    def post(self):

        return {'status': element}

api.add_resource(CreateUser, '/user')

if __name__ == '__main__':
    app.run(debug=True)
    