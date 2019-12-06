from init import create_app
from flask_script import Manager, Server

app = create_app()
manager = Manager(app)

manager.add_command("runserver", Server(use_reloader=False))

if __name__ == '__main__':
    manager.run()