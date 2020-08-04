import sqlite3
from sqlite3 import Error

class DBService:
    def __init__(self):
        try:
            self.con = sqlite3.connect('RL_experiments.db')
            self.cursorObj = self.con.cursor()
            self.createTable()
            print("Connection is established: Database is created in memory")
        except Error:
            print(Error)

    def createTable(self):
        self.cursorObj.execute(
            "CREATE TABLE experiments(id integer PRIMARY KEY, "
            "experimentType text, "
            "modelUsed text, "
            "episodes real, "
            "iterations real, "
            "episode_results json)"
        )
        self.con.commit()

    def insertRow(self, values):
        self.cursorObj.execute(
            '''INSERT INTO experiments(experimentType, modelUsed, episodes, iterations, episode_results) 
            VALUES(?, ?, ?, ?, ?)''',
            values)
        self.con.commit()
