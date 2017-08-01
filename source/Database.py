import sqlite3
import string
# -*- coding: utf-8 -*-

class Database(object):
    '''
    Opens a database connection to allow for  manipulation of data.

    PARAMETERS
    filename: name of the file, ends in "*.sql"

    ATTRIBUTES
    connection: connection to the database

    METHODS
    query(query): Input some SQL code in query, get the results in a return
    '''

    def __init__(self, file_name):
        '''
        initializer, opens the database connection.
        '''
        self.connection = sqlite3.connect('../sql/' + file_name)

    def query(self, query):
        '''
        Input some SQL code in query, get the results in a return
        '''
        cursor = self.connection.cursor()
        cursor.execute(query)
        if 'INSERT' in query or 'REPLACE' in query or 'UPDATE' in query:
            return self.connection.commit()
        else:
            return cursor.fetchall()

if __name__ == '__main__':
    db = Database('neverforgedData')
    ret = db.query('''
                 SELECT * FROM hitlocation
                 ''')
    b = [a for a in ret]
    for c in b[:2]:
        print('{}'.format(c))
