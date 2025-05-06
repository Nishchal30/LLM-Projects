import sqlite3

# connect to sqlite

connection = sqlite3.connect(database="student.db")

#create a curson object to perform CRUD operations on db
cursor = connection.cursor()

#create a table 
table_info = """
CREATE TABLE STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25),
                     SECTION VARCHAR(25), MARKS INT)
"""

cursor.execute(table_info)

# Insert some records in the above table

cursor.execute('''INSERT INTO STUDENT VALUES ('Nishchal', 'Data Science', 'A', 90)''')
cursor.execute('''INSERT INTO STUDENT VALUES ('abc', 'Data Engineering', 'B', 70)''')
cursor.execute('''INSERT INTO STUDENT VALUES ('pqr', 'Data Governance', 'A', 55)''')
cursor.execute('''INSERT INTO STUDENT VALUES ('xyz', 'Data Analysis', 'C', 96)''')
cursor.execute('''INSERT INTO STUDENT VALUES ('lmn', 'Software Engineer', 'D', 98)''')


# Display all the records
print("Inserted records are:\n")
data = cursor.execute('''SELECT * FROM STUDENT''')

for row in data:
    print(row)


# commit the changes in database
connection.commit()
connection.close()