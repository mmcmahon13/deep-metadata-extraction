import simstring
import os

# create the databases for Simstring to use for dicitonary matching during preprocessing

# create name database
db = simstring.writer('dicts' + os.sep + 'people.db')
with open('dicts' + os.sep + 'chinese_only.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'english_only.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'frequent_last_names.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'shared.txt') as f:
    for word in f:
        db.insert(word.strip())
db.close()

# create place database
db = simstring.writer('dicts' + os.sep + 'places.db')
with open('dicts' + os.sep + 'city_full.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'country_full.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'region_full.txt') as f:
    for word in f:
        db.insert((word.strip()))
db.close()

# create department database
db = simstring.writer('dicts' + os.sep + 'departments.db')
with open('dicts' + os.sep + 'department_full.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'department_keywords.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'faculty_full.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'faculty_keywords.txt') as f:
    for word in f:
        db.insert(word.strip())
db.close()

# create university database
db = simstring.writer('dicts' + os.sep + 'universities.db')
with open('dicts' + os.sep + 'university_full.txt') as f:
    for word in f:
        db.insert(word.strip())
with open('dicts' + os.sep + 'university_keywords.txt') as f:
    for word in f:
        db.insert(word.strip())
db.close()


db = simstring.reader('dicts' + os.sep + 'people.db')
print("testing person database")
print(db.retrieve('aaron'))
print(db.retrieve('abe'))

db = simstring.reader('dicts' + os.sep + 'places.db')
print("testing place database")
print(db.retrieve('boston'))
print(db.retrieve('china'))

db = simstring.reader('dicts' + os.sep + 'departments.db')
print("testing department database")
print(db.retrieve('medical'))
print(db.retrieve('association'))

db = simstring.reader('dicts' + os.sep + 'universities.db')
print("testing university database")
print(db.retrieve('university'))






