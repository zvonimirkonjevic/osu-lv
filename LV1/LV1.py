'''

# example of using arithmetic operators
x = 23
print(x)
x = x + 7
print(x)

# example of using comparison operators
x = 23
y = x > 10
print(y)

# example of control statements

# if-else statement
x = 23
if x < 10:
    print("x is smaller than 10")
else:
    print("x is larger than 10")

# for and while statements
i = 5
while i > 0:
    print(i)
    i -= 1
print("loop is finished")

for i in range(0, 5):
    print(i)

# example of working with lists
lstEmpty = []
lstFriend = ['Marko' , 'Luka' , 'Pero' ]
lstFriend.append('Ivan')
print(lstFriend [0])
print(lstFriend [0:1:2])
print(lstFriend [:2])
print(lstFriend [1:])
print(lstFriend [1:3])

a = [1 , 2 , 3]
b = [4 , 5 , 6]
c = a + b
print(c)
print(max(c))
c[0] = 7
c.pop()
for number in c:
    print('List number', number)
print('Done !')

# example of working with strings
fruit = 'banana'
index = 0
count = 0

while index < len(fruit):
    letter = fruit[index]
    if letter == 'a':
        count += 1
    print(letter)
    index += 1

print(count)
print (fruit[0:3])
print (fruit[0:])
print (fruit[2:6:1])
print (fruit[0:-1])

line = 'Dobrodosli u nas grad'
if(line.startswith ('Dobrodosli')):
    print('Prva rijec je Dobrodosli')
elif (line.startswith('dobrodosli')):
    print ('Prva rijec je dobrodosli')
line.lower()
print(line)
data = 'From: pero@yahoo.com'
atpos = data.find('@')
print(atpos)

# example of working with touples
letters = ('a', 'b', 'c', 'd', 'e')
numbers = (1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10 , 11)
mixed = (1 , 'Hello', 3.14 )
print (letters[0])
print (letters[1:4])
for letter in letters :
    print(letter)

# example of working with dictionaries
hr_num = {'jedan':1 , 'dva':2 , 'tri':3}
print( hr_num )
print( hr_num ['dva'])
hr_num['cetiri'] = 4
print (hr_num)

'''

'''
# Task 1
hours_worked = int(input().split('h')[0])
hour_wage = float(input())

print(f'total: {hours_worked*hour_wage} eura')

'''

'''

# Task 2
try:
    grade = float(input())
    if(grade < 0.0 or grade > 1.0):
        print("An exception has ocurred.")
    else:
        if grade >= 0.9:
            print('A')
        elif grade >=0.8:
            print('B')
        elif grade>=0.7:
            print('C')
        elif grade>=0.6:
            print('D')
        else:
            print('D')
except: 
    print("An exception has ocurred.")

'''

'''

# Task 3
numbers = []
while True:
    user_input = input()
    if user_input == 'Done':
        break
    else:
        try:
            number = int(user_input)
            numbers.append(number)
        except:
            continue
        size = len(numbers)
        average = sum(numbers) / len(numbers)
        min_value = min(numbers)
        max_value = max(numbers)

print(f'Average: {average}, Size: {size}, Minimal value: {min_value}, Maximum value: {max_value}')

'''

'''

# Task 4
file = open('song.txt')
words = []
dict = {}
for line in file:
    words_in_line = line.rstrip().split(' ')
    words += words_in_line
unique_words = list(set(words))
for unique_word in unique_words:
    count = 0
    for word in words:
        if word == unique_word:
            count += 1
    dict[unique_word] = count
one_appearance_words = []
for k,v in dict.items():
    if(v == 1):
        one_appearance_words.append(k)
print(len(one_appearance_words))

'''

'''

# Task 5
ham_word_count = 0
ham_count = 0
spam_word_count = 0
spam_count = 0
spam_with_question_mark = 0
file = open('SMSSpamCollection.txt')
for line in file:
    line = line.replace('	', ' ')
    email_category = line.rstrip().split(' ')[0]
    if email_category == 'ham':
        ham_count += 1
        line = line.rstrip().split(' ')
        line.pop(0)
        ham_word_count += len(line)

    if email_category == 'spam':
        spam_count += 1
        line = line.rstrip().split(' ')
        line.pop(0)
        if line[-1].endswith('?'):
            spam_with_question_mark += 1
        spam_word_count += len(line)

print(ham_word_count/ham_count)
print(spam_word_count/spam_count)
print(spam_with_question_mark)

'''