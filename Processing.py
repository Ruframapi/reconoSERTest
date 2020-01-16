"""
File Processing without Spark. Just to take a look.

"""

import sys
import os
import json
import re
import random
import nltk
import datetime
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.util import skipgrams


def main():
    
    filepath = sys.argv[1]
    processing_type = sys.argv[2] # "user_count, user_filter, user_filter_reviews, user_filter_with_reviews_2"
    filter_number = int(sys.argv[3]) 

    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    if processing_type == "user_count":
        user_count(filepath)
    elif processing_type == "user_filter":
        user_filter(filepath,filter_number)
    elif processing_type == "user_filter_reviews":
        user_filter_with_reviews(filepath,filter_number)      
    elif processing_type == "user_filter_reviews_2":
        user_filter_with_reviews_2(filepath,filter_number)       
    else:
        print("Wrong second argument (Processing type)")
        sys.exit()

def user_filter_with_reviews(filepath,filter_number):
    """For those user with more than -filter_number- reviews the comments are extracted"""

    filename = "UserFilter"+ str(filter_number) + ".json"
    with open(filename, 'r', encoding="utf8", errors='ignore') as fp:
        data=fp.read()

    dict_users = json.loads(data)
    filename = "UserFilter"+ str(filter_number) + "-Reviews.json"

    with open(filepath, encoding="utf8", errors='ignore') as fp:
        cnt = 0
        excnt = 0
        text_to_write = ""

        #for i in xrange(6):
        #    f.next()

        for line in fp:
            excnt += 1
            if excnt % 100000 == 0:
                print(excnt)

            if "review/userId" in line:
                actualuser = line.replace('review/userId:', '').strip()
            
            if "review/text:" in line:
                if actualuser in dict_users:
                    review = cleanhtml(line.replace('review/text:', '').strip())
                    review = tokenize(review)
                    text_to_write = text_to_write + actualuser + " || " + json.dumps(review) + "\n"
                    cnt += 1

                    if cnt == 10000:
                        print("cnt {} excnt {}".format(cnt,excnt))
                        with open(filename, 'a') as fw:
                            fw.write(text_to_write)
                        cnt = 1
                        text_to_write = ""
                        
def user_filter_with_reviews_2(filepath,filter_number):
    """ Found filter with reviews 2 """
    filename = "UserFilter"+ str(filter_number) + "-Random3.json"
    with open(filename, 'r', encoding="utf8", errors='ignore') as fp:
        data=fp.read()
    
    dict_users = json.loads(data)
    dict_total_training = {}
    dict_total_testing = {}

    cnt = 1
    print(datetime.datetime.now().time())
    filename = "UserFilter"+ str(filter_number) + "-Reviews.json"
    with open(filename, encoding="utf8", errors='ignore') as fp:
        for line in fp:
            list_line=line.split("||")
            user = list_line[0].strip()
            if user in dict_users:
                if user not in dict_total_training:
                    dict_total_training[user] = {"reviews": []}
                if user not in dict_total_testing:
                    dict_total_testing[user] = {"reviews": []}

                if random.random() < 0.5:
                    word_list_training = json.loads(list_line[1].strip())
                    dict_total_training[user]["reviews"].extend(word_list_training)
                else:
                    word_list_testing = json.loads(list_line[1].strip())
                    dict_total_testing[user]["reviews"].append(word_list_testing)
                
                #dict_total[user]["pos"].extend(pos_tagger(word_list))
            
            cnt += 1
            if cnt % 100000 == 0:
                print(datetime.datetime.now().time())

    list_total_training = []
    for key in dict_total_training:
        dictdemo = {}
        dictdemo["user"] = key
        dictdemo["reviews"] = dict_total_training[key]["reviews"]
        list_total_training.append(dictdemo)
        
    dict_total_training = {}

    filename = "UserFilter"+ str(filter_number) + "-Training-Random3.json"
    with open(filename, 'w') as fp:
        json.dump(list_total_training, fp,  indent=4)

    list_total_training = []


    list_total_testing = []
    for key in dict_total_testing:
        dictdemo = {}
        dictdemo["user"] = key
        dictdemo["reviews"] = dict_total_testing[key]["reviews"]
        list_total_testing.append(dictdemo)

    dict_total_testing = {}

    filename = "UserFilter"+ str(filter_number) + "-Testing-Random3.json"
    with open(filename, 'w') as fp:
        json.dump(list_total_testing, fp,  indent=4)

def user_filter(filepath,filter_number):
    """ Found users with more than -filter_number- reviews """
    with open(filepath, 'r', encoding="utf8", errors='ignore') as fp:
        data=fp.read()

    dict_users = json.loads(data)
    dict_users_filter = dict(filter(lambda elem: elem[1] >= filter_number ,dict_users.items()))

    len10p = round(len(dict_users_filter)*0.03)
    
    dict_users_filter_rand = dict(random.sample(dict_users_filter.items(), len10p))

    filename = "UserFilter"+ str(filter_number) + "-Random3.json"
    with open(filename, 'w') as fp:
        json.dump(dict_users_filter_rand, fp,  indent=4)

def user_count(filepath):
    """ Count the number of reviews per user"""
    bag_of_users = {}
    with open(filepath, encoding="utf8", errors='ignore') as fp:
        cnt = 0
        for line in fp:
            if "review/userId" in line:
                readuser = line.replace('review/userId:', '').strip()
                record_user_cnt(readuser, bag_of_users)
                cnt += 1
                
            if cnt == 100000:
                print("line {}".format(line))
                cnt = 1
                #break
    sorted_users = order_bag_of_users(bag_of_users, desc=True)
    
    with open('userReviewCount.json', 'w') as fp:
        json.dump(sorted_users, fp,  indent=4)


def order_bag_of_users(bag_of_users, desc=False):
    """Order by number of reviews"""
    users = [(user, cnt) for user, cnt in bag_of_users.items()]
    users_sort = sorted(users, key=lambda x: x[1], reverse=desc)
    print("User with more reviews {}".format(users_sort[:10]))
    return dict(users_sort)


def record_user_cnt(user, bag_of_users):
    """Record the reviews count """
    if user != '':
        if user in bag_of_users:
            bag_of_users[user] += 1
        else:
            bag_of_users[user] = 1

def cleanhtml(raw_html):
    """Delete HTML Tags"""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def pos_tagger(s):
    """POS"""
    return [i[1] for i in nltk.pos_tag(s)]

def tokenize(s):
    """Tokenizer"""
    s = s.lower()
    token = TweetTokenizer()
    return token.tokenize(s)

if __name__ == '__main__':
    main()