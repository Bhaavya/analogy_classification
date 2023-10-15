import json

count = 0

words = set(["male", "female", "transgender", 'women', 'woman', "girl", "boy", "lady", "body" \
         "heterosexual",  'gay', 'lesbian', 'homo', "bisexual", \
         'christian', 'jew', "hindu", "buddhist", "atheist", \
        'muslim',  'black', 'asian', 'white', 'latino',  'hawaii', 'mexico', "mexican", 'hispanic',  'thai',  \
    'mental',
    "vagina", "breast", "butt", "testicle", "penis", "cock", "nipple", "vulva"]
)


import tweetnlp
hate_model = tweetnlp.load_model('hate') 
of_model = tweetnlp.load_model('offensive')  # Or `model = tweetnlp.Offensive()` 

ii = 0
fm = open("offensive_model.txt", "w")
fi = open("offensive_words.txt", "w")

h = 0
o = 0
i = 0
io = 0
ho = 0
hi = 0
count = 0

for f in ["train", "test"]:
    print(f)
    file1 = open(f'data/extraction/split1/{f}.jsonl', 'r')
    while True:
        ii += 1
        print(ii, end="\r")
        # Get next line from file
        line = file1.readline()
        if not line:
            print("Ended")
            break
        json_object = json.loads(line)
        try:
            identity = False
            for word in words:
                if word in json_object["analogy"].lower():
                    identity = True
                    # print(json_object["key"])
                    # print(json_object["analogy"])
                    
            
            result1 = hate_model.predict(json_object["analogy"].lower())
            result2 = of_model.predict(json_object["analogy"].lower())

            if result1["label"] == "HATE":
                h += 1
            if identity:
                i += 1
            if result2["label"] == "offensive":
                o += 1

            if identity and result2["label"] == "offensive":
                io += 1

            if identity and result1["label"] == "HATE":
                ho += 1
            
            
            if result1["label"] == "HATE" or result2["label"] == "offensive" or identity:
                count += 1
            if (result1["label"] == "HATE" or result2["label"] == "offensive") and not identity:
                fm.write(json_object["analogy"].lower() + "\n")
            elif identity and not (result1["label"] == "HATE" or result2["label"] == "offensive"):
                fi.write(json_object["analogy"].lower() + "\n")
                # count+=1
        except Exception as e:
            print(line)
            print(e)
    file1.close()
        
    print(f"H: {h},  O: {o}, I: {i}, count: {count}")
    print(f"Both I and O: {io}")

fm.close()
fi.close()


# model = tweetnlp.Classifier("cardiffnlp/twitter-roberta-base-hate-latest")

# tasks=['hate-latest','offensive']
# MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

# tokenizer = AutoTokenizer.from_pretrained(MODEL)