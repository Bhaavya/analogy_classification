import openai
import json
openai.api_key = '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'
import time 

import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


preface = 'Find the exact analogy and all the sentences that explain it from the "document" below. \
The following is an example of an analogy: "My mom said life is like a box of chocolates. You never know what you\'re gonna get."\
Do not paraphrase, return the exact substring / sentences containing the analogy. \
Return this information in the following JSON format: \
{"analogy": <analogy>}. Return only one analogy even if there are multiple analogies present. \
In case no analogy is found in the text, explicitly return the string "No analogy found." Do not return any other string if no analogy is found. \
===== \
Document: '

# preface = 'Find the source and target concepts in the analogy below. For example, "My mom said life is like a box of chocolates. You never know what you\'re gonna get" has the source "life" and "box of chocolates" as the target. \
# Return this information in the following JSON format: \
# {"source": <source concept>, "target": <target concept> }.\
# In case no source and target concept is found in the text, explicitly return the string "No concept found." Do not return any other string if no concept is found. \
# ===== \
# Analogy: '


'''
Make GPT call using preface above and the text supplied.
'''
def gpt_call(text):
    # Sleep to account for rate limits
    time.sleep(1)

    messages = []
    messages.append(
            {"role": "user", "content": preface + text},
        )

    # print(messages)

    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k", messages=messages
    )

    reply = chat.choices[0].message.content
    return reply

'''
Used to chunk text into token limits.
'''
# def split_text_by_words(text, token_limit, stride):
#     sub_texts = []
#     index = 0

#     strings = text.split()
#     while index < len(strings):
#         sub_strings = strings[index : index + token_limit]
#         sub_text = ' '.join(sub_strings)
#         sub_texts.append(sub_text)
#         index += stride

#     return sub_texts

def split_text_by_words(text, token_limit, stride):
    sub_texts = []
    index = 0

    # strings = text.split()
    strings = enc.encode(text)
   
    
    while index < len(strings):
        sub_strings = strings[index : index + token_limit]
        # sub_text = ' '.join(sub_strings)
        # sub_texts.append(sub_text)
        sub_texts.append(sub_strings)
        index += stride

    actual_sub_texts = []
    for sb in sub_texts:
        actual_sub_texts.append(enc.decode(sb))
    return actual_sub_texts

def split_text_by_chars(text, token_limit, stride):
    sub_texts = []
    index = 0

    while index < len(text):
        sub_text = text[index : index + token_limit]
        sub_texts.append(sub_text)
        index += stride

    return sub_texts

text = "quick brown fox jumped over the lazy dog hallelujah"
print(split_text_by_words(text, 4, 2))

# err_idxs = [977, 1055, 1057]

for split_no in range(1):

    dir = f'splits/split{split_no+2}'

    outfile =  open(f'{dir}/test_output2.jsonl', 'a')
    errfile =  open(f'{dir}/test_error2.jsonl', 'a')
    ridx = 0
    found = False
    with open(f'{dir}/test.jsonl', 'r') as file:
        for line in file:
          
            ridx += 1
            if ridx <= 954:
                continue
            # if ridx not in err_idxs:
            #     continue 
            line = line.strip()
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error loading JSON: {e}")
                continue
            
            print(data['key'])
            # if data['key'] == 'shoes####Theo James':
            #     found = True

            # if not found:
            #     continue 

            token_limit = 16000

            token_char_limit = int(token_limit * 3.5)
            stride_char_limit = int(token_limit * 3.5 * 0.25)

            token_word_limit = int(token_limit * 0.7)
            stride_word_limit = int(token_limit * 0.7 * 0.25)

            texts = split_text_by_words(data['webpage'], token_limit= token_word_limit, stride=stride_word_limit)
            
            for text in texts:
                
                try:
                    reply = gpt_call(text)
                    print(reply)
                    if reply != 'No analogy found.':
                        try:
                            reply_object = json.loads(reply)
                            reply_object['idx'] = ridx
                            json.dump(reply_object, outfile)
                            outfile.write('\n')
                            # break
                        except Exception as e:
                            # print(reply)
                            err_object = {"idx": ridx, "error": str(e) }
                            json.dump(err_object, errfile)
                            errfile.write("\n")
                        
                    
                except Exception as e:
                    # print(reply)
                    err_object = {"idx": ridx, "error": str(e) }
                    json.dump(err_object, errfile)
                    errfile.write("\n")
                    print(e)

    outfile.close()
    errfile.close()