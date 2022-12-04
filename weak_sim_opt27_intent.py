

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import torch
import random


gr = []
from database.arangodb import DatabaseConnector
gdb = DatabaseConnector()
db = gdb.connect_db('prodemo')

if db.has_collection('llm_weak_sim_intent'):
        llm_weak_sim_intent = db.collection('llm_weak_sim_intent')
else:
    llm_weak_sim_intent = db.create_collection('llm_weak_sim_intent')


query = 'FOR doc IN  s4_llm_output RETURN doc'
cursor = db.aql.execute(query)
for line in cursor:
    gr.append({'candidate': line['candidate'], 'movie_id': line['movie_id'], 
    'frame_num': line['frame_num'], 'url': line['url']})


model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b", cache_dir="opt27_model/", torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", cache_dir="opt27_model/", use_fast=False)

events_and_intents = []
for event_ in gr:
    intents = []
    for sentence in event_['candidate'].split("."):
        if sentence != "":
            event = sentence
            prompt = '''generate intent of action:
        person is facing the crowd while holding a microphone in order to => make a public statement.
        pesron is sitting at the bar in a sexy dress and waiting for someone in order to => flirt with a man.
        person is an officer that has handcuffed a citizen to himself in order to => arrest the citizen.
        person is carrying her meal to her table in order to => sit down to eat.
        {} in order to => '''.format(event)
            set_seed(64)
            print("EVENT: ",event)
            
            #print(prompt)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
            #intents = []
            for i in range(5):
                
                generated_ids = model.generate(input_ids, do_sample=True, max_length=300)
                intent = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                #print("1    --->",intent)
                intent = intent[0].split(event)[1].replace("\n",".").split(".")[0].split("=>")
                
                if len(intent) > 1:
                    intent = intent[1]
                else:
                    continue
                if len(intent) > 8:
                    intents.append(intent.lstrip())
                    print("2    --->",intent.lstrip())
                #intents.append()
    event_['intents'] = intents
    llm_weak_sim_intent.insert(event_)
    events_and_intents.append(event_)
print(events_and_intents)