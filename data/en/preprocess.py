import json

file_both = 'test.json' # acl2023

newDialogueId=[]
newDialogueSentence=[]
newDialogueTriplet=[]
temp_cal=[]

with open(file_both, 'r', encoding='utf-8') as f:
    file = json.load(f)
    for each_dialogue in file:

        GT_all=each_dialogue['triplets']
        newDialogueId.append(each_dialogue['doc_id'])
        dialogue=each_dialogue['sentences']
        newDialogueSentence.append(dialogue)

        temp = []
        temp_i=0

        for i in dialogue:
    
           temp_i+=len(i.split(' '))
           temp.append(temp_i)

        for GT in GT_all:

            if(GT[7]!='' and GT[8]!='' and GT[9]!=''):

                temp_cal.append(id)
            
                for i in range(len(temp)):

                    index=temp[i]

                    if(GT[0]<index):
                    
                        target_belong_to= i+1
                        break

                for j in range(len(temp)):

                    index=temp[j]

                    if(GT[2]<index):

                        aspect_belong_to= j+1
                        break
        
                for k in range(len(temp)):

                    index=temp[k]

                    if(GT[4]<index):

                        opinion_belong_to= k+1

                        break

                temp_np = 0
                
                if(len(GT[7].split(' '))>2):

                    temp_np+=1

                if(len(GT[8].split(' '))>2):

                    temp_np+=1

                if(len(GT[9].split(' '))>2):

                    temp_np+=1

                
                if not(target_belong_to==aspect_belong_to==opinion_belong_to) and temp_np>1:

                    if((max(target_belong_to,aspect_belong_to,opinion_belong_to)-min(target_belong_to,aspect_belong_to,opinion_belong_to))==1):
                        print(each_dialogue['doc_id'],dialogue,GT,target_belong_to,aspect_belong_to,opinion_belong_to)
                        newDialogueTriplet.append(1)

print(len(newDialogueTriplet))
        
