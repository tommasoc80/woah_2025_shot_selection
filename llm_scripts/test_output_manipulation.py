import json
import re

#response = "I cannot classify the text as misogynous or not misogynous. The text contains hate speech and discrimination towards racial and ethnic groups."
#response = "{'response' : [ {'text': '''Well then good because someone has to knock her ass back in line when she acts like a woman and posts bullshit about her (((based minorities))). If someone is not white they are NOT one of us.''', 'label': 1}]"
response = {'response' : [    {'text': '#USA #Texas #Islam #Muslims #Islamization #ShariaLaw Sharia in Texas - Evil Muslim parents tried to murder their daughter with boiling oil after refusing arranged marriage [URL]', 'label': 0}]}


if str(response).startswith("{"):
    #label_extracted = re.search("\'label\': (\w+)", response)
    keyword = "\'label\':"
    before_keyword, keyword, after_keyword = str(response).partition(keyword)
    print(after_keyword.replace("}]","").replace("}", ""))
else:
    print("Refused")
