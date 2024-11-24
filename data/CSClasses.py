import json

import requests

url = "https://classes.cornell.edu/api/2.0/search/classes.json?roster=FA24&subject="

with open("INFOClasses.json", "w") as f:
    try:
        f.write('{\n"classes": { \n')
        url_request = url + "INFO"
        response = requests.get(url_request)
        classes = response.json()
        print(url_request)
        if response.status_code == 200:
            f.write('"INFO": ')
            json.dump(classes["data"]["classes"], f, ensure_ascii=False, indent=4)
            # f.write(", \n")
        else:
            print("Error: ", response.status_code)
        f.write("}\n}")
    except requests.exceptions.RequestException as e:
        print("Error: ", e)
