import json

import requests

url = "https://classes.cornell.edu/api/2.0/search/classes.json?roster=FA24&subject="

with open("subjects.json", "r") as subjects:
    file = json.load(subjects)
    subjects = file["data"]["subjects"]

    with open("classes.json", "w") as f:
        try:
            f.write('{\n"classes": { \n')
            for s in subjects:
                url_request = url + s["value"]
                response = requests.get(url_request)
                classes = response.json()
                print(url_request)
                if response.status_code == 200:
                    f.write('"' + s["value"] + '": ')
                    json.dump(
                        classes["data"]["classes"], f, ensure_ascii=False, indent=4
                    )
                    f.write(", \n")
                else:
                    print("Error: ", response.status_code)
            f.write("}\n}")
        except requests.exceptions.RequestException as e:
            print("Error: ", e)
