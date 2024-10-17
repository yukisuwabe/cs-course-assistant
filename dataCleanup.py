import json

with open("classes.json", "r") as classes:
    file = json.load(classes)
    f = open("data.txt", "w")
    for subjects in file["classes"]:
        # print(file["classes"][subjects])
        for c in file["classes"][subjects]:
            # print(c["subject"] + " " + c["catalogNbr"])
            f.write("class: " + c["subject"] + " " + c["catalogNbr"] + "\n")
            f.write("class title: " + c["titleLong"] + "\n")
            if c["description"]:
                f.write("description: " + c["description"] + "\n")
            if c["catalogOutcomes"]:
                f.write("outcome: ")
                for outcome in c["catalogOutcomes"]:
                    f.write(outcome + ", ")
                f.write("\n")
            if c["catalogDistr"]:
                f.write("distribution categories: " + c["catalogDistr"] + "\n")
            f.write("\n")
