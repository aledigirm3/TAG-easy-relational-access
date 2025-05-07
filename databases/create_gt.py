import json
import os


query_without_join = {
    'superhero': [745, 746, 747, 790, 791, 802, 803, 804, 805, 832, 833, 836, 837, 838],
    'thrombosis_prediction': [1149, 1150, 1151, 1152, 1162, 1163, 1176, 1177, 1178, 1179, 1182, 1186, 1187, 1188, 1189, 1190, 1191, 1196, 1197, 1198, 1199, 1200, 1201, 1206, 1210, 1290],
    'student_club': [1325, 1341, 1342, 1343, 1344, 1345, 1346, 1361, 1362, 1363, 1377, 1378, 1379, 1380, 1391, 1392, 1393, 1397, 1400, 1402, 1406, 1407, 1408, 1409, 1423, 1424, 1425, 1433, 1434, 1435, 1442, 1443, 1444, 1445, 1446]
}

filtered_query = {
    "superhero": [718, 720, 721, 725, 727, 731, 732, 741, 748, 750],
    "thrombosis_prediction": [1153, 1155, 1184, 1207, 1222, 1226, 1227, 1242, 1260, 1261],
    "student_club": [1312, 1313, 1314, 1320, 1323, 1326, 1334, 1336, 1347, 1353]
}


dev = json.load(open("dev.json"))

for database in filtered_query.keys():
    questions = []

    ids = set(query_without_join[database] + filtered_query[database])
    
    for element in dev:
        question_id = element["question_id"]

        if question_id in ids:
            questions.append(element)

    with open(os.path.join("..","gt", f"{database}.json"), "w") as single_gt:
        json.dump(questions, single_gt, indent=4)

