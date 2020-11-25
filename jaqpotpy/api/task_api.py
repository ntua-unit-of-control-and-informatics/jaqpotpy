import requests

task_path = "task"


def get_task(baseurl, api_key, taskid):
    uri = baseurl + task_path + "/" + taskid
    h = {'Content-Type': 'application/x-www-form-urlencoded',
         'Accept': 'application/json',
         'Authorization': "Bearer " + api_key}
    r = requests.get(uri, headers=h)
    return r.json()
