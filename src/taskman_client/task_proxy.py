import os
import time
import uuid
from typing import Dict

from taskman_client.RESTProxy import RESTProxy
from taskman_client.host_defs import webtool_host, webtool_port


class TaskManagerProxy(RESTProxy):
    def __init__(self, host, port):
        super(TaskManagerProxy, self).__init__(host, port)

    def task_update(self, run_name, uuid, tpu_name, machine, update_type, msg, job_id=None):
        if run_name == "dontreport":
            return

        data = {
            'run_name': run_name,
            'uuid': uuid,
            'tpu_name': tpu_name,
            'machine': machine,
            'update_type': update_type,
            'msg': msg
        }
        if job_id is not None:
            data['job_id'] = job_id
        return self.post("/task/update", data)

    def task_info_update(self, run_name, uuid, machine, msg, dict_opt=None):
        if run_name == "dontreport":
            return

        data = {
            'run_name': run_name,
            'uuid': uuid,
            'machine': machine,
            'msg': msg
        }
        if dict_opt is not None:
            data.update(dict_opt)
        return self.post("/task/info_update", data)

    def task_pending(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "PENDING"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_start(self, run_name, uuid_var, tpu_name, machine, msg, job_id):
        update_type = "START"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg, job_id)

    def task_complete(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "SUCCESSFUL_TERMINATE"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def task_interrupted(self, run_name, uuid_var, tpu_name, machine, msg):
        update_type = "ABNORMAL_TERMINATE"
        return self.task_update(run_name, uuid_var, tpu_name, machine, update_type, msg)

    def report_number(self, name, value, condition, field):
        machine = get_local_machine_name()
        data = {
            'name': name,
            "number": value,
            "condition": condition,
            "machine": machine,
            "field": field,
        }
        return self.post("/experiment/update", data)

    def get_tpu(self, tpu_condition=None, uuid=None, run_name=""):
        data = {
            "v": tpu_condition,
            "uuid": uuid,
            "run_name": run_name
        }
        return self.post("/task/get_tpu", data)

    def get_num_active_jobs(self, machine):
        data = {
            'machine_name': machine,
        }
        r = self.post("/task/get_num_active_jobs", data)
        return r['num_active_jobs']

    def get_num_pending_jobs(self, machine):
        data = {
            'machine_name': machine,
        }
        r = self.post("/task/get_num_pending_jobs", data)
        return r['num_pending_jobs']

    def pool_job(self, job_name, max_job, machine) -> int:
        data = {
            'max_job': max_job,
            'job_name': job_name,
            'machine': machine,
        }
        r = self.post("/task/pool_job", data, timeout=30)
        return r['job_id']

    def query_job_group_status(self, job_name) -> Dict:
        data = {
            'job_name': job_name,
        }
        r = self.post("/task/query_job_group_status", data)
        return r

    def query_task_status(self, run_name) -> Dict:
        data = {
            'run_name': run_name,
        }
        r = self.post("/task/query_task_status", data)
        return r

    def report_done_and_pool_job(self, job_name, max_job, machine, job_id) -> int:
        data = {
            'max_job': max_job,
            'job_name': job_name,
            'machine': machine,
            'job_id': job_id
        }
        r = self.post("/task/sub_job_done_and_pool_job", data, timeout=30)
        return r['job_id']

    def sub_job_update(self, job_name, machine, update_type, msg, job_id, max_job=None):
        data = {
            'job_name': job_name,
            'machine': machine,
            'update_type': update_type,
            'msg': msg
        }
        if max_job is not None:
            data['max_job'] = max_job
        if job_id is not None:
            data['job_id'] = job_id
        return self.post("/task/sub_job_update", data)

    def cancel_allocation(self, job_name, job_id):
        data = {
            'job_name': job_name,
            'job_id': job_id
        }
        r = self.post("/task/cancel_allocation", data)
        return r['job_id']

    def make_success_notification(self):
        data = {
            'msg': "SUCCESSFUL_TERMINATE",
        }
        print(data)
        r = self.post("/task/make_notification", data)
        return r

    def make_fail_notification(self):
        data = {
            'msg': "ABNORMAL_TERMINATE",
        }
        print(data)
        r = self.post("/task/make_notification", data)
        return r


class TaskProxy:
    def __init__(self, host, port, machine, tpu_name=None, uuid_var=None):
        self.proxy = TaskManagerProxy(host, port)
        self.tpu_name = tpu_name
        self.machine = machine

        if uuid_var is None:
            self.uuid_var = str(uuid.uuid1())
        else:
            self.uuid_var = uuid_var

    def task_pending(self, run_name, msg=None):
        return self.proxy.task_pending(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def task_start(self, run_name, msg=None, job_id=None):
        return self.proxy.task_start(run_name, self.uuid_var, self.tpu_name, self.machine, msg, job_id)

    def task_complete(self, run_name, msg=None):
        return self.proxy.task_complete(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def task_interrupted(self, run_name, msg=None):
        return self.proxy.task_interrupted(run_name, self.uuid_var, self.tpu_name, self.machine, msg)

    def request_tpu(self, run_name, wait=True):
        condition = get_applicable_tpu_condition()

        def request():
            return self.proxy.get_tpu(condition, self.uuid_var, run_name)['tpu_name']

        assigned_tpu = request()

        sleep_time = 5
        while wait and assigned_tpu is None:
            time.sleep(sleep_time)
            if sleep_time < 60:
                sleep_time += 10
            assigned_tpu = request()
        print("Assigned tpu : ", assigned_tpu)
        return assigned_tpu

    def task_info_update(self, run_name, msg=None, dict_data=None):
        return self.proxy.task_info_update(run_name,
                                           self.uuid_var,
                                           self.machine, msg, dict_data)



def get_local_machine_name():
    if os.name == "nt":
        return os.environ["COMPUTERNAME"]
    else:
        return os.uname()[1]


def get_task_proxy(tpu_name=None, uuid_var=None):
    machine = get_local_machine_name()
    return TaskProxy(webtool_host, webtool_port, machine, tpu_name, uuid_var)


def get_task_manager_proxy():
    return TaskManagerProxy(webtool_host, webtool_port)


def assign_tpu_anonymous(wait=True):
    print("Auto assign TPU")
    condition = get_applicable_tpu_condition()

    def request_tpu():
        return get_task_manager_proxy().get_tpu(condition)['tpu_name']

    assigned_tpu = request_tpu()

    sleep_time = 5
    while wait and assigned_tpu is None:
        time.sleep(sleep_time)
        if sleep_time < 300:
            sleep_time += 10
        assigned_tpu = request_tpu()
    print("Assigned tpu : ", assigned_tpu)
    return assigned_tpu


def get_applicable_tpu_condition():
    machine = get_local_machine_name()
    if machine in ["lesterny", "instance-5", "us-1"]:
        condition = "v2"
    elif machine == "instance-3":
        condition = "v3"
    elif machine == "instance-4":
        condition = "v3"
    else:
        print("Warning TPU condition not indicated")
        condition = None
    return condition


if __name__ == "__main__":
    import numpy
    arr = numpy.array([0.1,0.242])
    value = arr[0]

    TaskManagerProxy(webtool_host, webtool_port).get_num_active_jobs("lesterny")
