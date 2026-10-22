import random
from langchain_core.runnables import RunnableLambda

INPUT = {
    "failed_rate": 0.5
}

RUN_TIMES = 10

def rand_fail(d):
    if random.random() < d["failed_rate"]:
        print("[debug] failed")
        raise Exception("failed")
    print("[debug] success")
    return "success"

chain = RunnableLambda(rand_fail).with_retry(stop_after_attempt=2)

count_success = 0
for i in range(RUN_TIMES):
    try:
        ret = chain.invoke(INPUT)
        print(ret)
        count_success += 1
    except Exception as e:
        print(e)
success_rate = count_success/RUN_TIMES*100
print(f"{success_rate}% success ({count_success}/{RUN_TIMES})")
