from langchain_core.runnables import RunnableLambda
from langchain_core.callbacks import BaseCallbackHandler

class ShowError(BaseCallbackHandler):
    def on_chain_error(self, error, *, run_id, parent_run_id = None, **kwargs):
        print("[error]", error)
        return # super().on_chain_error(error, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
callback_error = ShowError()

def func1(*args):
    print("main")
    raise Exception("main failed.")

def func2(*args):
    print("down lv1")
    raise Exception("down1 failed.")

chain1 = RunnableLambda(func1)
chain2 = RunnableLambda(func2)
chain3 = RunnableLambda(lambda _: "down lv2")

chain = chain1.with_fallbacks([chain2, chain3])

ret = chain.invoke("", config={"callbacks": [callback_error]})
print(ret)
